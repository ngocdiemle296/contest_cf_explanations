"""
Black-box Oracle Module
Training a CatBoost regressor to predict a running case's remaining execution time, returning the (activity, resource) pair with the lowest predicted time.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable

import torch
import numpy as np

from baseline.revised_plus.data import event_log


# Joint action space  A^c = A × R

def joint_actions() -> List[Tuple[str, str]]:
    return [(a, r) for a in event_log.ACTIVITIES for r in event_log.RESOURCES]

def n_joint_actions() -> int:
    return len(event_log.ACTIVITIES) * len(event_log.RESOURCES)


def joint_action_idx(activity: str, resource: str) -> int:
    return event_log.ACT2IDX[activity] * len(event_log.RESOURCES) + event_log.RES2IDX[resource]


def idx_to_joint_action(idx: int) -> Tuple[str, str]:
    if not event_log.RESOURCES:
        return ("", "")
    act_idx = idx // len(event_log.RESOURCES)
    res_idx = idx % len(event_log.RESOURCES)
    return (event_log.IDX2ACT.get(act_idx, event_log.ACTIVITIES[0]),
            event_log.IDX2RES.get(res_idx, event_log.RESOURCES[0]))


# Synthetic / boundary activities that exist only to mark "this is the
# beginning/ending of a case" in the event log.
NON_RECOMMENDABLE_ACTIVITIES = {"START", "END"}
NON_RECOMMENDABLE_RESOURCES = {"System"}



def non_recommendable_action_indices() -> List[int]:
    """
    Joint-action indices that should never be selectable as a
    recommendation or counterfactual target. Two independent conditions,
    each sufficient on its own:
      (a) activity is a synthetic boundary marker (START/END) — not a
          real process step, regardless of resource.
      (b) resource is System (or any other automated/non-human resource
          in NON_RECOMMENDABLE_RESOURCES) 
    """
    bad_act_idxs = {
        event_log.ACT2IDX[a] for a in NON_RECOMMENDABLE_ACTIVITIES if a in event_log.ACT2IDX
    }
    bad_res_idxs = {
        event_log.RES2IDX[r] for r in NON_RECOMMENDABLE_RESOURCES if r in event_log.RES2IDX
    }
    n_res = len(event_log.RESOURCES)
    return [
        a_idx * n_res + r_idx
        for a_idx in range(len(event_log.ACTIVITIES))
        for r_idx in range(n_res)
        if a_idx in bad_act_idxs or r_idx in bad_res_idxs
    ]


@lru_cache(maxsize=None)
def feasible_joint_action_indices(min_observations: int = 1) -> torch.Tensor:
    """
    Restricted output space for LocalSurrogate (and anything else that
    wants to classify over "real" actions only).

    n_joint_actions() = |ACTIVITIES| x |RESOURCES| counts every possible
    combination, including thousands that never occur in the data (e.g. an
    intake-only resource paired with a validation activity).
    """
    counts = getattr(event_log, "RESOURCE_ACTIVITY_COUNTS", None)
    if counts is None:
        raise RuntimeError(
            "event_log.RESOURCE_ACTIVITY_COUNTS is not built yet. Call "
            "event_log.build_resource_activity_matrix(cases, ...) before "
            "constructing a LocalSurrogate with a restricted output space "
            "(this also requires the RESOURCE_ACTIVITY_COUNTS-caching patch "
            "to build_resource_activity_matrix)."
        )
    mat = torch.as_tensor(counts) >= min_observations  
    n_act = len(event_log.ACTIVITIES)
    n_res = len(event_log.RESOURCES)

    if mat.shape == (n_act, n_res):
        pass  # already matching (activity, resource) format
    elif mat.shape == (n_res, n_act):
        # Observed orientation in practice: (resource, activity). Transpose
        # so downstream flattening can assume (activity, resource) either way.
        mat = mat.T
    else:
        raise ValueError(
            f"RESOURCE_ACTIVITY_COUNTS shape {tuple(mat.shape)} matches neither "
            f"(n_activities={n_act}, n_resources={n_res}) nor its transpose "
            f"(n_resources, n_activities) — the act_idx*n_res+res_idx flattening "
            f"below assumes one of these two layouts. Check how "
            f"event_log.build_resource_activity_matrix() constructs this matrix."
        )

    flat_idx = mat.reshape(-1).nonzero(as_tuple=False).squeeze(-1)  # act_idx*n_res+res_idx
    bad = set(non_recommendable_action_indices())
    feasible = sorted(int(i) for i in flat_idx.tolist() if int(i) not in bad)

    if not feasible:
        raise RuntimeError(
            "feasible_joint_action_indices() computed an empty feasible set — "
            "check RESOURCE_ACTIVITY_MATRIX / min_observations."
        )
    return torch.as_tensor(feasible, dtype=torch.long)


# Transition system (feasible next activities + resources from event log)

class TransitionSystem:
    """Encodes feasible next activities and resources from the event log."""

    def __init__(self, history_len: int = 5) -> None:
        self.next_acts: Dict[str, List[str]] = {}
        self.resources_by_act: Dict[str, List[str]] = {}
        self.history_len = history_len

    @staticmethod
    def from_cases(cases: List[event_log.Case], history_len: int = 5) -> "TransitionSystem":
        ts = TransitionSystem(history_len=history_len)
        next_acts_set: Dict[Tuple[str, ...], set] = {}
        resources_by_act_set: Dict[str, set] = {}

        for case in cases:
            evs = case.events
            for i, ev in enumerate(evs):
                # Tracking which resources can do which activities globally
                resources_by_act_set.setdefault(ev.activity, set()).add(ev.resource)

                if i + 1 < len(evs):
                    nxt = evs[i + 1].activity

                    if nxt not in NON_RECOMMENDABLE_ACTIVITIES:
                        start_idx = max(0, i - ts.history_len + 1)
                        history_slice = evs[start_idx : i + 1]

                        history_key = tuple(e.activity for e in history_slice)

                        next_acts_set.setdefault(history_key, set()).add(nxt)

        ts.next_acts = {k: sorted(list(v)) for k, v in next_acts_set.items()}
        ts.resources_by_act = {k: sorted(list(v)) for k, v in resources_by_act_set.items()}
        return ts

    def feasible_pairs(self, history_activities: List[str]) -> List[Tuple[str, str]]:
            """
            Accepts a list of past activities (any length), truncates to the last 5 (window_size), 
            and extracts the valid (activity, resource) joint actions.
            """
        
            recent_history = tuple(history_activities[-self.history_len:]) # Ensure we only look at the most recent 5 events
            
            next_acts = self.next_acts.get(recent_history)
            
            backoff_len = len(recent_history) - 1
            while not next_acts and backoff_len > 0:
                shortened_history = recent_history[-backoff_len:]
                next_acts = self.next_acts.get(shortened_history)
                backoff_len -= 1
                
            if not next_acts:
                next_acts = [
                    a for a in self.resources_by_act.keys()
                    if a not in NON_RECOMMENDABLE_ACTIVITIES
                ]

            pairs: List[Tuple[str, str]] = []
            for act in next_acts:
                res_list = self.resources_by_act.get(act, [])
                for res in res_list:
                    pairs.append((act, res))
            return pairs
    

    def to_dict(self) -> Dict:
        """Converts tuple keys to delimited strings so JSON can serialize them."""
        # Join tuple elements with a comma, e.g., ('A', 'B') -> "A,B"
        stringified_next_acts = {
            ",".join(k): v for k, v in self.next_acts.items()
        }
        
        return {
            "next_acts": stringified_next_acts,
            "resources_by_act": self.resources_by_act,
        }

    @staticmethod
    def from_dict(data: Dict, history_len: int = 5) -> "TransitionSystem":
        """Reconstructs the system, restoring string keys back into tuples."""
        ts = TransitionSystem(history_len=history_len)
        
        raw_next_acts = data.get("next_acts", {})
        
        # Split the comma-separated strings back into structural tuples
        ts.next_acts = {
            tuple(k.split(",")): list(v) for k, v in raw_next_acts.items()
        }
        
        ts.resources_by_act = {
            k: list(v) for k, v in data.get("resources_by_act", {}).items()
        }
        return ts


# Black-box model (CatBoost regressor)

class CatBoostPPM:
    """
    CatBoost-based recommender.
    Trained to predict execution time for (σ_t, a^c_{t+1}).
    Recommendation = feasible (activity, resource) pair with smallest predicted time.
    """

    def __init__(self, transition_system: TransitionSystem, max_trace_len: int = 20,
                 thread_count: Optional[int] = 4,
                 depth: int = 6, learning_rate: float = 0.1,
                 iterations: int = 300, variance_power: float = 1.5):
        self.transition_system = transition_system
        self.max_trace_len = max_trace_len
        self.thread_count = thread_count
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.variance_power = variance_power
        self.model = None
        self.cv_summary: Optional[Dict] = None

    def _ensure_model(self) -> None:
        if self.model is None:
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError(
                    "catboost is required for the CatBoost black-box. "
                    "Install with `pip install catboost`."
                ) from exc
            self.model = CatBoostRegressor(
                loss_function=f"Tweedie:variance_power={self.variance_power}",
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_seed=42,
                verbose=False,
                thread_count=self.thread_count,
            )

    def _build_feature_vector(
        self,
        trace_mat: np.ndarray,
        length: int,
        act_idx: int,
        res_idx: int,
        extra_feats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        last_event = trace_mat[length - 1]
        length_norm = float(length) / float(self.max_trace_len)
        act_norm = float(act_idx) / float(max(1, len(event_log.ACTIVITIES) - 1))
        res_norm = float(res_idx) / float(max(1, len(event_log.RESOURCES) - 1))

        features = np.concatenate([last_event, np.array([length_norm, act_norm, res_norm], dtype=np.float32)])
        if extra_feats is not None:
            features = np.concatenate([features, extra_feats])
        return features

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._ensure_model()
        self.model.fit(features, targets)

    def predict_time(self, features: np.ndarray) -> np.ndarray:
        self._ensure_model()
        return np.asarray(self.model.predict(features), dtype=np.float32)

    def _last_activity_from_trace(self, trace_mat: np.ndarray, length: int) -> str:
        last_event = trace_mat[length - 1]
        act_idx = int(np.argmax(last_event[:len(event_log.ACTIVITIES)]))
        return event_log.IDX2ACT.get(act_idx, event_log.ACTIVITIES[0])

    def predict_time_for_pair(
        self,
        trace_mat: np.ndarray,
        length: int,
        activity: str,
        resource: str,
    ) -> float:
        """
        KPI lookup for one specific (activity, resource) pair
        """
        feat = self._build_feature_vector(
            trace_mat, length, event_log.ACT2IDX[activity], event_log.RES2IDX[resource]
        )
        time_pred = self.predict_time(feat[np.newaxis, :])
        return float(time_pred[0])

    def recommend(
        self,
        trace_mat: np.ndarray,
        length: int,
        return_time: bool = False,
    ) -> Tuple[str, str]:
        history_acts = []
        start_lookback = max(0, length - self.max_trace_len) 
        for step_idx in range(start_lookback, length):
            event_vec = trace_mat[step_idx]
            act_idx = int(np.argmax(event_vec[:len(event_log.ACTIVITIES)]))
            history_acts.append(event_log.IDX2ACT.get(act_idx, event_log.ACTIVITIES[0]))
            
        pairs = self.transition_system.feasible_pairs(history_acts)
        if not pairs:
            pairs = [
                (a, r) for (a, r) in joint_actions()
                if a not in NON_RECOMMENDABLE_ACTIVITIES
            ]

        feat_list = []
        for act, res in pairs:
            feat_list.append(
                self._build_feature_vector(trace_mat, length, event_log.ACT2IDX[act], event_log.RES2IDX[res])
            )
        feats = np.stack(feat_list)
        times = self.predict_time(feats)
        best_idx = int(np.argmin(times))

        if return_time:
            return pairs[best_idx], float(times[best_idx])
        return pairs[best_idx]


    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if lengths is None:
            lengths = torch.tensor([x.size(1)])
        trace_mat = x[0].cpu().numpy()
        length = int(lengths[0].item())
        return self.recommend(trace_mat, length)

    @torch.no_grad()
    def predict_idx(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> int:
        act, res = self.predict(x, lengths)
        return joint_action_idx(act, res)

    @torch.no_grad()
    def predict_idx_batch(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        out = []
        for i in range(x.size(0)):
            trace_mat = x[i].cpu().numpy()
            length = int(lengths[i].item())
            act, res = self.recommend(trace_mat, length)
            out.append(joint_action_idx(act, res))
        return torch.tensor(out, dtype=torch.long)

    def save(self, path: Path) -> None:
        self._ensure_model()
        model_path = path / "black_box.cbm"
        ts_path = path / "transition_system.json"
        self.model.save_model(str(model_path))
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(self.transition_system.to_dict(), f)
        cv_summary = getattr(self, "cv_summary", None)
        if cv_summary is not None:
            with open(path / "black_box_cv_summary.json", "w", encoding="utf-8") as f:
                json.dump(cv_summary, f, indent=2)

    @staticmethod
    def load(path: Path) -> "CatBoostPPM":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for the CatBoost black-box. "
                "Install with `pip install catboost`."
            ) from exc

        model = CatBoostRegressor()
        model.load_model(str(path / "black_box.cbm"))
        with open(path / "transition_system.json", "r", encoding="utf-8") as f:
            ts = TransitionSystem.from_dict(json.load(f))
        bb = CatBoostPPM(transition_system=ts)
        bb.model = model
        cv_summary_path = path / "black_box_cv_summary.json"
        bb.cv_summary = None
        if cv_summary_path.exists():
            with open(cv_summary_path, "r", encoding="utf-8") as f:
                bb.cv_summary = json.load(f)
        return bb


# Latent-space oracle wrapper (used by surrogate & CF search)

class LatentOracle:
    """
    Wraps the black-box PPM so it can be queried with a latent vector z.
    decode_fn: z → σ matrix (event sequence)
    """

    def __init__(
        self,
        black_box: CatBoostPPM,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        seq_len: int,
        device: torch.device,
    ):
        self.black_box  = black_box
        self.decode_fn  = decode_fn
        self.seq_len    = seq_len
        self.device     = device

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)  → action_idx (B,) int64
        Decodes to sequence, queries black-box.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)
        with torch.no_grad():
            seq = self.decode_fn(z)   # (B, T, event_dim)
            lengths = torch.full((B,), seq.size(1), dtype=torch.long)
            preds = self.black_box.predict_idx_batch(seq, lengths)
        return preds


class TargetedLatentOracle:
    
    def __init__(
        self,
        black_box: CatBoostPPM,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        seq_len: int,
        device: torch.device,
        target_action_idx: int,
        original_action_idx: int,
        delta: float = 0.0,
    ):
        self.black_box = black_box
        self.decode_fn = decode_fn
        self.seq_len = seq_len
        self.device = device
        self.target_action_idx = target_action_idx
        self.original_action_idx = original_action_idx
        self.delta = delta
        self.target_act, self.target_res = idx_to_joint_action(target_action_idx)
        self.orig_act, self.orig_res = idx_to_joint_action(original_action_idx)
        self.last_t_target: Optional[float] = None
        self.last_t_orig: Optional[float] = None
        self.last_margin: Optional[float] = None     
        self.last_flipped: Optional[bool] = None       

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)
        with torch.no_grad():
            seq = self.decode_fn(z)  # (B, T, event_dim) 
            out = []
            for i in range(B):
                trace_mat = seq[i].cpu().numpy()
                length = seq.size(1)
                t_target = self.black_box.predict_time_for_pair(
                    trace_mat, length, self.target_act, self.target_res
                )
                t_orig = self.black_box.predict_time_for_pair(
                    trace_mat, length, self.orig_act, self.orig_res
                )
                flipped = (t_orig - t_target) > self.delta
                if flipped:
                    out.append(self.target_action_idx)
                else:
                    out.append(self.original_action_idx)
                self.last_t_target = t_target
                self.last_t_orig = t_orig
                self.last_margin = t_orig - t_target
                self.last_flipped = flipped
            return torch.tensor(out, dtype=torch.long)


# Explanation decoder
def feature_names() -> List[str]:
    return event_log.feature_names()

def decode_explanation(
    z_orig: torch.Tensor,              # (1, latent_dim)
    z_cf:   torch.Tensor,              # (1, latent_dim)
    raw_sequence_matrix: np.ndarray,
    original_action: Tuple[str, str],
    cf_action:       Tuple[str, str],
    decode_fn: Callable,
    v_orig: Optional[float] = None,
    v_cf:   Optional[float] = None,
    case_id: str = "?",
    step: int = 0,
    eps: float = 0.05,
) -> Dict:
    """
    Translates the latent counterfactual into a structured, human-readable explanation.

    Returns a dict with:
      - summary_text: plain English narrative
      - changed_features: list of (feature_name, orig_val, cf_val)
    """
    
    with torch.no_grad():
        seq_orig = decode_fn(z_orig).squeeze(0)   # (T, event_dim) or (event_dim,)
        seq_cf   = decode_fn(z_cf).squeeze(0)

    changed_events = []
    changed_features = []

    n_act = len(event_log.ACTIVITIES)
    n_res = len(event_log.RESOURCES)
    scalar_start = n_act + n_res

    if seq_orig.dim() == 2:
        seq_vals_orig = raw_sequence_matrix 
        seq_vals_cf = decode_fn(z_cf).squeeze(0).detach().cpu().numpy()

        for t in range(seq_vals_orig.shape[0]):
            # 1. EVALUATE CATEGORICALS VIA ARGMAX 
            o_act = int(np.argmax(seq_vals_orig[t, :n_act]))
            c_act = int(np.argmax(seq_vals_cf[t, :n_act]))
            
            o_res = int(np.argmax(seq_vals_orig[t, n_act:scalar_start])) if n_res > 0 else 0
            c_res = int(np.argmax(seq_vals_cf[t, n_act:scalar_start])) if n_res > 0 else 0
            
            # Only record an event change if the actual categorical label flipped!
            if o_act != c_act or o_res != c_res:
                changed_events.append({                                                                      
                    "t": t + 1,
                    "activity_from": event_log.IDX2ACT.get(o_act, "?"),
                    "activity_to": event_log.IDX2ACT.get(c_act, "?"),
                    "resource_from": event_log.IDX2RES.get(o_res, "?"),
                    "resource_to": event_log.IDX2RES.get(c_res, "?"),
                })

        # 2. EVALUATE CONTINUOUS NUMERICAL SCALARS VIA EPSILON (0.05)
        feature_list = feature_names()
        n_total = seq_vals_orig.shape[1]

        free_feature_alias_groups = [
            {"AMOUNT_REQ", "RequestedAmount"},
            {"Leucocytes"},
            {"CRP"},
            {"LacticAcid"}, {"case:SUMleges"},
        ]
        free_categorical_group_names = ["LoanGoal", "ApplicationType", "CLOSURE_TYPE", "CLOSURE_REASON"]
        free_categorical_columns = set()
        for _grp_name in free_categorical_group_names:
            free_categorical_columns |= set(event_log.categorical_block_columns(_grp_name))
 
        free_feature_names = set().union(*free_feature_alias_groups) \
            | free_categorical_columns \
            | set(event_log.ACTIVITIES) | set(event_log.RESOURCES)
        
        seen_features = set()
        derived_features = []

        T_last = seq_vals_orig.shape[0] - 1
        for _grp_name in free_categorical_group_names:
            _col_names = event_log.categorical_block_columns(_grp_name)
            _idx = [feature_list.index(n) for n in _col_names if n in feature_list]
            if not _idx:
                continue  # this dataset doesn't have this categorical group

            o_local = int(np.argmax(seq_vals_orig[T_last, _idx]))
            c_local = int(np.argmax(seq_vals_cf[T_last, _idx]))
            if o_local == c_local:
                continue  # no change in this group
 
            idx2val = {i: v for v, i in event_log.CAT_VOCABULARIES.get(_grp_name, {}).items()}
            orig_val_str = idx2val.get(o_local, "?")
            cf_val_str = idx2val.get(c_local, "?")
 
            changed_features.append({
                "feature": _grp_name,
                "original": orig_val_str,
                "counterfactual": cf_val_str,
                "delta": f"{orig_val_str} -> {cf_val_str}",
            })
            seen_features.update(_col_names)  

        for t in range(1, seq_vals_orig.shape[0]):
            for i in range(scalar_start, n_total):
                orig_val = float(seq_vals_orig[t, i])
                cf_val   = float(seq_vals_cf[t, i])
                delta_val = abs(cf_val - orig_val)

                fname = feature_list[i] if i < len(feature_list) else f"feat_{i}"
                if fname in free_categorical_columns:
                    continue  
            
                if fname in seen_features:
                    continue
                if t != seq_vals_orig.shape[0] - 1:
                    continue

                is_free = fname in free_feature_names
                is_derived = event_log.is_derived_feature(fname)
                if not is_free and delta_val <= 0.05:
                    continue  # Skip frozen feature

                seen_features.add(fname)

                real_orig = event_log.denormalize_feature(fname, orig_val)
                real_cf   = event_log.denormalize_feature(fname, cf_val)
                real_delta = real_cf - real_orig

                if not is_free and not is_derived and abs(real_delta) < 1e-6:
                    continue

                entry = {                                          
                    "feature": f"{fname}",
                    "original": f"{real_orig:.2f}",
                    "counterfactual": f"{real_cf:.2f}",
                    "delta": f"{real_delta:+.2f}",
                }

                if is_derived:
                    derived_features.append(entry)
                    continue

                changed_features.append(entry)

    orig_act, orig_res = original_action
    cf_act,   cf_res   = cf_action



    # Create human-readable texts
    cf_feature_text = "".join(
        f"{c['feature']} would change from {c['original']} to {c['counterfactual']}\n"
        for c in changed_features
    ) or "no significant feature changes detected"

    event_change_text = "".join(
        f"Step {e['t']}: ({e['activity_from']}, {e['resource_from']}) to ({e['activity_to']}, {e['resource_to']})\n"
        for e in changed_events[:]
    )

    feat_strings = [
        f"feature {c['feature']} had been {c['counterfactual']} instead of {c['original']}"
        for c in changed_features
    ]
    
    # Format sequence path variations cleanly: "t2 (A_SUBMITTED → A_PREACCEPTED)"
    event_strings = [
        f"at step {e['t']} the path had mutated to {e['activity_to']} and {e['resource_to']} instead of {e['activity_from']} and {e['resource_from']}"
        for e in changed_events
    ]

    # Prefer concise per-event change text in the counterfactual sentence when available
    if event_change_text:
        cf_short = event_change_text
    else:
        cf_short = cf_feature_text

    # Attribute changes
    if changed_features:
        attr_change_text = "; ".join(
            f"{c['feature']} from {c['original']} to {c['counterfactual']}"
            for c in changed_features
        )
    else:
        attr_change_text = "no attribute-level changes detected"

    cause_desc = ""
    attrs_phrase = f"{', '.join(feat_strings)}"
    events_phrase = f"{', '.join(event_strings)}"

    if changed_features and changed_events:
        cause_desc = (
            f"a combination of attribute shifts and sequence structural observed in the past. "
            f"Particularly, {attrs_phrase} and {events_phrase}."
        )
    elif changed_features:
        cause_desc = f"a shift in the historical attributes: {attrs_phrase}"
    else:
        cause_desc = f"a re-sequencing of past workflow steps: {events_phrase}"

    # If BOTH activity and resource changed
    if orig_act != cf_act and orig_res != cf_res:
        dynamic_narrative = (
            f"The model would completely change its next-step recommendation from {original_action} to {cf_action}," 
            f"if there is {cause_desc}"
        )

    # If ONLY activity or resource changed
    # RESOURCE CHANGE ONLY (Activity Invariant)
    elif orig_act == cf_act and orig_res != cf_res:
        dynamic_narrative = (
            f"The model would maintain its recommendation for the task {orig_act}, but would change the resource assignment from {orig_res} to {cf_res}," 
            f"if there is {cause_desc}"
        )

    # ACTIVITY CHANGE ONLY (Resource Invariant)
    elif orig_res == cf_res and orig_act != cf_act:
        dynamic_narrative = (
            f"While {orig_res} was retained as the optimal actor, the model would change the activity from {orig_act} to {cf_act}, "
            f"if there is {cause_desc}" 
        )

    elif orig_act == cf_act and orig_res == cf_res:
        dynamic_narrative = (
            f"Optimization was unable to find a viable path to flip the model's recommendation. "
            f"The original recommendation {original_action} remains unchanged under all tested perturbations."
        )
    
    # Predicted KPI (execution time)
    kpi_text = ""
    if v_orig is not None and v_cf is not None:
        kpi_delta = v_cf - v_orig
        if kpi_delta > 0:
            kpi_text = (
                f"Predicted KPI (execution time - outcome from the black-box model):\n"
                f"  Original recommendation ({orig_act}, {orig_res}): {v_orig:.2f}h\n"
                f"  Counterfactual recommendation ({cf_act}, {cf_res}): {v_cf:.2f}h\n"
                f"  Δ = {kpi_delta:+.2f}h (Recommendation is BETTER)\n\n"
            )
            prospective_kpi_text = ("Moreover, following the recommendation pair could IMPROVE the predicted execution time by saving {:.2f} hours compared to the counterfactual recommendation.".format(kpi_delta))
        else:
            kpi_text = (
                f"Predicted KPI (execution time - outcome from the black-box model):\n"
                f"  Original recommendation ({orig_act}, {orig_res}): {v_orig:.2f}h\n"
                f"  Counterfactual recommendation ({cf_act}, {cf_res}): {v_cf:.2f}h\n"
                f"  Δ = {kpi_delta:+.2f}h (Recommendation is WORSE)\n\n"
            )
            prospective_kpi_text = ("Following the recommendation pair could WORSEN the predicted execution time by {:.2f} hours compared to the counterfactual recommendation!!".format(kpi_delta))

    

    summary_text = (
        f"Original Black-Box Recommendation:  ({orig_act}, {orig_res})\n"
        f"Counterfactual Recommendation:     ({cf_act}, {cf_res})\n\n"
        f"{kpi_text}"
        f"EXPLANATION:\n"
        f"{dynamic_narrative}\n"  
        f"{prospective_kpi_text}\n\n"
        f"DETAILED PAST CHANGES:\n"
        f"Changed Events:\n"
        f"{event_change_text}\n"
        f"Changed Attributes:\n"
        f"{attr_change_text}\n"
        + (
            "\nDerived (context only, not counted as an independent change):\n"
            + "; ".join(
                f"{d['feature']} from {d['original']} to {d['counterfactual']}"
                for d in derived_features
            )
            + "\n"
            if derived_features else ""
        )
    )

   
    return {
        "summary_text":      summary_text,
        "changed_features":  changed_features,
        "narrative":         dynamic_narrative,
        "sparsity":          len(changed_features) + len(changed_events),
        "changed_events":    changed_events,
        "original_action":   original_action,
        "counterfactual_action": cf_action,
        "kpi_original":      v_orig,
        "kpi_counterfactual": v_cf,
        "kpi_delta":         (v_cf - v_orig) if (v_orig is not None and v_cf is not None) else None,
        "derived_features":    derived_features,
    }