
import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, Set

from baseline.revised_plus.data import event_log
from baseline.revised_plus.models.vae import LSTMVAE
from baseline.revised_plus.models.transition import TransitionModel, RewardFunction
from baseline.revised_plus.data import event_log
from baseline.revised_plus.models.blackbox import (
    CatBoostPPM, LatentOracle, TargetedLatentOracle, FaithfulLatentOracle, decode_explanation,
    n_joint_actions, idx_to_joint_action, joint_action_idx, feature_names,
    non_recommendable_action_indices,
)
from baseline.revised_plus.search.counterfactual import LocalSurrogate, CounterfactualSearch, CounterfactualResult
from baseline.revised_plus.utils.training import (
    train_black_box, train_vae, train_transition_model, train_reward_function, pad_sequence
)
from baseline.revised_plus.utils.evaluation import evaluate_explanations, EvaluationReport

logger = logging.getLogger(__name__)


class LatentMDP:

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 124,
        beta_vae: float = 0.01,
        sim_horizon: int = 5,
        gamma: float = 0.95,
        cf_lambda1: float = 0.01,
        cf_lambda2: float = 0.001,
        cf_lambda3: float = 5.0,
        cf_lambda_ldc: float = 0.0,
        cf_enforce_ldc: bool = False,
        cf_score_ldc: bool = True,
        cf_delta: float = 0.02,
        cf_lr: float = 0.05,
        cf_max_iter: int = 200,
        surrogate_n_samples: int = 200,
        surrogate_rho: float = 0.15,
        device: str = "cpu",
    ):
        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim
        self.beta_vae     = beta_vae
        self.sim_horizon  = sim_horizon
        self.gamma        = gamma
        self.device       = torch.device(device)
        self.cf_delta     = cf_delta
        self.cf_max_iter  = cf_max_iter
        self.cf_lr        = cf_lr
        self.cf_lambda1   = cf_lambda1
        self.cf_lambda2   = cf_lambda2
        self.cf_lambda3   = cf_lambda3
        self.cf_lambda_ldc  = cf_lambda_ldc
        self.cf_enforce_ldc = cf_enforce_ldc
        self.cf_score_ldc   = cf_score_ldc

        self._cf_kwargs = dict(
            latent_dim=latent_dim,
            lambda1=cf_lambda1,
            lambda2=cf_lambda2,
            lambda3=cf_lambda3,
            lambda_ldc=cf_lambda_ldc,
            enforce_ldc=cf_enforce_ldc,
            score_ldc=cf_score_ldc,
            delta=cf_delta,
            lr=cf_lr,
            max_iter=cf_max_iter,
            device=self.device,
        )
        self._surr_kwargs = dict(
            n_samples=surrogate_n_samples,
            rho=surrogate_rho,
        )

        self.black_box:       Optional[CatBoostPPM]      = None
        self.vae:             Optional[LSTMVAE]          = None
        self.transition:      Optional[TransitionModel]  = None
        self.reward:          Optional[RewardFunction]   = None
        self._is_fitted:      bool = False

        self._ldc_cache: Dict[int, Optional[list]] = {}


    def fit(
        self,
        cases: List[event_log.Case],
        epochs_bb:         int = 15,
        epochs_vae:        int = 20,
        epochs_transition: int = 20,
        epochs_reward:     int = 15,
        batch_size:        int = 64,
        lr:                float = 1e-3,
        PM_ground_truth: Optional[torch.Tensor] = None,
        gamma_conf:        float = 0.0,
        declare_lambda_tdc: float = 0.1,
        lambda_res:        float = 0.0,
        bb_run_cv:         bool = True,
        bb_cv_k:           int = 5,
        bb_precomputed_cv_summary: Optional[Dict[str, Any]] = None,
        reuse_black_box:   Optional[Any] = None,
    ) -> "LatentMDP":
        logger.info("Phase 1: Offline training started.")
        print("Batch size:", batch_size)
        print("Declare lambda TDC (weight):", declare_lambda_tdc)
        self._fit_cases = cases

        self._ldc_cache = {}

        declare_constraints = None
        if declare_lambda_tdc > 0.0:
            from baseline.revised_plus.iBCM.run_iBCM import mine_constraints, reduce_feature_space

            all_traces = [[e.activity for e in c.events] for c in cases if len(c.events) >= 1]

            _tdc_min_sup = 1
            _tdc_no_win = 1
            _act_counts: Dict[str, int] = {}
            for trace in all_traces:
                for act in set(trace):
                    _act_counts[act] = _act_counts.get(act, 0) + 1
            non_redundant_activities = sorted(
                a for a, cnt in _act_counts.items() if cnt >= len(all_traces) * _tdc_min_sup
            )

            constraint_keys, _annotated = mine_constraints(
                all_traces, non_redundant_activities, "ALL", _tdc_min_sup, _tdc_no_win
            )

            n_before_rfs = len(constraint_keys)
            declare_constraints = list(reduce_feature_space(set(constraint_keys)))
            logger.info(
                f"REVISED+ baseline arm: mined {n_before_rfs} trace-level Declare "
                f"constraints (TDC), reduced to {len(declare_constraints)} via "
                f"reduce_feature_space() for lambda_tdc={declare_lambda_tdc}."
            )

        sample_matrix = cases[0].to_matrix(len(cases[0].events))
        true_event_dim = sample_matrix.shape[1]

        self.vae = LSTMVAE(
            event_dim=true_event_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)

        self.transition = TransitionModel(
            latent_dim=self.latent_dim,
            n_activities=len(event_log.ACTIVITIES),
            n_resources=len(event_log.RESOURCES),
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.reward = RewardFunction(
            latent_dim=self.latent_dim,
            n_activities=len(event_log.ACTIVITIES),
            n_resources=len(event_log.RESOURCES)
        ).to(self.device)

        if reuse_black_box is not None:
            self.black_box = reuse_black_box
            logger.info("  Reusing existing black-box (not retraining) -- "
                        "oracle held fixed for VAE-only retrain")
        else:
            self.black_box = train_black_box(
                cases, self.device, batch_size=batch_size, lr=lr,
                run_cv=bb_run_cv, cv_k=bb_cv_k,
                precomputed_cv_summary=bb_precomputed_cv_summary,
            )
        self.vae = train_vae(
            cases, self.device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            beta=self.beta_vae,
            n_epochs=epochs_vae,
            batch_size=batch_size,
            lr=lr,
            PM_ground_truth=PM_ground_truth,
            gamma_conf=gamma_conf,
            declare_constraints=declare_constraints,
            declare_lambda_tdc=declare_lambda_tdc,
            lambda_res=lambda_res,
            use_scheduled_sampling=True,
        )
        if declare_lambda_tdc > 0.0:
            self.vae.set_declare_constraints(declare_constraints, declare_lambda_tdc)
        self.transition = train_transition_model(
            cases, self.vae, self.device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_epochs=epochs_transition,
            batch_size=batch_size * 2,
            lr=lr,
        )
        self.reward = train_reward_function(
            cases, self.vae, self.device,
            latent_dim=self.latent_dim,
            n_epochs=epochs_reward,
            batch_size=batch_size * 2,
            lr=lr,
        )

        self._is_fitted = True
        logger.info("Phase 1 complete. All components trained.")
        return self


    def _encode_case(self, case: event_log.Case, step: int) -> torch.Tensor:
        mat = torch.tensor(
            case.to_matrix(step)[np.newaxis].astype(np.float32),
            device=self.device,
        )
        lengths = torch.tensor([step], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z_t = self.vae.encode_trace(mat, lengths)
        return z_t


    def _decode_z(self, z: torch.Tensor, seq_len: int = 4) -> torch.Tensor:
        return self.vae.decode_autoregressive(z, seq_len)


    def _simulate_forward(
        self,
        z0: torch.Tensor,
        max_K: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_acts = len(event_log.ACTIVITIES)
        n_res  = len(event_log.RESOURCES)

        end_activity_idx = event_log.ACT2IDX['END']

        executed_acts = []
        executed_res  = []

        z = z0.clone()
        last_selected_activity = None
        stuck_counter = 0

        for k in range(max_K):
            best_r = -float("inf")
            best_a, best_r_ = 0, 0

            with torch.no_grad():
                for a in range(n_acts):
                    act_name = event_log.IDX2ACT.get(a, "")
                    if act_name == 'START':
                        continue
                    if act_name == last_selected_activity:
                        stuck_counter += 1
                        if stuck_counter > 2:
                            continue

                    for r in range(n_res):
                        a_t = torch.tensor([a], dtype=torch.long, device=self.device)
                        r_t = torch.tensor([r], dtype=torch.long, device=self.device)
                        rv = self.reward(z, a_t, r_t).item()
                        if rv > best_r:
                            best_r, best_a, best_r_ = rv, a, r

            if event_log.IDX2ACT.get(best_a, "") == last_selected_activity:
                stuck_counter += 1
            else:
                stuck_counter = 0

            last_selected_activity = event_log.IDX2ACT.get(best_a, "")
            executed_acts.append(best_a)
            executed_res.append(best_r_)

            if best_a == end_activity_idx:
                break

            with torch.no_grad():
                a_t = torch.tensor([best_a], dtype=torch.long, device=self.device)
                r_t = torch.tensor([best_r_], dtype=torch.long, device=self.device)
                z = self.transition(z, a_t, r_t)

        K_actual = len(executed_acts)
        act_seq = torch.tensor(executed_acts, dtype=torch.long, device=self.device).unsqueeze(0)
        res_seq = torch.tensor(executed_res, dtype=torch.long, device=self.device).unsqueeze(0)

        traj = self.transition.simulate(
            z0,
            act_seq.squeeze(0),
            res_seq.squeeze(0),
        )
        return traj, act_seq, res_seq

    @staticmethod
    def _validate_resource_activity_pairs(
        decoded_cf: np.ndarray,
    ) -> dict:
        n_act = len(event_log.ACTIVITIES)
        n_res = len(event_log.RESOURCES)
        violations = []

        T = decoded_cf.shape[0]
        for t in range(T):
            act_idx = int(np.argmax(decoded_cf[t, :n_act]))
            res_idx = int(np.argmax(decoded_cf[t, n_act:n_act + n_res]))
            activity = event_log.IDX2ACT.get(act_idx, "UNKNOWN")
            resource = event_log.IDX2RES.get(res_idx, "UNKNOWN")

            if not event_log.is_feasible_pair(activity, resource):
                violations.append({
                    "step": t + 1,
                    "activity": activity,
                    "resource": resource,
                    "reason": f"{resource} never performed {activity} in training log"
                })

        return {
            "resource_valid": len(violations) == 0,
            "violations": violations,
        }


    def explain(
        self,
        case: event_log.Case,
        step: int,
        surrogate_epochs: int = 200,
        verbose: bool = True,
        cf_max_iter: Optional[int] = None,
        cf_lr: Optional[float] = None,
        cf_lambda1: Optional[float] = None,
        cf_lambda2: Optional[float] = None,
        cf_lambda3: Optional[float] = None,
        k: int = 5,
        max_attempts: int = 20,
        contrast_mode: str = "any",
        debug_attempts_dir: Optional[str] = None,
        kpi_significance_threshold: float = 0.05,
        kpi_min_baseline: float = 1e-3,
        require_target_match: bool = False,
        editable_steps: Optional[List[int]] = None,
    ) -> Dict:
        valid_contrast_modes = {"activity", "resource", "joint", "any"}
        if contrast_mode not in valid_contrast_modes:
            raise ValueError(
                f"contrast_mode must be one of {valid_contrast_modes}, got {contrast_mode!r}"
            )
        assert self._is_fitted, "Call .fit() before .explain()"
        assert step <= len(case.events), f"step {step} > trace length {len(case.events)}"

        _editable_row_indices: Set[int] = {step - 1}
        if editable_steps:
            for s in editable_steps:
                assert 1 <= s <= step, (
                    f"editable_steps entry {s} out of range — must be between "
                    f"1 and {step} (the current encoded prefix length)."
                )
                _editable_row_indices.add(s - 1)
        history_edit_multi_step = len(_editable_row_indices) > 1

        max_iter = cf_max_iter if cf_max_iter is not None else self.cf_max_iter
        lr_step  = cf_lr if cf_lr is not None else self.cf_lr
        lam1     = cf_lambda1 if cf_lambda1 is not None else self.cf_lambda1
        lam2     = cf_lambda2 if cf_lambda2 is not None else self.cf_lambda2
        lam3     = cf_lambda3 if cf_lambda3 is not None else self.cf_lambda3

        raw_matrix = case.to_matrix(step)

        cf_search = CounterfactualSearch(
            latent_dim=self.latent_dim,
            lambda1=lam1,
            lambda2=lam2,
            lambda3=lam3,
            delta=self.cf_delta,
            lr=lr_step,
            max_iter=max_iter,
            device=self.device,
        )

        z_t = self._encode_case(case, step)

        mat     = torch.tensor(case.to_matrix(step)[np.newaxis].astype(np.float32), device=self.device)
        lengths = torch.tensor([step], dtype=torch.long, device=self.device)
        orig_act, orig_res = self.black_box.predict(mat, lengths)
        orig_action_idx    = joint_action_idx(orig_act, orig_res)

        v_orig = self.black_box.predict_time_for_pair(
            raw_matrix, step, orig_act, orig_res
        )


        from baseline.revised_plus.models.blackbox import feature_names
        all_features = feature_names()
        event_dim = len(all_features)

        free_feature_alias_groups: List[Set[str]] = [
                    {"AMOUNT_REQ", "RequestedAmount"},
                    {"Leucocytes"},
                    {"CRP"},
                    {"LacticAcid"}, {"case:SUMleges", "SUMleges"},

                ]

        free_categorical_group_names: List[str] = ["LoanGoal", "ApplicationType", "CLOSURE_TYPE", "CLOSURE_REASON"]
        free_categorical_columns: Set[str] = set()
        for _grp_name in free_categorical_group_names:
            free_categorical_columns |= set(event_log.categorical_block_columns(_grp_name))

        free_feature_names = set().union(*free_feature_alias_groups) \
            | free_categorical_columns \
            | set(event_log.ACTIVITIES) | set(event_log.RESOURCES)

        invariant_names = [n for n in all_features if n not in free_feature_names]
        invariant_indices = [all_features.index(n) for n in invariant_names]

        act_res_indices = [
            all_features.index(n)
            for n in (event_log.ACTIVITIES + event_log.RESOURCES)
            if n in all_features
        ]
        act_indices = [all_features.index(n) for n in event_log.ACTIVITIES if n in all_features]
        res_indices = [all_features.index(n) for n in event_log.RESOURCES if n in all_features]

        freq_indices = [
            all_features.index(f"freq_{a}") for a in event_log.ACTIVITIES
            if f"freq_{a}" in all_features
        ]

        _unmatched_groups = [
            grp for grp in free_feature_alias_groups
            if not (grp & set(all_features))
        ]
        _act_res_names = set(event_log.ACTIVITIES) | set(event_log.RESOURCES)
        _unrecognized_act_res = [n for n in _act_res_names if n not in all_features]
        if _unmatched_groups or _unrecognized_act_res:
            logger.warning(
                f"rigid_invariant_decoder: expected free feature(s) not found "
                f"in feature_names(): alias groups with zero matches={_unmatched_groups}, "
                f"unmatched activities/resources={_unrecognized_act_res} — check spelling "
                f"against event_log.feature_names()."
            )

        _anchor = torch.tensor(raw_matrix, dtype=torch.float32, device=self.device)
        _frozen_mask = torch.zeros(event_dim, dtype=torch.bool, device=self.device)
        if invariant_indices:
            _frozen_mask[invariant_indices] = True

        _SNAP_SENTINEL_ACTS = {"START", "END"}
        _SNAP_SENTINEL_RES = {"SYSTEM"}
        _snap_blocks: List[Tuple[List[int], List[int]]] = []
        if act_indices:
            _blk_names = [n for n in event_log.ACTIVITIES if n in all_features]
            _snap_blocks.append((act_indices,
                                 [i for i, n in enumerate(_blk_names) if str(n).upper() in _SNAP_SENTINEL_ACTS]))
        if res_indices:
            _blk_names = [n for n in event_log.RESOURCES if n in all_features]
            _snap_blocks.append((res_indices,
                                 [i for i, n in enumerate(_blk_names) if str(n).upper() in _SNAP_SENTINEL_RES]))
        for _grp_name in free_categorical_group_names:
            _cols = [all_features.index(n) for n in event_log.categorical_block_columns(_grp_name)
                     if n in all_features]
            if _cols:
                _snap_blocks.append((_cols, []))

        def _snap_to_onehot(seq: torch.Tensor) -> torch.Tensor:
            out = seq.clone()
            T = seq.size(1)
            for cols, forbidden in _snap_blocks:
                col_t = torch.tensor(cols, dtype=torch.long, device=seq.device)
                block = seq.index_select(-1, col_t)
                if forbidden and len(forbidden) < len(cols):
                    fb_t = torch.tensor(forbidden, dtype=torch.long, device=seq.device)
                    real_pos = _anchor[:T].index_select(-1, col_t).argmax(dim=-1)
                    real_is_sentinel = (real_pos.unsqueeze(-1) == fb_t.unsqueeze(0))
                    fb_vals = block.index_select(-1, fb_t)
                    fb_vals = torch.where(real_is_sentinel.unsqueeze(0), fb_vals,
                                          torch.full_like(fb_vals, float("-inf")))
                    block = block.clone()
                    block[..., fb_t] = fb_vals
                onehot = F.one_hot(block.argmax(dim=-1), num_classes=len(cols)).to(seq.dtype)
                out[..., col_t] = onehot
            return out

        def _build_rigid_invariant_decoder(
            freeze_activity_at_last: bool = False,
            freeze_resource_at_last: bool = False,
            editable_row_indices: Optional[Set[int]] = None,
            materialize: bool = False,
        ):
            if editable_row_indices is None:
                editable_row_indices = {step - 1}
            editable_row_indices = set(editable_row_indices)

            extra_last_step_frozen = []
            if freeze_activity_at_last:
                extra_last_step_frozen.extend(act_indices)
            if freeze_resource_at_last:
                extra_last_step_frozen.extend(res_indices)

            def rigid_invariant_decoder(z_tensor):
                raw_seq = self._decode_z(z_tensor, seq_len=step)
                if materialize:
                    raw_seq = _snap_to_onehot(raw_seq)
                T = raw_seq.size(1)

                if invariant_indices or act_res_indices:
                    anchor_rows = _anchor[:T, :].unsqueeze(0).expand(raw_seq.size(0), -1, -1).clone()

                    if freq_indices and act_indices:
                        for k in sorted(editable_row_indices):
                            if k >= T - 1:
                                continue
                            if materialize:
                                decoded_act_k = raw_seq[:, k, :][:, act_indices]
                            else:
                                decoded_act_k = F.softmax(raw_seq[:, k, :][:, act_indices], dim=-1)
                            anchor_act_k = _anchor[k, act_indices].unsqueeze(0)
                            delta_k = decoded_act_k - anchor_act_k
                            for i in range(k + 1, T):
                                denom = max(1.0, float(i))
                                anchor_rows[:, i, freq_indices] = (
                                    anchor_rows[:, i, freq_indices] + delta_k / denom
                                )

                    mask = _frozen_mask.view(1, 1, -1).expand(raw_seq.size(0), T, -1).clone()

                    if act_res_indices:
                        frozen_rows = [r for r in range(T) if r not in editable_row_indices]
                        if frozen_rows:
                            row_idx = torch.tensor(frozen_rows, device=mask.device)
                            col_idx = torch.tensor(act_res_indices, device=mask.device)
                            mask[:, row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = True

                    if extra_last_step_frozen and editable_row_indices:
                        last_editable = max(editable_row_indices)
                        mask[:, last_editable, extra_last_step_frozen] = True

                    raw_seq = torch.where(mask, anchor_rows, raw_seq)

                return raw_seq

            return rigid_invariant_decoder

        rigid_invariant_decoder = _build_rigid_invariant_decoder(
            freeze_activity_at_last=False,
            freeze_resource_at_last=False,
            editable_row_indices=_editable_row_indices,
        )

        rigid_invariant_history = _build_rigid_invariant_decoder(
            freeze_activity_at_last=False,
            freeze_resource_at_last=False,
            editable_row_indices=_editable_row_indices,
            materialize=True,
        )

        oracle = LatentOracle(self.black_box, rigid_invariant_history, seq_len=step, device=self.device)

        from baseline.revised_plus.models.blackbox import feasible_joint_action_indices
        surrogate = LocalSurrogate(
            latent_dim=self.latent_dim,
            n_actions=n_joint_actions(),
            feasible_indices=feasible_joint_action_indices(min_observations=1).to(self.device),
        ).to(self.device)
        surr_history = surrogate.fit(
            z_center=z_t,
            black_box_fn=oracle,
            n_epochs=surrogate_epochs,
            device=self.device,
            **self._surr_kwargs,
        )
        surrogate_precision = surrogate.evaluate_precision(
            z_center=z_t,
            black_box_fn=oracle,
            n_actions=n_joint_actions(),
            rho=self._surr_kwargs["rho"],
            device=self.device,
        )
        if verbose:
            with torch.no_grad():
                surr_idx_t = surrogate.predict(z_t).item()
                bb_idx_t = oracle(z_t).item()
            surr_act_t, surr_res_t = idx_to_joint_action(surr_idx_t)
            bb_act_t, bb_res_t = idx_to_joint_action(bb_idx_t)

        with torch.no_grad():
            surr_logits = surrogate(z_t).squeeze(0)
            surr_logits[orig_action_idx] = -float('inf')
            for bad_idx in non_recommendable_action_indices():
                surr_logits[bad_idx] = -float('inf')
            ranked_targets = torch.argsort(surr_logits, descending=True).cpu().tolist()
            ranked_targets = [idx for idx in ranked_targets if idx != orig_action_idx]
            ranked_targets = [idx for idx in ranked_targets if surr_logits[idx].item() != -float('inf')]

            def _classify_contrast(idx: int) -> str:
                cand_act, cand_res = idx_to_joint_action(idx)
                act_changed = (cand_act != orig_act)
                res_changed = (cand_res != orig_res)
                if act_changed and res_changed:
                    return "joint"
                elif act_changed:
                    return "activity"
                elif res_changed:
                    return "resource"
                else:
                    return "none"

            if contrast_mode != "any":
                ranked_targets = [
                    idx for idx in ranked_targets if _classify_contrast(idx) == contrast_mode
                ]
                if verbose and not ranked_targets:
                    logger.warning(
                        f"contrast_mode={contrast_mode!r} left no eligible candidate "
                        f"targets in the ranked queue (e.g. no feasible action changes "
                        f"only the {contrast_mode} component while holding the other "
                        f"fixed at this step) — 0 explanations will be returned."
                    )

        searcher = CounterfactualSearch(**self._cf_kwargs)


        def _get_target_constraints(target_idx: int):
            t_act, t_res = idx_to_joint_action(target_idx)
            if t_act in self._ldc_cache:
                return self._ldc_cache[t_act]
            from baseline.revised_plus.iBCM.run_iBCM import mine_constraints, reduce_feature_space
            same_label_prefixes_raw: List[List[str]] = []
            for c in getattr(self, "_fit_cases", []):
                for i in range(len(c.events) - 1):
                    nxt = c.events[i + 1]
                    if nxt.activity == t_act:
                        same_label_prefixes_raw.append([e.activity for e in c.events[: i + 1]])

            if not same_label_prefixes_raw:
                logger.warning(
                    f"REVISED+ baseline arm: no training prefixes found with "
                    f"next activity {t_act} — LDC unavailable for "
                    f"target_idx={target_idx} (activity={t_act})."
                )
                self._ldc_cache[t_act] = None
                return None

            _ldc_min_sup = 1
            _ldc_no_win = 1
            _act_counts: Dict[str, int] = {}
            for trace in same_label_prefixes_raw:
                for act in set(trace):
                    _act_counts[act] = _act_counts.get(act, 0) + 1
            non_redundant_activities = sorted(
                a for a, cnt in _act_counts.items()
                if cnt >= len(same_label_prefixes_raw) * _ldc_min_sup
            )

            constraint_keys, _annotated = mine_constraints(
                same_label_prefixes_raw, non_redundant_activities,
                t_act, _ldc_min_sup, _ldc_no_win,
            )
            constraints = list(reduce_feature_space(set(constraint_keys)))
            self._ldc_cache[t_act] = constraints
            return constraints

        all_explanations = []
        discovered_vectors = []
        seen_actions = set()
        seen_indices = set()

        queue_pos = 0
        attempts = 0
        attempt_log: List[Dict] = []

        while (
            len(all_explanations) < k
            and queue_pos < len(ranked_targets)
            and attempts < max_attempts
        ):
            target_idx = ranked_targets[queue_pos]
            queue_pos += 1
            attempts += 1

            if target_idx in seen_indices:
                continue

            rank = len(all_explanations)

            target_contrast = _classify_contrast(target_idx)
            target_decoder = _build_rigid_invariant_decoder(
                freeze_activity_at_last=(target_contrast == "resource"),
                freeze_resource_at_last=(target_contrast == "activity"),
                editable_row_indices=_editable_row_indices,
            )
            target_history_fn = _build_rigid_invariant_decoder(
                freeze_activity_at_last=(target_contrast == "resource"),
                freeze_resource_at_last=(target_contrast == "activity"),
                editable_row_indices=_editable_row_indices,
                materialize=True,
            )
            target_oracle = FaithfulLatentOracle(
                self.black_box, target_history_fn, seq_len=step, device=self.device,
                original_action_idx=orig_action_idx,
            )
            target_act_s, target_res_s = idx_to_joint_action(target_idx)

            def _accept(idx: int, _target=target_idx) -> bool:
                if idx == orig_action_idx:
                    return False
                if idx in non_recommendable_action_indices():
                    return False
                if contrast_mode != "any" and _classify_contrast(idx) != contrast_mode:
                    return False
                if idx in seen_indices:
                    return False
                if require_target_match and idx != _target:
                    return False
                return True

            _use_ldc = (self.cf_lambda_ldc > 0.0 or self.cf_score_ldc)
            cf_result = searcher.search(
                z_t=z_t,
                original_action_idx=orig_action_idx,
                surrogate=surrogate,
                black_box_fn=target_oracle,
                decode_fn=target_decoder,
                step=step,
                target_action_idx=target_idx,
                past_cf_vectors=discovered_vectors,
                target_constraints=(_get_target_constraints(target_idx) if _use_ldc else None),
                constraints_fn=(_get_target_constraints if _use_ldc else None),
                n_act=len(event_log.ACTIVITIES),
                raw_sequence_matrix=raw_matrix,
                accept_fn=_accept,
            )

            cf_act, cf_res = idx_to_joint_action(cf_result.cf_action_idx)

            _target_feasible = event_log.is_feasible_pair(target_act_s, target_res_s)
            _attempt_record = {
                "attempt": attempts,
                "queue_pos": queue_pos - 1,
                "target_idx": int(target_idx),
                "target_act": target_act_s,
                "target_res": target_res_s,
                "target_contrast": target_contrast,
                "orig_act": orig_act,
                "orig_res": orig_res,
                "target_pair_feasible": _target_feasible,
                "t_orig": target_oracle.last_t_orig,
                "t_target": target_oracle.last_t_rec,
                "t_bb_rec": target_oracle.last_t_rec,
                "margin": target_oracle.last_margin,
                "orig_is_candidate_under_cf": target_oracle.last_orig_is_candidate,
                "cf_delta": self.cf_delta,
                "oracle_flipped": bool(cf_result.verified_by_blackbox),
                "hit_target": cf_result.hit_target,
                "cf_action_idx": int(cf_result.cf_action_idx),
                "cf_act": cf_act,
                "cf_res": cf_res,
                "ldc_final_violation": cf_result.ldc_final_violation,
                "n_iterations": cf_result.n_iterations,
                "used_expulsion": cf_result.used_expulsion,
                "outcome": None,
            }

            if not cf_result.verified_by_blackbox:
                if cf_result.cf_action_idx == orig_action_idx:
                    _attempt_record["outcome"] = "not_flipped"
                elif cf_result.cf_action_idx in seen_indices:
                    _attempt_record["outcome"] = "duplicate"
                elif contrast_mode != "any" and _classify_contrast(cf_result.cf_action_idx) != contrast_mode:
                    _attempt_record["outcome"] = "contrast_mismatch"
                elif require_target_match and cf_result.cf_action_idx != target_idx:
                    _attempt_record["outcome"] = "target_mismatch"
                elif self.cf_enforce_ldc and cf_result.ldc_final_violation > 1e-3:
                    _attempt_record["outcome"] = "ldc_violation_rejected"
                else:
                    _attempt_record["outcome"] = "rejected"
                attempt_log.append(_attempt_record)
                continue

            with torch.no_grad():
                _decoded_check = target_history_fn(cf_result.z_cf).squeeze(0)
                _last_row = _decoded_check[-1]
                _decoded_act = event_log.IDX2ACT.get(int(torch.argmax(_last_row[:len(event_log.ACTIVITIES)]).item()))
                _res_slice = _last_row[len(event_log.ACTIVITIES):len(event_log.ACTIVITIES) + len(event_log.RESOURCES)]
                _decoded_res = event_log.IDX2RES.get(int(torch.argmax(_res_slice).item()))

            _faithful = True
            if target_contrast == "activity" and _decoded_res != cf_res:
                _faithful = False
            elif target_contrast == "resource" and _decoded_act != cf_act:
                _faithful = False

            if not _faithful:
                _attempt_record["outcome"] = "faithfulness_check_failed"
                attempt_log.append(_attempt_record)
                continue

            if self.cf_enforce_ldc and cf_result.ldc_final_violation > 1e-3:
                _attempt_record["outcome"] = "ldc_violation_rejected"
                attempt_log.append(_attempt_record)
                continue

            with torch.no_grad():
                cf_history_np = target_history_fn(cf_result.z_cf).squeeze(0).cpu().numpy()
            (bb_act, bb_res), t_bb_cf = self.black_box.recommend(
                cf_history_np, step, return_time=True
            )
            if (bb_act, bb_res) != (cf_act, cf_res):
                _attempt_record["outcome"] = "bb_recheck_mismatch"
                attempt_log.append(_attempt_record)
                logger.warning(
                    f"  --> Dropping CF for target idx {target_idx}: F re-check on the "
                    f"reported history gave ({bb_act}, {bb_res}) but search reported "
                    f"({cf_act}, {cf_res})."
                )
                continue
            t_orig_on_cf = self.black_box.predict_time_for_pair(
                cf_history_np, step, orig_act, orig_res
            )
            orig_is_candidate_on_cf = (orig_act, orig_res) in self.black_box.candidate_pairs(
                cf_history_np, step
            )
            _attempt_record["outcome"] = "accepted_stage1"
            attempt_log.append(_attempt_record)

            v_cf = self.black_box.predict_time_for_pair(
                raw_matrix, step, cf_act, cf_res
            )

            explanation = decode_explanation(
                z_orig=z_t,
                z_cf=cf_result.z_cf,
                raw_sequence_matrix=raw_matrix,
                original_action=(orig_act, orig_res),
                cf_action=(cf_act, cf_res),
                decode_fn=target_history_fn,
                v_orig=v_orig, v_cf=v_cf,
                case_id=case.case_id,
                step=step
            )

            decoded_cf_np = cf_history_np


            resource_check = self._validate_resource_activity_pairs(
                decoded_cf = decoded_cf_np,
            )

            if not resource_check["resource_valid"]:
                logger.info(f"  --> Discarding CF (target idx {target_idx}) due to feasibility violations.")
                continue

            if resource_check["violations"] and not resource_check["resource_valid"]:
                logger.warning(
                    f"  Resource violations in candidate (would-be rank {rank+1}, target idx {target_idx}) "
                    f"({len(resource_check['violations'])} step(s) infeasible). "
                )
                for v in resource_check["violations"]:
                    violated_item = next((item for item in explanation['changed_events'] if item['t'] == v['step']), None)
                    logger.warning(
                        f"    Step {v['step']}: '{v['resource']}' cannot perform "
                        f"'{v['activity']}' (was: {violated_item['resource_from']} → {violated_item['activity_from']})"
                    )


            compatibility_matrix = getattr(event_log, "RESOURCE_ACTIVITY_COMPATIBILITY", None)

            if compatibility_matrix and not resource_check["resource_valid"]:
                continue

            seen_actions.add((cf_act, cf_res))
            seen_indices.add(cf_result.cf_action_idx)
            discovered_vectors.append(cf_result.z_cf.detach().clone())

            explanation["resource_valid"]       = resource_check["resource_valid"]
            explanation["resource_violations"]  = resource_check["violations"]
            explanation["target_rank"] = rank + 1

            explanation["cf_pair_source"]              = "blackbox_on_cf_history"
            explanation["bb_verified_on_cf_history"]   = True
            explanation["surrogate_target_act"]        = target_act_s
            explanation["surrogate_target_res"]        = target_res_s
            explanation["hit_surrogate_target"]        = bool(cf_result.hit_target)
            explanation["orig_pair_is_candidate_under_cf_history"] = bool(orig_is_candidate_on_cf)
            explanation["kpi_counterfactual_cf_history"] = float(t_bb_cf)
            explanation["kpi_original_cf_history"]       = float(t_orig_on_cf)

            explanation["contrast_type"] = _classify_contrast(cf_result.cf_action_idx)
            explanation["contrast_mode_requested"] = contrast_mode

            explanation["history_edit_multi_step"] = history_edit_multi_step
            explanation["editable_steps_used"] = sorted(s + 1 for s in _editable_row_indices)

            explanation["cf_result"] = cf_result
            explanation["z_t"]       = z_t
            explanation["z_cf"]      = cf_result.z_cf

            validity_note = (
                "\nAll suggested (activity, resource) "
                "pairs are empirically valid — each resource has performed the "
                "proposed activity in historical cases."
            )

            _contrast_type = explanation["contrast_type"]
            if _contrast_type == "activity":
                contrast_note = (
                    f"\nThis counterfactual holds the resource assignment fixed "
                    f"({orig_res}) and contrasts only the activity decision "
                    f"({orig_act} → {cf_act}). It does not explain the resource "
                    f"assignment."
                )
            elif _contrast_type == "resource":
                contrast_note = (
                    f"\nThis counterfactual holds the recommended activity fixed "
                    f"({orig_act}) and contrasts only the resource assignment "
                    f"({orig_res} → {cf_res}). It does not explain the activity "
                    f"choice."
                )
            else:
                contrast_note = (
                    f"\nThis counterfactual contrasts the full recommendation: "
                    f"both the activity ({orig_act} → {cf_act}) and the resource "
                    f"({orig_res} → {cf_res}) differ."
                )

            if "summary_text" in explanation:
                explanation["summary_text"] = explanation["summary_text"].rstrip() \
                                              + '\n' + contrast_note \
                                              + '\n' + validity_note

            all_explanations.append(explanation)

        def _kpi_delta_pct(expl: Dict) -> Optional[float]:
            v_o = expl.get("kpi_original")
            v_c = expl.get("kpi_counterfactual")
            if v_o is None or v_c is None:
                return None
            if abs(v_o) < kpi_min_baseline:
                return v_c - v_o
            return (v_c - v_o) / v_o

        best_abs_delta = 0.0
        for expl in all_explanations:
            delta_pct = _kpi_delta_pct(expl)
            expl["kpi_delta_pct"] = delta_pct
            if delta_pct is not None:
                best_abs_delta = max(best_abs_delta, abs(delta_pct))

        no_significant_kpi_cf = best_abs_delta < kpi_significance_threshold

        for expl in all_explanations:
            delta_pct = expl.get("kpi_delta_pct")
            expl["kpi_significant"] = (
                delta_pct is not None
                and abs(delta_pct) >= kpi_significance_threshold
            )
            expl["kpi_significance_threshold"] = kpi_significance_threshold

        all_explanations.sort(
            key=lambda e: abs(e.get("kpi_delta_pct") or 0.0), reverse=True
        )
        for new_rank, expl in enumerate(all_explanations):
            expl["kpi_rank"] = new_rank + 1

        if no_significant_kpi_cf and all_explanations and verbose:
            logger.warning(
                f"No CF in this batch cleared the KPI significance threshold "
                f"({kpi_significance_threshold:.0%}); best |delta| = "
                f"{best_abs_delta:.2%}. Returning best-available CF(s) tagged "
                f"kpi_significant=False rather than discarding the batch."
            )

        if verbose:
            if len(all_explanations) < k:
                stopped_reason = (
                    "exhausted the ranked action queue"
                    if queue_pos >= len(ranked_targets)
                    else f"hit max_attempts={max_attempts}"
                )
                logger.warning(
                    f"Requested k={k} distinct counterfactuals but only found {len(all_explanations)}!"
                )
            else:
                logger.info(
                    f"Found {len(all_explanations)} distinct counterfactuals "
                    f"after {attempts} attempt(s))."
                )

        if debug_attempts_dir:
            os.makedirs(debug_attempts_dir, exist_ok=True)
            debug_path = os.path.join(
                debug_attempts_dir, f"case_{case.case_id}_step_{step}_attempts.json"
            )
            with open(debug_path, "w") as f:
                json.dump({
                    "case_id": str(case.case_id),
                    "step": step,
                    "contrast_mode_requested": contrast_mode,
                    "n_attempts": attempts,
                    "n_found": len(all_explanations),
                    "attempts": attempt_log,
                }, f, indent=2, default=str)
            if verbose:
                logger.info(f"  Wrote {len(attempt_log)} attempt record(s) → {debug_path}")

        return {
            "multiple_explanations": all_explanations,
            "surrogate_loss_history": surr_history,
            "surrogate_precision": surrogate_precision,
            "n_requested": k,
            "n_found": len(all_explanations),
            "n_attempts": attempts,
            "max_attempts": max_attempts,
            "n_candidates_available": len(ranked_targets),
            "no_significant_kpi_cf": no_significant_kpi_cf if all_explanations else None,
            "best_kpi_delta_pct": best_abs_delta if all_explanations else None,
            "kpi_significance_threshold": kpi_significance_threshold,
        }


    def evaluate(
        self,
        cases: List[event_log.Case],
        n_samples: int = 30,
        min_step: int = 2,
        seed: int = 0,
    ) -> EvaluationReport:
        assert self._is_fitted, "Call .fit() before .evaluate()"
        rng = np.random.default_rng(seed)

        samples = []
        for case in cases:
            for step in range(min_step, len(case.events)):
                samples.append((case, step))
        if len(samples) > n_samples:
            idx = rng.choice(len(samples), size=n_samples, replace=False)
            samples = [samples[i] for i in idx]

        results_list:       List = []
        cf_z_list:          List = []
        cf_decoded_list:    List = []
        changed_feat_list:  List = []
        dV_predicted:       List = []
        dV_actual:          List = []

        logger.info(f"Evaluating {len(samples)} (case, step) pairs…")
        for i, (case, step) in enumerate(samples):
            try:
                out = self.explain(case, step, verbose=False)
            except Exception as e:
                logger.warning(f"  Sample {i} failed: {e}")
                continue

            candidates = out.get("multiple_explanations", [])
            if not candidates:
                logger.warning(f"  Sample {i} produced no counterfactual candidates, skipping.")
                continue
            expl = candidates[0]

            results_list.append(expl["cf_result"])
            cf_z_list.append(expl["z_cf"])

            with torch.no_grad():
                dec = self._decode_z(expl["z_cf"], seq_len=step).squeeze(0).cpu().numpy()
            cf_decoded_list.append(dec)
            changed_feat_list.append(expl["changed_features"])

            dV_predicted.append(expl.get("kpi_delta", 0.0) or 0.0)

            remaining = case.events[step].remaining_time
            if remaining is None or np.isnan(remaining):
                actual_outcome = 0.0
            else:
                actual_outcome = -float(remaining)
            dV_actual.append(actual_outcome)

            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(samples)} done")

        return evaluate_explanations(
            results=results_list,
            cf_z_list=cf_z_list,
            cf_decoded_list=cf_decoded_list,
            changed_features_list=changed_feat_list,
            feature_names=feature_names(),
            delta_V_predicted=dV_predicted,
            delta_V_actual=dV_actual,
            vae=self.vae,
            device=self.device,
        )


    def save(self, path: str = "latent_mdp_weights") -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.black_box.save(p)
        torch.save(self.vae.state_dict(),         p / "vae.pt")
        torch.save(self.transition.state_dict(),  p / "transition.pt")
        torch.save(self.reward.state_dict(),      p / "reward.pt")
        logger.info(f"Models saved to {path}/")

    def load(
        self,
        path: str = "latent_mdp_weights",
        declare_constraints: Optional[list] = None,
        declare_lambda_tdc: float = 0.0,
    ) -> "LatentMDP":
        p = Path(path)

        self.black_box  = CatBoostPPM.load(p)

        self.vae = LSTMVAE(event_log.EVENT_DIM, self.hidden_dim, self.latent_dim).to(self.device)

        self.transition = TransitionModel(
            self.latent_dim, len(event_log.ACTIVITIES), len(event_log.RESOURCES),
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.reward = RewardFunction(
            self.latent_dim, len(event_log.ACTIVITIES), len(event_log.RESOURCES)
        ).to(self.device)

        _vae_res = self.vae.load_state_dict(torch.load(p / "vae.pt", map_location=self.device), strict=False)
        self.vae_missing_keys    = list(_vae_res.missing_keys)
        self.vae_unexpected_keys = list(_vae_res.unexpected_keys)
        if self.vae_missing_keys or self.vae_unexpected_keys:
            logger.warning(
                f"vae.pt key mismatch: missing={self.vae_missing_keys} "
                f"unexpected={self.vae_unexpected_keys}"
            )

        if (p / "transition.pt").exists():
            self.transition.load_state_dict(torch.load(p / "transition.pt", map_location=self.device))
        else:
            logger.warning(f"{p / 'transition.pt'} not found: transition model not loaded "
                           f"(not needed for CF search).")
            self.transition = None
        if (p / "reward.pt").exists():
            self.reward.load_state_dict(torch.load(p / "reward.pt", map_location=self.device))
        else:
            logger.warning(f"{p / 'reward.pt'} not found: reward model not loaded "
                           f"(not needed for CF search).")
            self.reward = None

        if declare_lambda_tdc > 0.0:
            self.vae.set_declare_constraints(declare_constraints, declare_lambda_tdc)

        for m in [self.vae, self.transition, self.reward]:
            if m is not None:
                m.eval()

        self._is_fitted = True
        logger.info(f"Models loaded from {path}/")
        return self
