"""
Two-phase framework for counterfactual explanation of black-box next-step recommendations in business processes.
Baseline Implementation: REVISED+_pres
"""

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
from baseline.revised_plus.data import event_log
from baseline.revised_plus.models.blackbox import (
    CatBoostPPM, LatentOracle, TargetedLatentOracle, decode_explanation,
    n_joint_actions, idx_to_joint_action, joint_action_idx, 
    non_recommendable_action_indices,
) 
from baseline.revised_plus.search.counterfactual import LocalSurrogate, CounterfactualSearch
from baseline.revised_plus.utils.training import (
    train_black_box, train_vae
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Full two-phase framework for counterfactual explanation of black-box
    next-step recommendations in business processes.

    Usage
    -----
    >>> pipeline = Pipeline(latent_dim=32, device='cpu')
    >>> pipeline.fit(train_cases, epochs_vae=20, epochs_bb=15)
    >>> explanation = pipeline.explain(case, step=4)
    >>> print(explanation['summary_text'])
    """

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
        cf_lambda_ldc: float = 0.0,      # REVISED+_pres
        cf_enforce_ldc: bool = False,    # REVISED+_pres (hard L_LDC=0 acceptance gate)
        cf_score_ldc: bool = True,       # REVISED+_pres
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
        self._is_fitted:      bool = False

        self._ldc_cache: Dict[int, Optional[list]] = {} # REVISED+_pres


    # Phase 1: Training
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
        declare_lambda_tdc: float = 0.1,  # REVISED+_pres 
        lambda_res:        float = 0.0,  # REVISED+_pres 
        bb_run_cv:         bool = True,
        bb_cv_k:           int = 5,
        bb_precomputed_cv_summary: Optional[Dict[str, Any]] = None,
        reuse_black_box:   Optional[Any] = None,
    ) -> "LatentSpace":
        """
        Phase 1 — offline training.
        Order: black-box → VAE → transition model → reward function.

        declare_lambda_tdc > 0.0 activates the REVISED+ baseline by mining
        Trace Declare Constraints (label-agnostic, 100%-supported across the
        whole log) via rule_distillation.mine_constraints() and folds them
        into the VAE's ELBO as L_TDC.
        """
        logger.info("Phase 1: Offline training started.")
        print("Batch size:", batch_size)
        print("Declare lambda TDC (weight):", declare_lambda_tdc)
    
        self._fit_cases = cases

        self._ldc_cache = {}

        declare_constraints = None
        if declare_lambda_tdc > 0.0:
            from baseline.revised_plus.iBCM.run_iBCM import mine_constraints, reduce_feature_space

            all_traces = [[e.activity for e in c.events] for c in cases if len(c.events) >= 1]

            _tdc_min_sup = 1  # matches run_iBCM.py's own default
            _tdc_no_win = 1      # matches run_iBCM.py's own default
            _act_counts: Dict[str, int] = {}
            for trace in all_traces:
                for act in set(trace):  # count trace-presence, not per-occurrence, matching activity_count logic
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
        
        self._is_fitted = True
        logger.info("Phase 1 complete. All components trained.")
        return self

    # Helper: encode a partial trace to z_t

    def _encode_case(self, case: event_log.Case, step: int) -> torch.Tensor:
        """Encode σ_t = case.events[:step] → z_t ∈ ℝ^latent_dim."""
        mat = torch.tensor(
            case.to_matrix(step)[np.newaxis].astype(np.float32),
            device=self.device,
        )
        lengths = torch.tensor([step], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z_t = self.vae.encode_trace(mat, lengths)   # (1, latent_dim)
        return z_t

    # Helper: decode z → event sequence matrix

    def _decode_z(self, z: torch.Tensor, seq_len: int = 4) -> torch.Tensor:
        """Decode z → event sequence (B, seq_len, event_dim)."""
        return self.vae.decode_autoregressive(z, seq_len)

    @staticmethod
    def _validate_resource_activity_pairs(
        decoded_cf: np.ndarray,   # (T, event_dim)
    ) -> dict:
        """
        For each changed step in the counterfactual, check whether the
        (activity, resource) pair is empirically feasible.
        Returns a dict with validity flag and list of violations.
        """
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
    
    # Phase 2: Explanation generation

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
        kpi_significance_threshold: float = 0.05, # Meaningful KPI change threshold for tagging a CF as `kpi_significant=True` 
        kpi_min_baseline: float = 1e-3,
        editable_steps: Optional[List[int]] = None,
    ) -> Dict:
        """
        Phase 2 — online counterfactual explanation.

        Parameters
        ----------
        case : event_log.Case
            Running process case.
        step : int
            Current step index (explains recommendation at position `step`).
        surrogate_epochs : int
            Training epochs for the local surrogate.
        verbose : bool
            Print progress.
        k : int
            Number of distinct counterfactuals to return.
        max_attempts : int
            Hard cap on how many candidate targets to try (gradient
            searches are expensive — this bounds wall-clock cost even
            when the ranked action queue is much larger than `k`, e.g.
            if only 1-2 distinct CFs actually exist and the rest of a
            large action space would otherwise all be tried). If the
            cap is hit before `k` distinct CFs are found, fewer than
            `k` are returned and a warning is logged.
        contrast_mode : str
            Which joint-action targets are eligible to be searched for.
            The joint validity constraint g_ξ(z')[â^c] ≠ â^c_{t+1} is
            satisfied whenever EITHER the activity or the resource
            component flips — which previously let an activity-only CF
            get narrated as if it explained the full (activity, resource)
            recommendation, and vice versa. One of:
              "activity" — only target actions that keep the SAME resource
                            (r̂) and change ONLY the activity. Answers
                            "why this activity, holding the resource fixed?"
              "resource" — only target actions that keep the SAME activity
                            (â) and change ONLY the resource. Answers
                            "why this resource, holding the activity fixed?"
              "joint"    — only target actions where BOTH activity AND
                            resource differ from (â, r̂). Answers
                            "why this full recommendation, rather than a
                            wholly different one?"
              "any"      — (default, previous behaviour) no restriction.
            Every returned explanation is tagged with `contrast_type` (the
            actual "activity"/"resource"/"joint" classification of what
            changed) regardless of contrast_mode, so downstream evaluation
            can stratify by it even when contrast_mode="any".

        kpi_significance_threshold : float
            Minimum |% change| in predicted KPI (relative to the
            original recommendation) for a CF to be tagged
            `kpi_significant=True`. Candidates are NEVER discarded for
            failing this — they are tagged and re-ranked so the caller
            (or downstream evaluation) can filter as needed. See
            `_kpi_delta_pct` below.
        kpi_min_baseline : float
            Guards against unstable percent-change when the
            original predicted KPI is near zero; falls back to an
            absolute-difference threshold in that regime.

        Returns
        -------
        dict with keys:
          multiple_explanations (List[Dict]), surrogate_loss_history,
          n_requested, n_found, n_attempts, n_candidates_available,
          no_significant_kpi_cf (NEW), best_kpi_delta_pct (NEW)
        """
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
        
        # Instantiation wrapper
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

        # Encode to z_t
        z_t = self._encode_case(case, step)   # (1, latent_dim)

        # Query black-box
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
                    {"LacticAcid"}, {"SUMleges"},

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

        _anchor = torch.tensor(raw_matrix, dtype=torch.float32, device=self.device)  # (step, event_dim)
        _frozen_mask = torch.zeros(event_dim, dtype=torch.bool, device=self.device)
        if invariant_indices:
            _frozen_mask[invariant_indices] = True

        
        def _build_rigid_invariant_decoder(
            freeze_activity_at_last: bool = False,
            freeze_resource_at_last: bool = False,
            editable_row_indices: Optional[Set[int]] = None,  # NEW: ported from latent_mdp.py
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
                T = raw_seq.size(1)

                if invariant_indices or act_res_indices:
                    anchor_rows = _anchor[:T, :].unsqueeze(0).expand(raw_seq.size(0), -1, -1).clone()  # (B, T, event_dim)

                    if freq_indices and act_indices:
                        for k in sorted(editable_row_indices):
                            if k >= T - 1:
                                continue  # no rows after k in this prefix — nothing to correct
                            decoded_act_k = F.softmax(raw_seq[:, k, :][:, act_indices], dim=-1)  # (B, n_act)
                            anchor_act_k = _anchor[k, act_indices].unsqueeze(0)                    # (1, n_act) real one-hot
                            delta_k = decoded_act_k - anchor_act_k                                 # (B, n_act)
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
            editable_row_indices=_editable_row_indices,  # NEW
        )

        oracle = LatentOracle(self.black_box, rigid_invariant_decoder, seq_len=step, device=self.device)

        
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
                    return "none"  # should not occur (orig_action_idx already excluded)

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

        # REVISED+_pres baseline

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
        discovered_vectors = []  # Track previously discovered counterfactual vectors to avoid duplicates
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

            rank = len(all_explanations)  # rank among ACCEPTED explanations, not attempts

            target_contrast = _classify_contrast(target_idx)
            target_decoder = _build_rigid_invariant_decoder(
                freeze_activity_at_last=(target_contrast == "resource"),
                freeze_resource_at_last=(target_contrast == "activity"),
                editable_row_indices=_editable_row_indices,  
            )
            
            target_oracle = TargetedLatentOracle(
                self.black_box, target_decoder, seq_len=step, device=self.device,
                target_action_idx=target_idx, original_action_idx=orig_action_idx,
                delta=self.cf_delta,
            )

            cf_result = searcher.search(
                z_t=z_t,
                original_action_idx=orig_action_idx,
                surrogate=surrogate,
                black_box_fn=target_oracle,
                decode_fn=target_decoder,
                step=step,
                target_action_idx=target_idx, # Pass target down to optimization head,
                past_cf_vectors=discovered_vectors, # Repel from previously discovered points in latent space
                target_constraints=(
                    _get_target_constraints(target_idx) if (self.cf_lambda_ldc > 0.0 or self.cf_score_ldc) else None
                ),
                n_act=len(event_log.ACTIVITIES),
                raw_sequence_matrix=raw_matrix
            )
            
            cf_act, cf_res = idx_to_joint_action(cf_result.cf_action_idx)

            _target_feasible = event_log.is_feasible_pair(
                target_oracle.target_act, target_oracle.target_res
            )
            _attempt_record = {
                "attempt": attempts,
                "queue_pos": queue_pos - 1,
                "target_idx": int(target_idx),
                "target_act": target_oracle.target_act,
                "target_res": target_oracle.target_res,
                "target_contrast": target_contrast,
                "orig_act": orig_act,
                "orig_res": orig_res,
                "target_pair_feasible": _target_feasible,
                "t_orig": target_oracle.last_t_orig,
                "t_target": target_oracle.last_t_target,
                "margin": target_oracle.last_margin,          # t_orig - t_target
                "cf_delta": self.cf_delta,
                "oracle_flipped": target_oracle.last_flipped,  # did the oracle itself accept the flip
                "n_iterations": cf_result.n_iterations,
                "used_expulsion": cf_result.used_expulsion,
                "outcome": None,   # filled in below at whichever branch this attempt exits through
            }


            with torch.no_grad():
                _decoded_check = target_decoder(cf_result.z_cf).squeeze(0)  # (T, event_dim)
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
                if verbose:
                    logger.warning(
                        f"  --> Rejected candidate (target idx {target_idx}, "
                        f"contrast={target_contrast}): black-box predicted "
                        f"label ({cf_act}, {cf_res}) disagrees with the "
                        f"actually-decoded, frozen input ({_decoded_act}, "
                        f"{_decoded_res}). This is a black-box faithfulness "
                        f"failure, most common on out-of-distribution "
                        f"(expulsion-regime) candidates — trying next "
                        f"candidate rather than reporting a self-"
                        f"contradictory explanation."
                    )
                continue

                   
            if cf_act == orig_act and cf_res == orig_res:
                _attempt_record["outcome"] = "not_flipped"
                attempt_log.append(_attempt_record)
                continue 

            # REVISED+_pres baseline
            if self.cf_enforce_ldc and cf_result.ldc_final_violation > 1e-3:
                _attempt_record["outcome"] = "ldc_violation_rejected"
                attempt_log.append(_attempt_record)
                if verbose:
                    logger.warning(
                        f"  --> Rejected candidate (target idx {target_idx}): "
                        f"flipped the recommendation but violated a label-"
                        f"specific Declare constraint (ldc_final_violation="
                        f"{cf_result.ldc_final_violation:.4f}). REVISED+ "
                        f"fidelity mode (cf_enforce_ldc=True) requires "
                        f"L_LDC=0 for acceptance."
                    )
                continue

            if (cf_act, cf_res) in seen_actions or cf_result.cf_action_idx in seen_indices:
                _attempt_record["outcome"] = "duplicate"
                attempt_log.append(_attempt_record)
                continue

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
                decode_fn=target_decoder,
                v_orig=v_orig, v_cf=v_cf,
                case_id=case.case_id,
                step=step
            )

            with torch.no_grad():
                decoded_cf_np   = target_decoder(cf_result.z_cf)\
                                      .squeeze(0).cpu().numpy()   # (T, event_dim)

 
            resource_check = self._validate_resource_activity_pairs(
                decoded_cf = decoded_cf_np,
            )

            if not resource_check["resource_valid"]:
                logger.info(f"  --> Discarding CF (target idx {target_idx}) due to feasibility violations.")
                continue # Safely skip appending this invalid explanation
 
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
                continue # Skip appending this explanation entirely!

            seen_actions.add((cf_act, cf_res))
            seen_indices.add(cf_result.cf_action_idx)
            discovered_vectors.append(cf_result.z_cf.detach().clone())

            explanation["resource_valid"]       = resource_check["resource_valid"]
            explanation["resource_violations"]  = resource_check["violations"]
            explanation["target_rank"] = rank + 1
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
            else:  # "joint"
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


    def save(self, path: str = "latent_space_weights") -> None:
        """Save all model weights to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.black_box.save(p)
        torch.save(self.vae.state_dict(), p / "vae.pt")
        logger.info(f"Models saved to {path}/")

    def load(
        self,
        path: str = "latent_space_weights",
        declare_constraints: Optional[list] = None,
        declare_lambda_tdc: float = 0.0,
    ) -> "Pipeline":
        """
        Load all model weights from a directory.
        """
        p = Path(path)

        self.black_box  = CatBoostPPM.load(p)
       
        self.vae = LSTMVAE(event_log.EVENT_DIM, self.hidden_dim, self.latent_dim).to(self.device)

        self.vae.load_state_dict(torch.load(p / "vae.pt", map_location=self.device), strict=False)

        if declare_lambda_tdc > 0.0:
            self.vae.set_declare_constraints(declare_constraints, declare_lambda_tdc)

        for m in [self.vae]:
            m.eval()

        self._is_fitted = True
        logger.info(f"Models loaded from {path}/")
        return self
