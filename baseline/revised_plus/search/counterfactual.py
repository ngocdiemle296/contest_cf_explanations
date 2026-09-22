"""
Counterfactual Search Phase

Given a black-box recommendation for a running case, search the latent space learned by the VAE for a nearby latent point z* 
whose decoded trace would have led the black box to recommend a different (activity, resource) pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np

from baseline.revised_plus.models.vae import declare_soft_violation_batch


# Local surrogate g_ξ
class LocalSurrogate(nn.Module):
    """
    Lightweight MLP trained in the latent neighbourhood of z_t.
    """

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        feasible_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_actions_full = n_actions

        if feasible_indices is None:
            feasible_indices = torch.arange(n_actions, dtype=torch.long)
        feasible_indices = feasible_indices.to(dtype=torch.long).flatten()
        if feasible_indices.numel() == 0:
            raise ValueError("feasible_indices must be non-empty.")
        if int(feasible_indices.max()) >= n_actions or int(feasible_indices.min()) < 0:
            raise ValueError("feasible_indices contains an index outside [0, n_actions).")

        n_local = feasible_indices.numel()

        self.register_buffer("feasible_indices", feasible_indices)
        full_to_local = torch.full((n_actions,), -100, dtype=torch.long)
        full_to_local[feasible_indices] = torch.arange(n_local, dtype=torch.long)
        self.register_buffer("full_to_local", full_to_local)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_local),
        )

    def forward_local(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        local_logits = self.forward_local(z)          # (B, n_local)
        full = local_logits.new_full((z.size(0), self.n_actions_full), float("-inf"))
        full[:, self.feasible_indices] = local_logits
        return full

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z)  # (B, n_actions_full)

        from baseline.revised_plus.models.blackbox import non_recommendable_action_indices

        for bad_idx in non_recommendable_action_indices():
            logits[:, bad_idx] = -1e9

        return logits.argmax(dim=-1)

    def fit(
        self,
        z_center: torch.Tensor,           # (1, latent_dim) reference point
        black_box_fn: Callable,           # f: z → action_idx  
        n_samples: int = 500,
        rho: float = 0.3,
        n_epochs: int = 200,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> List[float]:
        """
        Sample neighbourhood, query black-box, fit surrogate with kernel weights.
        Returns training loss history.
        """
        self.train()
        self.to(device)
        z_center = z_center.to(device)

        # Draw samples 
        eps = torch.randn(n_samples, z_center.size(-1), device=device) * rho
        z_samples = z_center + eps   # (N, latent_dim)

        # Query black-box for labels 
        with torch.no_grad():
            labels_full = black_box_fn(z_samples)   # (N,) int64

        labels_local = self.full_to_local[labels_full]
        valid = labels_local != -100
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            import warnings
            warnings.warn(
                f"LocalSurrogate.fit: {n_invalid}/{n_samples} black-box labels fell "
                f"outside the restricted feasible action space and were dropped from "
                f"the training loss. If this happens often, feasible_indices is too "
                f"narrow — check RESOURCE_ACTIVITY_MATRIX's min_observations."
            )

        dist_sq = (eps ** 2).sum(dim=-1)       # (N,)
        kappa = torch.exp(-dist_sq / (2 * rho ** 2)) * valid.float()   # (N,)

        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        safe_labels_local = labels_local.clamp(min=0)  
        for _ in range(n_epochs):
            local_logits = self.forward_local(z_samples)     # (N, n_local)
            ce = F.cross_entropy(local_logits, safe_labels_local, reduction="none")   # (N,)
            denom = kappa.sum().clamp(min=1e-8)
            loss = (kappa * ce).sum() / denom
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            history.append(loss.item())

        self.eval()
        return history

    @torch.no_grad()
    def evaluate_precision(
        self,
        z_center: torch.Tensor,
        black_box_fn: Callable,
        n_actions: int,
        n_holdout: int = 500,
        rho: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Held-out precision/recall of g_ξ vs the real black-box f, evaluated on a
        fresh neighbourhood sample
        """
        if rho is None:
            raise ValueError("Pass the same rho used in .fit() explicitly — "
                              "it is not stored on the module.")

        self.eval()
        self.to(device)
        z_center = z_center.to(device)

        eps = torch.randn(n_holdout, z_center.size(-1), device=device) * rho
        z_samples = z_center + eps

        y_true = black_box_fn(z_samples).cpu().numpy()      # (N,) real labels, FULL space
        y_pred = self.predict(z_samples).cpu().numpy()      # (N,) surrogate labels, FULL space

        accuracy = float((y_true == y_pred).mean())

        classes_to_scan = self.feasible_indices.cpu().tolist()

        precision_per_class = {}
        support_per_class = {}
        for c in classes_to_scan:
            support = int((y_true == c).sum())
            if support > 0:
                support_per_class[c] = support
            n_pred_c = int((y_pred == c).sum())
            if n_pred_c == 0:
                continue   # precision undefined for classes surrogate model never predicts
            tp = int(((y_pred == c) & (y_true == c)).sum())
            precision_per_class[c] = tp / n_pred_c

        if precision_per_class:
            precision_macro = float(np.mean(list(precision_per_class.values())))
            total_support = sum(support_per_class.get(c, 0) for c in precision_per_class)
            if total_support > 0:
                precision_weighted = float(sum(
                    p * support_per_class.get(c, 0) for c, p in precision_per_class.items()
                ) / total_support)
            else:
                precision_weighted = float("nan")
        else:
            precision_macro = float("nan")
            precision_weighted = float("nan")

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "precision_per_class": precision_per_class,
            "support_per_class": support_per_class,
            "n_holdout": n_holdout,
        }


# Counterfactual search

@dataclass
class CounterfactualResult:
    z_cf: torch.Tensor              # optimal z*
    converged: bool
    n_iterations: int
    proximity: float                # How close to original z_t in latent space
    sparsity: int                   # Number of changed features in decoded trace
    verified_by_blackbox: bool      # If the pair is flipped
    cf_action_idx: int              # CF pair
    original_action_idx: int        # Original pair
    loss_history: List[float] = field(default_factory=list)
    exit_reason: str = "max_iter"
    used_expulsion: bool = False

    # REVISED+_pres baseline 
    ldc_final_violation: float = 0.0
    ldc_unavailable: bool = False


class CounterfactualSearch:
    """
    Gradient-descent search in latent space.
    """

    def __init__(
        self,
        latent_dim: int,
        lambda1: float = 0.5,    # manifold weight
        lambda2: float = 0.3,    # sparsity weight
        lambda3: float = 1.5,    # classification constrant weight
        lambda_ldc: float = 0.0,  # REVISED+_pres baseline LDC weight
        delta: float = 0.1,      # margin threshold
        lr: float = 0.05,
        max_iter: int = 200,
        patience: int = 100,
        enforce_ldc: bool = False,  # REVISED+ fidelity mode: hard-reject candidates with L_LDC > 0
        score_ldc: bool = True,  # REVISED+_pres evaluation mode: report LDC score even if lambda_ldc==0.0
        device: torch.device = torch.device("cpu"),
    ):
        self.latent_dim = latent_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_ldc = lambda_ldc
        self.delta = delta
        self.lr = lr
        self.max_iter = max_iter
        self.patience = patience
        self.enforce_ldc = enforce_ldc
        self.score_ldc = score_ldc
        self.device = device

    def _manifold_penalty(self, z: torch.Tensor) -> torch.Tensor:
    
        return 0.5 * (z ** 2).sum(dim=-1).mean()

    def _sparsity_approx(
        self,
        z_cf: torch.Tensor,
        z_orig: torch.Tensor,
        decode_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        
        if decode_fn is None:
            return (z_cf - z_orig).abs().sum(dim=-1).mean()
    
        dec_cf = decode_fn(z_cf)
        dec_orig = decode_fn(z_orig)
    
        from baseline.revised_plus.data import event_log
    
        n_act = len(event_log.ACTIVITIES)
        n_res = len(event_log.RESOURCES)
    
        act_cf, act_orig = dec_cf[..., :n_act], dec_orig[..., :n_act]
        act_dist = (F.softmax(act_cf, dim=-1) - F.softmax(act_orig, dim=-1)).abs().sum(dim=-1)
    
        if n_res > 0:
            res_cf = dec_cf[..., n_act:n_act + n_res]
            res_orig = dec_orig[..., n_act:n_act + n_res]
            res_dist = (F.softmax(res_cf, dim=-1) - F.softmax(res_orig, dim=-1)).abs().sum(dim=-1)
        else:
            res_dist = torch.zeros_like(act_dist)
    
        all_features = event_log.feature_names()
        scalar_idx_list = [
            all_features.index(n) for n in event_log._SCALAR_NAMES if n in all_features
        ]
        if scalar_idx_list:
            scalar_idx = torch.tensor(scalar_idx_list, dtype=torch.long, device=dec_cf.device)
            scalar_dist = (dec_cf[..., scalar_idx] - dec_orig[..., scalar_idx]).abs().sum(dim=-1)
        else:
            scalar_dist = torch.zeros_like(act_dist)
    
        per_timestep = act_dist + res_dist
        return per_timestep.mean() + scalar_dist.mean()

    def _ldc_penalty(
        self,
        z_cf: torch.Tensor,
        decode_fn: Optional[Callable],
        target_constraints,
        n_act: int,
    ) -> torch.Tensor:
        """
        L_LDC: REVISED+_pres baseline soft violation score against Declare constraints.
        """
        if decode_fn is None or not target_constraints:
            return torch.zeros((), device=z_cf.device)

        from baseline.revised_plus.data import event_log

        dec = decode_fn(z_cf)  # (1, T, event_dim) or (1, event_dim)
        if dec.dim() == 2:
            dec = dec.unsqueeze(1)
        O = F.softmax(dec[..., :n_act], dim=-1)  # (1, T, n_act)

        names, a_idxs, b_idxs = [], [], []
        for c in target_constraints:
            a_idx = event_log.ACT2IDX.get(c.a)
            b_idx = event_log.ACT2IDX.get(c.b)
            if a_idx is None or b_idx is None:
                continue
            names.append(c.name)
            a_idxs.append(a_idx)
            b_idxs.append(b_idx)

        return declare_soft_violation_batch(names, a_idxs, b_idxs, O)

    def search(
        self,
        z_t: torch.Tensor,                   
        original_action_idx: int,            
        surrogate: LocalSurrogate,           
        black_box_fn: Callable,              
        decode_fn: Optional[Callable] = None,
        step: int = None,                      # current trace step 
        target_action_idx: Optional[int] = None,  # Instead of simply pushing away from the original recommendation, the loss function will actively pull the latent coordinates toward that specific target alternative.
        past_cf_vectors: Optional[List[torch.Tensor]] = None,
        target_constraints=None,              # REVISED+_pres baseline: List[iBCM.Constraint] for this target
        n_act: Optional[int] = None,          # required if target_constraints is used
        raw_sequence_matrix: Optional[np.ndarray] = None,
    ) -> CounterfactualResult:
        """
        Run optimisation to find z* that flips the recommendation.
        """
       
        z_t = z_t.to(self.device).detach()

        # Initialise z' near z_t with small noise
        z_cf = (z_t + 0.01 * torch.randn_like(z_t)).requires_grad_(True)
        optimiser = torch.optim.Adam([z_cf], lr=self.lr)

        best_z   = z_cf.detach().clone()
        best_loss = float("inf")
        stagnant  = 0
        loss_hist: List[float] = []

        exit_reason   = "max_iter"   
        used_expulsion = False       
        ldc_unavailable = False      

        surrogate.eval()

        for it in range(self.max_iter):
            optimiser.zero_grad()

            # Proximity: d(z_t, z')
            prox = ((z_t - z_cf) ** 2).sum()

            # Manifold penalty
            l_man = self._manifold_penalty(z_cf)

            # Sparsity (L1)
            l_spar = self._sparsity_approx(z_cf, z_t, decode_fn)

            # Classification constraint: max-margin hinge
            logits    = surrogate(z_cf)
            act_score = logits[0, original_action_idx]

            if target_action_idx is not None:
                target_score = logits[0, target_action_idx]
                margin_diff = act_score - target_score + self.delta
            else:
                other_scores = torch.cat([logits[0, :original_action_idx], logits[0, original_action_idx + 1:]])
                max_other = other_scores.max()
                margin_diff = act_score - max_other + self.delta

            l_class = torch.log1p(torch.exp(margin_diff))
            
            # REVISED+_pres baseline: LDC constraints penalty.
            if self.lambda_ldc > 0.0:
                l_ldc = self._ldc_penalty(z_cf, decode_fn, target_constraints, n_act or 0)
            else:
                l_ldc = torch.zeros((), device=z_cf.device)

            l_repulsion = 0.0
            if past_cf_vectors:
                for past_z in past_cf_vectors:
                    distance_sq = ((z_cf - past_z.to(self.device)) ** 2).sum()
                    l_repulsion += torch.exp(-distance_sq * 2.0)

            EXPULSION_STAGNATION_THRESHOLD = 15
            is_genuinely_stuck = (it > 10) and (stagnant > EXPULSION_STAGNATION_THRESHOLD)
            if is_genuinely_stuck:
                used_expulsion = True

            if is_genuinely_stuck:
                expulsion_penalty = torch.exp(-((z_cf - z_t) ** 2).sum())
                loss = (prox + self.lambda1 * l_man + self.lambda2 * l_spar + self.lambda3 * l_class
                        + self.lambda_ldc * l_ldc + 2.0 * expulsion_penalty + 5.0 * l_repulsion)
            else:
                loss = (prox + self.lambda1 * l_man + self.lambda2 * l_spar + self.lambda3 * l_class
                        + self.lambda_ldc * l_ldc + 5.0 * l_repulsion)

            loss.backward()

            with torch.no_grad():
                if is_genuinely_stuck and z_cf.grad is not None:
                    z_cf.grad.data.mul_(1.5)

            optimiser.step()

            with torch.no_grad():
                deviation = z_cf - z_t
                clipped_deviation = torch.clamp(deviation, min=-1.5, max=1.5)
                z_cf.copy_(z_t + clipped_deviation)

            loss_hist.append(loss.item())

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z    = z_cf.detach().clone()
                stagnant  = 0
            else:
                stagnant += 1

            # Check if surrogate already flips
            with torch.no_grad():
                # Query the real black-box via the latent oracle wrapper
                real_bb_pred = black_box_fn(z_cf).item()

                # ONLY stop early if we have truly, verified-flipped the real model!
                flipped = (
                    (real_bb_pred == target_action_idx) if target_action_idx is not None
                    else (real_bb_pred != original_action_idx)
                )
                if flipped:
                    # REVISED+_pres baseline: hard L_LDC=0 acceptance gate (Algorithm 1 Stevens et al.)
                    if self.enforce_ldc:
                        if not target_constraints:
                            # No LDC set could be mined for this target
                            best_z = z_cf.detach().clone()
                            exit_reason = "flipped"
                            ldc_unavailable = True
                            break
                        with torch.no_grad():
                            ldc_final = self._ldc_penalty(z_cf, decode_fn, target_constraints, n_act or 0).item()
                        if ldc_final > 1e-3:
                            # Candidate flips the recommendation but violates
                            continue
                        best_z = z_cf.detach().clone()
                        exit_reason = "flipped"
                        break
                    else:
                        best_z = z_cf.detach().clone()
                        exit_reason = "flipped"
                        break

            if stagnant >= self.patience:
                exit_reason = "patience_exhausted"
                break

        # Verification against real black-box 
        with torch.no_grad():
            cf_action = black_box_fn(best_z).item()
            verified   = (cf_action != original_action_idx)

        # REVISED+_pres baseline: report the LDC score
        if (self.lambda_ldc > 0.0 or self.score_ldc) and target_constraints:
            with torch.no_grad():
                ldc_final_violation = self._ldc_penalty(best_z, decode_fn, target_constraints, n_act or 0).item()
        else:
            ldc_final_violation = 0.0
            if (self.lambda_ldc > 0.0 or self.score_ldc) and not target_constraints:
                ldc_unavailable = True

        if self.enforce_ldc and target_constraints and ldc_final_violation > 1e-3:
            verified = False
            if exit_reason == "flipped":
                exit_reason = "ldc_rejected"
            elif exit_reason in ("patience_exhausted", "max_iter"):
                exit_reason = f"{exit_reason}_ldc_violated"

        # Proximity in latent space
        prox_val = ((z_t - best_z) ** 2).sum().item()

        # Sparsity count in decoded space
        if decode_fn is not None:

            with torch.no_grad():
                d_cf   = decode_fn(best_z).squeeze(0)

            if raw_sequence_matrix is not None:
                if isinstance(raw_sequence_matrix, np.ndarray):
                    d_orig = torch.as_tensor(
                        raw_sequence_matrix, dtype=d_cf.dtype, device=d_cf.device
                    )
                else:
                    d_orig = raw_sequence_matrix
                use_ground_truth = True
            else:
                with torch.no_grad():
                    d_orig = decode_fn(z_t).squeeze(0)
                use_ground_truth = True

            if d_orig.dim() == 2:
                from baseline.revised_plus.data import event_log
                n_act = len(event_log.ACTIVITIES)
                n_res = len(event_log.RESOURCES)

                sparsity = 0
                T = d_orig.size(0)
                for t in range(T):
                    o_act = int(torch.argmax(d_orig[t, :n_act]).item())
                    c_act = int(torch.argmax(d_cf[t, :n_act]).item())
                    o_res = int(torch.argmax(d_orig[t, n_act:n_act + n_res]).item()) if n_res > 0 else 0
                    c_res = int(torch.argmax(d_cf[t, n_act:n_act + n_res]).item()) if n_res > 0 else 0
                    if o_act != c_act or o_res != c_res:
                        sparsity += 1

                all_features = event_log.feature_names()
                scalar_indices = [
                    all_features.index(name)
                    for name in event_log._SCALAR_NAMES
                    if name in all_features
                ]
                if scalar_indices:
                    free_scalar_alias_groups = [
                        {"AMOUNT_REQ", "RequestedAmount"},
                        {"Leucocytes"},
                        {"CRP"},
                        {"LacticAcid"}, {"case:SUMleges"},
                    ]
                    free_scalar_names = set().union(*free_scalar_alias_groups)
 
                    idx_t = torch.tensor(scalar_indices, dtype=torch.long, device=d_orig.device)
                    scalar_delta = (d_cf[T - 1, idx_t] - d_orig[T - 1, idx_t]).abs()
 
                    for name, delta in zip(
                        (all_features[i] for i in scalar_indices), scalar_delta
                    ):
                        if name in free_scalar_names:
                            sparsity += 1  # free scalar: unconditional, matches blackbox.py
                        elif delta.item() > 0.05:
                            sparsity += 1  # 

                free_categorical_group_names = ["LoanGoal", "ApplicationType", "CLOSURE_TYPE", "CLOSURE_REASON"]
                for _grp_name in free_categorical_group_names:
                    _col_names = event_log.categorical_block_columns(_grp_name)
                    _idx = [all_features.index(n) for n in _col_names if n in all_features]
                    if not _idx:
                        continue
                    _idx_t = torch.tensor(_idx, dtype=torch.long, device=d_orig.device)
                    o_local = int(torch.argmax(d_orig[T - 1, _idx_t]).item())
                    c_local = int(torch.argmax(d_cf[T - 1, _idx_t]).item())
                    if o_local != c_local:
                        sparsity += 1
                        
            else:
                delta_dec = (d_cf - d_orig).abs()
                sparsity = int((delta_dec > 0.05).sum().item())
        else:
            sparsity = int(((best_z - z_t).abs() > 0.05).sum().item())

        return CounterfactualResult(
            z_cf=best_z,
            converged=(stagnant < self.patience),
            n_iterations=len(loss_hist),
            proximity=prox_val,
            sparsity=sparsity,
            verified_by_blackbox=verified,
            cf_action_idx=cf_action,
            original_action_idx=original_action_idx,
            loss_history=loss_hist,
            exit_reason=exit_reason,
            used_expulsion=used_expulsion,
            ldc_final_violation=ldc_final_violation,
            ldc_unavailable=ldc_unavailable,
        )
