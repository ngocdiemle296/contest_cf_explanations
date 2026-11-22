import torch
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import pearsonr

from baseline.revised_plus.search.counterfactual import CounterfactualResult
from baseline.revised_plus.data import event_log


def _controllable_features() -> set:
    return {
        *{f"res_{r}" for r in event_log.RESOURCES},
        "workload_res_norm",
        "num_missing_docs_norm",
    }

VALID_NEXT_ACTS = {
    "A_SUBMITTED":                   {"A_PARTLYSUBMITTED", "A_DECLINED"},
    "A_PARTLYSUBMITTED":             {"W_Completeren aanvraag", "A_DECLINED"},
    "W_Completeren aanvraag":        {"A_PREACCEPTED", "W_Nabellen incomplete dossiers", "A_DECLINED"},
    "A_PREACCEPTED":                 {"W_Valideren aanvraag", "W_Nabellen offertes", "A_DECLINED"},
    "W_Valideren aanvraag":          {"A_ACCEPTED", "W_Nabellen offertes", "A_DECLINED"},
    "W_Nabellen offertes":           {"A_ACCEPTED", "A_DECLINED"},
    "A_ACCEPTED":                    {"A_FINALIZED", "A_DECLINED"},
    "W_Nabellen incomplete dossiers":{"W_Completeren aanvraag", "A_DECLINED"},
}


@dataclass
class EvaluationReport:
    validity:           float
    proximity_mean:     float
    proximity_std:      float
    feasibility_mean:   float
    sparsity_mean:      float
    sparsity_std:       float
    temporal_coherence: float
    future_fidelity:    float
    actionability:      float
    n_samples:          int
    details:            List[Dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  COUNTERFACTUAL EXPLANATION EVALUATION REPORT\n"
            f"{'='*55}\n"
            f"  Samples evaluated    : {self.n_samples}\n"
            f"{'─'*55}\n"
            f"  Validity             : {self.validity:.3f}   (↑ higher is better)\n"
            f"  Proximity (mean±std) : {self.proximity_mean:.4f} ± {self.proximity_std:.4f}  (↓)\n"
            f"  Feasibility (ELBO↑)  : {self.feasibility_mean:.4f}\n"
            f"  Sparsity (mean±std)  : {self.sparsity_mean:.2f} ± {self.sparsity_std:.2f}  (↓)\n"
            f"  Temporal Coherence   : {self.temporal_coherence:.3f}   (↑)\n"
            f"  Future Fidelity (ρ)  : {self.future_fidelity:.3f}   (↑)\n"
            f"  Actionability        : {self.actionability:.3f}   (↑)\n"
            f"{'='*55}\n"
        )


def compute_validity(results: List[CounterfactualResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r.verified_by_blackbox for r in results]))


def compute_proximity(results: List[CounterfactualResult]) -> Tuple[float, float]:
    vals = [np.sqrt(r.proximity) for r in results]
    return float(np.mean(vals)), float(np.std(vals))


def compute_feasibility(
    cf_z_list: List[torch.Tensor],
    vae,
    device: torch.device,
) -> float:
    scores = []
    for z in cf_z_list:
        z = z.to(device)
        log_prior = -0.5 * (z ** 2).sum().item()
        scores.append(log_prior)
    return float(np.mean(scores))


def compute_sparsity(results: List[CounterfactualResult]) -> Tuple[float, float]:
    vals = [r.sparsity for r in results]
    return float(np.mean(vals)), float(np.std(vals))


def compute_temporal_coherence(
    cf_decoded_list: List[np.ndarray],
    feature_names: List[str],
) -> float:
    if not cf_decoded_list:
        return 0.0

    act_start = len(event_log.ACTIVITIES)
    valid_count = 0

    for vec in cf_decoded_list:
        if vec.ndim == 2:
            vec = vec[-1]

        act_probs = vec[:len(event_log.ACTIVITIES)]
        pred_act_idx = int(np.argmax(act_probs))
        pred_act = event_log.ACTIVITIES[pred_act_idx]

        all_valid_targets = set()
        for targets in VALID_NEXT_ACTS.values():
            all_valid_targets |= targets
        if pred_act in all_valid_targets:
            valid_count += 1

    return valid_count / len(cf_decoded_list)


def compute_future_fidelity(
    delta_V_predicted: List[float],
    delta_V_actual:    List[float],
) -> float:
    if len(delta_V_predicted) < 3:
        return float("nan")
    r, _ = pearsonr(delta_V_predicted, delta_V_actual)
    return float(r)


def compute_actionability(
    changed_features_list: List[List[Dict]],
    controllable: Optional[set] = None,
) -> float:
    if controllable is None:
        controllable = _controllable_features()

    ratios = []
    for cf_features in changed_features_list:
        if not cf_features:
            ratios.append(1.0)
            continue
        n_ctrl = sum(1 for f in cf_features if f["feature"] in controllable)
        ratios.append(n_ctrl / len(cf_features))
    return float(np.mean(ratios)) if ratios else 0.0


@dataclass
class StratifiedEvaluationReport:
    overall:          EvaluationReport
    by_contrast_type: Dict[str, EvaluationReport] = field(default_factory=dict)
    counts:           Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["\n" + "=" * 55,
                  "  STRATIFIED COUNTERFACTUAL EVALUATION REPORT",
                  "=" * 55,
                  "\n[OVERALL — pooled across all contrast types]",
                  str(self.overall)]
        for ctype in ("activity", "resource", "joint"):
            if ctype in self.by_contrast_type:
                lines.append(f"\n[CONTRAST TYPE: {ctype}  (n={self.counts.get(ctype, 0)})]")
                lines.append(str(self.by_contrast_type[ctype]))
        extra = set(self.by_contrast_type) - {"activity", "resource", "joint"}
        for ctype in sorted(extra):
            lines.append(f"\n[CONTRAST TYPE: {ctype}  (n={self.counts.get(ctype, 0)})]")
            lines.append(str(self.by_contrast_type[ctype]))
        return "\n".join(lines)


def evaluate_explanations_stratified(
    results:              List[CounterfactualResult],
    cf_z_list:            List[torch.Tensor],
    cf_decoded_list:      List[np.ndarray],
    changed_features_list: List[List[Dict]],
    feature_names:        List[str],
    delta_V_predicted:    List[float],
    delta_V_actual:       List[float],
    vae,
    device: torch.device,
    contrast_types:       List[str],
) -> StratifiedEvaluationReport:
    n = len(results)
    if not (len(cf_z_list) == len(cf_decoded_list) == len(changed_features_list)
            == len(contrast_types) == n):
        raise ValueError(
            "evaluate_explanations_stratified: results, cf_z_list, "
            "cf_decoded_list, changed_features_list, and contrast_types must "
            "all be the same length and index-aligned to the same CFs "
            f"(got lengths {n}, {len(cf_z_list)}, {len(cf_decoded_list)}, "
            f"{len(changed_features_list)}, {len(contrast_types)})."
        )
    delta_v_alignable = (len(delta_V_predicted) == n and len(delta_V_actual) == n)
    if not delta_v_alignable and (delta_V_predicted or delta_V_actual):
        import warnings
        warnings.warn(
            "evaluate_explanations_stratified: delta_V_predicted/delta_V_actual "
            "are not index-aligned to the per-CF lists (different length) — "
            "future_fidelity will be computed on the FULL lists for every "
            "stratum (i.e. identical across strata), not split by contrast_type."
        )

    overall = evaluate_explanations(
        results, cf_z_list, cf_decoded_list, changed_features_list,
        feature_names, delta_V_predicted, delta_V_actual, vae, device,
    )

    by_contrast_type: Dict[str, EvaluationReport] = {}
    counts: Dict[str, int] = {}
    for ctype in sorted(set(contrast_types)):
        idxs = [i for i, c in enumerate(contrast_types) if c == ctype]
        counts[ctype] = len(idxs)
        if not idxs:
            continue
        sub_results          = [results[i] for i in idxs]
        sub_cf_z             = [cf_z_list[i] for i in idxs]
        sub_cf_decoded       = [cf_decoded_list[i] for i in idxs]
        sub_changed_features = [changed_features_list[i] for i in idxs]
        sub_dv_pred = [delta_V_predicted[i] for i in idxs] if delta_v_alignable else delta_V_predicted
        sub_dv_act  = [delta_V_actual[i] for i in idxs] if delta_v_alignable else delta_V_actual

        by_contrast_type[ctype] = evaluate_explanations(
            sub_results, sub_cf_z, sub_cf_decoded, sub_changed_features,
            feature_names, sub_dv_pred, sub_dv_act, vae, device,
        )

    return StratifiedEvaluationReport(
        overall=overall,
        by_contrast_type=by_contrast_type,
        counts=counts,
    )


def evaluate_explanations(
    results:              List[CounterfactualResult],
    cf_z_list:            List[torch.Tensor],
    cf_decoded_list:      List[np.ndarray],
    changed_features_list: List[List[Dict]],
    feature_names:        List[str],
    delta_V_predicted:    List[float],
    delta_V_actual:       List[float],
    vae,
    device: torch.device,
) -> EvaluationReport:

    validity           = compute_validity(results)
    prox_mean, prox_std = compute_proximity(results)
    feasibility        = compute_feasibility(cf_z_list, vae, device)
    spar_mean, spar_std = compute_sparsity(results)
    temp_coh           = compute_temporal_coherence(cf_decoded_list, feature_names)
    fut_fid            = compute_future_fidelity(delta_V_predicted, delta_V_actual)
    actionability      = compute_actionability(changed_features_list)

    return EvaluationReport(
        validity=validity,
        proximity_mean=prox_mean,
        proximity_std=prox_std,
        feasibility_mean=feasibility,
        sparsity_mean=spar_mean,
        sparsity_std=spar_std,
        temporal_coherence=temp_coh,
        future_fidelity=fut_fid,
        actionability=actionability,
        n_samples=len(results),
    )
