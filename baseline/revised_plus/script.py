"""
Main script for running the REVISED+_pres baseline
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

from baseline.revised_plus.data.event_log_csv import load_event_log_csv
from baseline.revised_plus.data import event_log as el
from baseline.revised_plus.pipeline import Pipeline
from baseline.revised_plus.data import domain_persistence


# Milestone slice-index computation

def milestone_slice_idx(n: int, milestone: float) -> int:
    """
    slice_idx = max(1, min(round(n * milestone), n - 1))

    n         : full trace length (number of events) for this case
    milestone : fraction in (0, 1), e.g. 0.2 for 20%, 0.5 for 50%

    """
    if n < 2:
        raise ValueError(
            f"Cannot compute a milestone slice for a trace with only {n} event(s) "
            f"— need at least 2 events (1 to encode, 1 to explain)."
        )
    raw = round(n * milestone)
    return max(1, min(raw, n - 1))

def compute_length_buckets(case_lengths, n_buckets: int = 4, method: str = "quantile"):
    """
    case_lengths : array-like of trace lengths (n_events) for all test cases.
    method       : "quantile" (equal-count buckets, comparable per-bucket
                   statistical power) or "log" (log-spaced edges — better if
                   a long tail of very long cases would otherwise dominate
                   the top quantile bucket).

    Returns (edges, labels). edges has len(labels) + 1 entries. 
    """
    case_lengths = np.asarray(case_lengths)

    if method == "quantile":
        quantiles = np.linspace(0, 1, n_buckets + 1)
        edges = np.unique(np.quantile(case_lengths, quantiles))
    elif method == "log":
        lo, hi = case_lengths.min(), case_lengths.max()
        edges = np.unique(np.round(
            np.logspace(np.log10(max(lo, 1)), np.log10(max(hi, lo + 1)), n_buckets + 1)
        ).astype(int))
    else:
        raise ValueError(f"unknown bucket method {method!r}")

    if len(edges) < 3:
        raise ValueError(
            f"Length-bucket edges collapsed to {edges.tolist()} — the test set's "
            f"length distribution is too concentrated for n_buckets={n_buckets}. "
            f"Reduce --n-buckets."
        )

    labels = [f"bucket{i}_[{int(edges[i])},{int(edges[i + 1])})" for i in range(len(edges) - 1)]
    return edges, labels


def save_bucket_edges(path: str, edges: np.ndarray, labels: list, method: str, n_buckets: int) -> None:
    payload = {
        "edges": edges.tolist(),
        "labels": labels,
        "bucket_method": method,
        "n_buckets": n_buckets,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_bucket_edges(path: str):
    with open(path) as f:
        payload = json.load(f)
    return (
        np.asarray(payload["edges"]),
        payload["labels"],
        payload["bucket_method"],
        payload["n_buckets"],
    )

def assign_length_bucket(n_events: int, edges: np.ndarray, labels: list) -> str:
    idx = np.searchsorted(edges, n_events, side="right") - 1
    idx = min(max(idx, 0), len(labels) - 1)
    return labels[idx]


def compute_milestone_slices(n: int, milestones, min_case_length: int = 5):
    """
    Returns a list of (milestone, slice_idx) pairs for one case.
    Cases with n < min_case_length return an empty list.
    """
    if n < min_case_length:
        return []

    seen = set()
    out = []
    for m in milestones:
        step = milestone_slice_idx(n, m)
        if step in seen:
            continue
        seen.add(step)
        out.append((m, step))
    return out


def plausibility_score(z_cf: torch.Tensor) -> Tuple[float, float]: # Manifold Feasibility 
    with torch.no_grad():
        l_man = 0.5 * (z_cf ** 2).sum(dim=-1).mean().item()
    return l_man, -l_man


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the CF explanation experiment over a user-provided "
                    "test set at a fixed per-case milestone cut point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--dataset-name", type=str, default=None,
                       help="Name of the dataset, used for labeling and reporting purposes.")
    p.add_argument("--train-csv", required=True, help="Path to the training-set CSV.")
    p.add_argument("--test-csv", required=True, help="Path to the test-set CSV.")

    # Milestone 
    p.add_argument("--milestones", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8],
                   help="Cut-point fractions in (0, 1), e.g. 0.2 0.4 0.6 0.8. "
                        "NOT percentage integers — pass 0.2, not 20. Applied WITHIN "
                        "each length bucket (see --n-buckets), not globally, so a "
                        "given milestone corresponds to a comparable absolute prefix "
                        "length across the cases it's applied to.")
    p.add_argument("--n-buckets", type=int, default=2,
                   help="Number of case-length buckets to stratify the test set into "
                        "before applying milestones. Bucket edges are derived from "
                        "the test set's own empirical length distribution.")
    p.add_argument("--bucket-method", type=str, default="quantile",
                   choices=["quantile", "log"],
                   help="'quantile' = equal-count buckets (comparable statistical "
                        "power per bucket). 'log' = log-spaced edges (better when a "
                        "long tail of very long cases would otherwise dominate the "
                        "top quantile bucket).")
    p.add_argument("--min-case-length", type=int, default=5,
                   help="Cases with fewer events than this are excluded from "
                        "milestone-based sampling (too short to meaningfully "
                        "differentiate milestones — see rounding-collapse note in "
                        "compute_milestone_slices) and instead get a single "
                        "fixed-prefix explanation, reported separately as bucket "
                        "'excluded_short'.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the pre-flight (length_bucket, milestone) case-count "
                        "cross-tab and exit BEFORE running any explain() calls. Use "
                        "this to check --n-buckets/--milestones/--min-case-length are "
                        "well-calibrated for your test set's size before committing "
                        "to the full (expensive) run.")
    p.add_argument("--contrast-mode", type=str, default="joint",
                   choices=["activity", "resource", "joint", "any"],
                   help="Which joint-action targets explain() is allowed to "
                        "search for. See LatentMDP.explain()'s docstring.")
    p.add_argument("--cf-k", type=int, default=5,
                   help="Max number of distinct CFs to generate/save per test case.")
    p.add_argument("--limit-cases", type=int, default=None,
                   help="Only process N test cases (after loading, before "
                        "bucketing/milestone slicing). Use this to validate "
                        "a code change (e.g. the contrast-aware decoder fix) "
                        "on a small, fast subset before committing to a "
                        "full re-run — e.g. --limit-cases 20.")
    p.add_argument("--case-offset", type=int, default=0,
                   help="Skip the first N cases before applying --limit-cases. "
                        "NOTE: this relies on test_cases ordering being "
                        "identical across separate script invocations "
                        "(same --test-csv, same pandas groupby behavior) — "
                        "if you're not confident that holds, use "
                        "--skip-processed-dir instead, which is robust to "
                        "ordering entirely.")
    p.add_argument("--bucket-edges-file", type=str, default=None,
                   help="Path to a JSON file of length-bucket edges (as "
                        "written by this script). If the file exists, edges "
                        "are LOADED from it (and --n-buckets/--bucket-method "
                        "are ignored — the file's own values are used, with "
                        "a warning if you passed different ones), so every "
                        "batch invocation buckets against the SAME fixed "
                        "edges. If the file does NOT exist, edges are "
                        "computed from the FULL --test-csv (before "
                        "--case-offset/--limit-cases are applied) and saved "
                        "here for subsequent batches to reuse. Required for "
                        "coherent bucket labels when processing a test set "
                        "in multiple separate batched runs; default "
                        "location is {weights-dir}/bucket_edges.json if not "
                        "given explicitly.")
    p.add_argument("--skip-processed-dir", type=str, default=None, nargs="+",
                   help="Path(s) to one or more explanations/ directories from PRIOR run(s) "
                        "(e.g. --output-dir's explanations/ subfolder). Any "
                        "case whose case_id already has at least one JSON "
                        "file in this directory is skipped — matched by the "
                        "actual case_id recorded inside each JSON, not by "
                        "position/order. Use this instead of --case-offset "
                        "whenever you're not 100% sure test_cases will be "
                        "ordered identically across runs (e.g. different "
                        "--test-csv path, re-exported CSV, different "
                        "pandas/environment) — this guarantees no overlap "
                        "regardless of ordering. Can be combined with "
                        "--limit-cases to process the next N NOT-yet-done "
                        "cases.")

    # Model weights: load existing, or train fresh from --train-csv
    p.add_argument("--weights-dir", type=str, default="demo/weights_milestone",
                   help="Directory to load trained weights from (if present) or save "
                        "newly-trained weights to.")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining from --train-csv even if --weights-dir "
                        "already contains a complete set of weights.")
    p.add_argument("--reuse-black-box-from", type=str, default=None,
                   help="Path to a weights directory containing an existing "
                        "black_box.cbm to REUSE (not retrain) during --retrain. "
                        "Without this, --retrain silently retrains the black-box "
                        "too, which breaks cross-run comparability -- the "
                        "black-box is meant to be a fixed query oracle (Section "
                        "2) shared identically across every arm/run. Use this "
                        "whenever the retrain's purpose is isolating a VAE-side "
                        "change (lambda_res, gamma_conf, TDC, ...) rather than "
                        "intentionally training a new black-box.")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to save results. Defaults to "
                        "{weights-dir}/milestone_sweep_{contrast-mode}.")
    p.add_argument("--debug-attempts-dir", type=str, default=None,
                   help="If set, explain() writes one JSON per (case, step) "
                        "to this directory with a record for EVERY candidate "
                        "target attempted — not just accepted CFs — "
                        "including target pair feasibility, predicted-time "
                        "margin vs. cf_delta, and rejection reason "
                        "(faithfulness_check_failed / not_flipped / "
                        "duplicate / accepted_stage1). Use this to diagnose "
                        "why n_attempts is high but n_found is low for a "
                        "given contrast type — e.g. "
                        "--debug-attempts-dir {output-dir}/debug_attempts.")

    # CF search hyperparameters 
    p.add_argument("--cf-max-iter", type=int, default=500)
    p.add_argument("--cf-lr", type=float, default=0.1)
    p.add_argument("--cf-lambda1", type=float, default=5.0)
    p.add_argument("--cf-lambda2", type=float, default=5.0)
    p.add_argument("--cf-lambda3", type=float, default=5.0)
    p.add_argument("--cf-delta", type=float, default=0.02)
    p.add_argument("--surrogate-epochs", type=int, default=500)
    p.add_argument("--editable-history-steps", type=int, default=5,
                    help="Number of historical steps (immediately before the "
                         "current recommended step) to additionally free for "
                         "the CF search, on top of the always-editable last "
                         "step. 0 (default) reproduces the original "
                         "single-step-only behavior. E.g. 3 at step=16 frees "
                         "steps 13,14,15,16. Multi-step CFs are tagged "
                         "history_edit_multi_step=True / editable_steps_used "
                         "in the output — keep these in a separate results "
                         "slice from single-step CFs, same as expulsion-regime "
                         "CFs.")
    
    # KPI-delta significance
    p.add_argument("--kpi-significance-threshold", type=float, default=0.05,
                    help="Minimum |percent change| in the black box's "
                         "predicted KPI (predict_time_for_pair, relative to "
                         "the ORIGINAL recommendation) for a CF to be tagged "
                         "kpi_significant=True. Candidates are never "
                         "discarded for failing this — only tagged/re-ranked.")
    p.add_argument("--kpi-min-baseline", type=float, default=1e-3,
                    help="If the original prediction's magnitude falls below "
                         "this, percent change is numerically unstable, so "
                         "the significance check falls back to an absolute-"
                         "KPI-unit delta instead.")
    
    # Training hyperparameters — only used if a fresh train is triggered
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=124)
    p.add_argument("--beta-vae", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epochs-vae", type=int, default=100) 
    p.add_argument("--epochs-transition", type=int, default=50)
    p.add_argument("--epochs-reward", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512) #from 32
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--surrogate-n", type=int, default=2000) 
    p.add_argument("--surrogate-rho", type=float, default=0.15)
    p.add_argument("--use-conformance", action="store_true", default=True)
    p.add_argument("--gamma-conf", type=float, default=0.0)
    p.add_argument("--declare-lambda-tdc", type=float, default=0.1,
                    help="REVISED+ baseline arm (Phase 1): weight for the Trace Declare "
                         "Constraint loss, mined label-agnostically over the full TRAIN log. "
                         "0.0 (default) = no-op, your own method's normal training is unaffected. "
                         "Set >0 with --gamma-conf 0.0 for a TDC-only ablation, or leave "
                         "--gamma-conf at its default for the combined DIFF-ERO+TDC arm.")
    p.add_argument("--lambda-res", type=float, default=0.1,
                    help="Weight for the resource-activity feasibility training loss")
    p.add_argument("--cf-lambda-ldc", type=float, default=1.0,
                    help="REVISED+ baseline arm (Phase 2): weight for the per-target "
                         "Label-specific Declare Constraint loss during CF search. "
                         "0.0 (default) = no-op.")
    p.add_argument("--cf-enforce-ldc", action="store_true",
                    help="REVISED+ fidelity mode: hard-reject CF candidates with L_LDC > 0 "
                         "(matches Algorithm 1's 'F(sigma_CF) > p and L_LDC = 0' acceptance "
                         "rule) rather than only soft-penalizing via --cf-lambda-ldc's loss "
                         "term. Meaningless unless --cf-lambda-ldc > 0.")
    p.add_argument("--device", type=str, default="cpu")

    # Black-box CV hyperparameters — only used if a fresh train is triggered
    p.add_argument("--skip-bb-cv", action="store_true",
                    help="Skip the black-box k-fold CV before the final fit "
                         "(only used if a fresh train is triggered). If set "
                         "and --bb-cv-summary-path is NOT given, CV runs "
                         "anyway with a warning — see train_black_box().")
    p.add_argument("--bb-cv-k", type=int, default=5,
                    help="Number of folds for black-box CV (ignored if "
                         "--skip-bb-cv is set together with "
                         "--bb-cv-summary-path).")
    p.add_argument("--bb-cv-summary-path", type=str, default=None,
                    help="Path to a JSON file (matching cross_validate_"
                         "black_box()'s output schema: rmse_mean, rmse_std, "
                         "mae_mean, mae_std, neg_rate_mean, neg_rate_std, k) "
                         "to attach as model.cv_summary instead of "
                         "recomputing it. Only takes effect together with "
                         "--skip-bb-cv. Only valid if train_data.csv and "
                         "the black-box hyperparameters haven't changed "
                         "since that summary was computed.")

    args = p.parse_args()

    bad = [m for m in args.milestones if not (0.0 < m < 1.0)]
    if bad:
        p.error(f"--milestones values must each be a fraction strictly between 0 and 1, got {bad}")
    if len(set(args.milestones)) != len(args.milestones):
        p.error(f"--milestones contains duplicate values: {args.milestones}")
    if args.n_buckets < 1:
        p.error(f"--n-buckets must be >= 1, got {args.n_buckets}")
    if args.min_case_length < 2:
        p.error(f"--min-case-length must be >= 2 (need at least 1 history event + 1 "
                 f"future event), got {args.min_case_length}")

    return args


def banner(title: str) -> None:
    width = 65
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# Main

def main() -> None:
    args = parse_args()

    out_dir = args.output_dir or os.path.join(
        args.weights_dir, f"milestone_{args.contrast_mode}"
    )
    os.makedirs(out_dir, exist_ok=True)

    banner("Length-Bucket x Milestone CF Explanation Experiment")
    print(f"  Train CSV        : {args.train_csv}")
    print(f"  Test CSV         : {args.test_csv}")
    print(f"  Milestones       : {args.milestones}")
    print(f"  Length buckets   : {args.n_buckets} ({args.bucket_method})")
    print(f"  Min case length  : {args.min_case_length} (shorter cases -> "
          f"single fixed-prefix pass, bucket='excluded_short')")
    print(f"  Contrast mode    : {args.contrast_mode}")
    print(f"  cf_k             : {args.cf_k}")
    print(f"  Limit cases      : {args.limit_cases if args.limit_cases is not None else '(none — full test set)'}")
    print(f"  Case offset      : {args.case_offset}")
    print(f"  Debug attempts   : {args.debug_attempts_dir if args.debug_attempts_dir else '(off)'}")
    print(f"  skip_bb_cv         : {args.skip_bb_cv}  (only used if a fresh train is triggered)")
    print(f"  bb_cv_k            : {args.bb_cv_k}")
    print(f"  bb_cv_summary_path : {args.bb_cv_summary_path if args.bb_cv_summary_path else '(none — CV recomputed if triggered)'}")
    print(f"  Weights dir      : {args.weights_dir}")
    print(f"  Output dir       : {out_dir}")

    # ── 1. Data — train and test are two separate ──────
    section("1 / Data loading")

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    train_cases = load_event_log_csv(args.train_csv, sample_n_cases=None, dataset_label=args.dataset_name)
    print(f"  Loaded train CSV: {args.train_csv}  ({len(train_cases)} cases)")

    # Build domain artifacts from TRAIN cases only 
    el.build_resource_activity_matrix(train_cases, min_observations=1)
    el.discover_case_invariant_features(train_cases)
    el.discover_activity_determined_categoricals(train_cases)   
    el.build_activity_to_category_maps(train_cases)      
    print(f"  Resource-activity matrix: {el.RESOURCE_ACTIVITY_MATRIX.shape}, "
          f"feasible pairs: {int(el.RESOURCE_ACTIVITY_MATRIX.sum())}")
    print(f"  Case-invariant features : {el.CASE_INVARIANT_FEATURES}")

    # ── 2. Phase 1 — load existing weights, or train fresh ─────────────────
    section("2 / Phase 1 — load or train")

    lmdp = Pipeline(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta_vae=args.beta_vae,
        gamma=args.gamma,
        cf_lambda1=args.cf_lambda1,
        cf_lambda2=args.cf_lambda2,
        cf_lambda3=args.cf_lambda3,
        cf_lambda_ldc=args.cf_lambda_ldc,
        cf_enforce_ldc=args.cf_enforce_ldc,
        cf_delta=args.cf_delta,
        cf_lr=args.cf_lr,
        cf_max_iter=args.cf_max_iter,
        surrogate_n_samples=args.surrogate_n,
        surrogate_rho=args.surrogate_rho,
        device=args.device,
    )

    weights_dir = os.path.abspath(args.weights_dir)
    required_files = ["black_box.cbm", "transition_system.json", "vae.pt"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(weights_dir, f))]
    weights_ok = len(missing_files) == 0

    if weights_ok and not args.retrain:
        domain_persistence.load_domain(weights_dir)
        lmdp.load(weights_dir)

        # REVISED+_pres baseline
        lmdp._fit_cases = train_cases

        pm_path = os.path.join(weights_dir, "pm_ground_truth.pt")
        if os.path.exists(pm_path):
            lmdp.vae.set_ground_truth_transitions(torch.load(pm_path), args.gamma_conf)
            print(f"  Loaded ground-truth transition matrix from {pm_path}")

        # REVISED+_pres baseline
        tdc_path = os.path.join(weights_dir, "declare_constraints.pt")
        if args.declare_lambda_tdc > 0.0:
            if os.path.exists(tdc_path):
                declare_constraints = torch.load(tdc_path, weights_only=False)
                lmdp.vae.set_declare_constraints(declare_constraints, args.declare_lambda_tdc)
                print(f"  Loaded {len(declare_constraints)} TDC constraints from {tdc_path}")
            else:
                print(f"  WARNING: --declare-lambda-tdc={args.declare_lambda_tdc} set but no "
                      f"{tdc_path} found — these weights were saved WITHOUT TDC active. "
                      f"Loaded VAE will behave as lambda_tdc=0.0 regardless of this flag. "
                      f"Use --retrain to train a TDC-active checkpoint.")
        
        lmdp.vae.set_resource_lambda(lmdp.vae.lambda_res_tensor.item())
        if args.lambda_res > 0.0 and abs(args.lambda_res - lmdp.vae.lambda_res_tensor.item()) > 1e-9:
            print(f"  WARNING: --lambda-res={args.lambda_res} was passed, but the loaded "
                  f"checkpoint was trained with lambda_res={lmdp.vae.lambda_res_tensor.item()} "
                  f"-- the CHECKPOINT'S value is what's actually active (weights were never "
                  f"retrained with your CLI value). Use --retrain to train a checkpoint at "
                  f"the CLI-specified lambda_res.")
        print(f"  Loaded existing weights from {weights_dir}")
    else:
        if missing_files:
            print(f"  Missing weights: {', '.join(missing_files)} — training fresh from --train-csv.")
        else:
            print(f"  --retrain set — training fresh from --train-csv.")

        t0 = time.time()
        PM_ground_truth = None
        if args.use_conformance:
            from baseline.revised_plus.data.event_log import build_activity_transition_matrix, ACTIVITIES, pm_row_entropy_floor
            PM_ground_truth = build_activity_transition_matrix(train_cases, n_activities=len(ACTIVITIES))
            floor = pm_row_entropy_floor(PM_ground_truth, train_cases, n_activities=len(ACTIVITIES))
            print(f"  DIFF-ERO theoretical floor (weighted avg row entropy of PM): {floor:.4f} nats")
            os.makedirs(weights_dir, exist_ok=True)
            pm_path = os.path.join(weights_dir, "pm_ground_truth.pt")
            torch.save(PM_ground_truth, pm_path)
            print(f"  Ground-truth transition matrix saved → {pm_path}")

        bb_precomputed_cv_summary = None
        if args.bb_cv_summary_path:
            if not os.path.exists(args.bb_cv_summary_path):
                raise FileNotFoundError(
                    f"--bb-cv-summary-path {args.bb_cv_summary_path} not found."
                )
            with open(args.bb_cv_summary_path) as f:
                bb_precomputed_cv_summary = json.load(f)
            print(f"  Loaded precomputed black-box CV summary from {args.bb_cv_summary_path}")

        reuse_black_box = None
        if args.reuse_black_box_from:
            from baseline.revised_plus.models.blackbox import CatBoostPPM
            bb_path = Path(args.reuse_black_box_from)
            reuse_black_box = CatBoostPPM.load(bb_path)
            print(f"  Reusing EXISTING black-box from {bb_path} (not retraining it) — "
                  f"oracle held fixed for this VAE-only retrain")

        lmdp.fit(
            train_cases,
            epochs_vae=args.epochs_vae,
            epochs_transition=args.epochs_transition,
            epochs_reward=args.epochs_reward,
            batch_size=args.batch_size,
            lr=args.lr,
            PM_ground_truth=PM_ground_truth,
            gamma_conf=args.gamma_conf,
            declare_lambda_tdc=args.declare_lambda_tdc,
            lambda_res=args.lambda_res,
            bb_run_cv=not args.skip_bb_cv,
            bb_cv_k=args.bb_cv_k,
            bb_precomputed_cv_summary=bb_precomputed_cv_summary,
            reuse_black_box=reuse_black_box,
        )
        print(f"  Training completed in {time.time() - t0:.1f}s")

        lmdp.save(weights_dir)
        domain_persistence.save_domain(weights_dir)

        if args.declare_lambda_tdc > 0.0 and lmdp.vae.declare_constraints is not None:
            tdc_path = os.path.join(weights_dir, "declare_constraints.pt")
            torch.save(lmdp.vae.declare_constraints, tdc_path)
            print(f"  Saved {len(lmdp.vae.declare_constraints)} TDC constraints → {tdc_path}")

        print(f"  Saved weights + domain snapshot to {weights_dir}")

    # ── 3. Load TEST cases and re-assert the training-time domain ──────────
    section("3 / Test-set loading")

    test_cases = load_event_log_csv(args.test_csv, sample_n_cases=None)
    print(f"  Loaded test CSV: {args.test_csv}  ({len(test_cases)} cases)")

    bucket_edges_file = args.bucket_edges_file or os.path.join(weights_dir, "bucket_edges.json")
    full_lens_for_bucketing = [len(c.events) for c in test_cases if len(c.events) >= 2]

    if os.path.exists(bucket_edges_file):
        bucket_edges, bucket_labels, loaded_method, loaded_n_buckets = load_bucket_edges(bucket_edges_file)
        print(f"  Loaded FIXED length-bucket edges from {bucket_edges_file} "
              f"(method={loaded_method}, n_buckets={loaded_n_buckets}) — "
              f"reusing across batches for coherent bucket labels.")
        if loaded_method != args.bucket_method or loaded_n_buckets != args.n_buckets:
            print(f"  WARNING: --bucket-method/--n-buckets ({args.bucket_method}/"
                  f"{args.n_buckets}) differ from the file's saved values "
                  f"({loaded_method}/{loaded_n_buckets}) — the FILE's edges "
                  f"win. Delete {bucket_edges_file} and re-run on the full "
                  f"test set first if you actually want to change bucketing.")
    else:
        bucket_edges, bucket_labels = compute_length_buckets(
            full_lens_for_bucketing, n_buckets=args.n_buckets, method=args.bucket_method
        )
        save_bucket_edges(bucket_edges_file, bucket_edges, bucket_labels,
                           args.bucket_method, args.n_buckets)
        print(f"  Computed length-bucket edges from the FULL test set "
              f"({len(full_lens_for_bucketing)} cases) and saved to "
              f"{bucket_edges_file} for reuse by subsequent batches.")

    if args.skip_processed_dir:
        import glob as _glob
        already_done_ids = set()
        n_parse_errors = 0
        for skip_dir in args.skip_processed_dir:
            for path in _glob.glob(os.path.join(skip_dir, "*.json")):
                try:
                    with open(path) as f:
                        payload = json.load(f)
                    cid = payload.get("case_id")
                    if cid is not None:
                        already_done_ids.add(str(cid))
                except Exception:
                    n_parse_errors += 1
        n_before = len(test_cases)
        test_cases = [c for c in test_cases if str(c.case_id) not in already_done_ids]
        print(f"  --skip-processed-dir {args.skip_processed_dir} set — found "
              f"{len(already_done_ids)} already-processed case_id(s) "
              f"({n_parse_errors} unreadable file(s) ignored). "
              f"Test set: {n_before} -> {len(test_cases)} case(s) remaining "
              f"(matched by actual case_id, not position — safe regardless "
              f"of ordering).")

    if args.case_offset:
        n_before = len(test_cases)
        test_cases = test_cases[args.case_offset:]
        print(f"  --case-offset {args.case_offset} set — skipped the first "
              f"{args.case_offset} case(s) ({n_before} -> {len(test_cases)} "
              f"remaining), so this run pulls a different slice of the test "
              f"set than a prior --case-offset 0 run. NOTE: this assumes "
              f"identical case ordering to whatever prior run you're trying "
              f"not to overlap with — if unsure, prefer --skip-processed-dir.")

    if args.limit_cases is not None:
        n_before = len(test_cases)
        test_cases = test_cases[:args.limit_cases]
        print(f"  --limit-cases {args.limit_cases} set — truncated test set "
              f"from {n_before} to {len(test_cases)} case(s) for a quick "
              f"validation run.")

    event_dim_after_test_load = el.EVENT_DIM
    domain_persistence.load_domain(weights_dir)
    event_dim_after_restore = el.EVENT_DIM
    if event_dim_after_test_load != event_dim_after_restore:
        print(f"  Domain restore CHANGED EVENT_DIM: {event_dim_after_test_load} "
              f"(from test-only schema) → {event_dim_after_restore} (training-time snapshot). "
              f"This confirms the restore was necessary and took effect.")
    else:
        print(f"  Domain restore left EVENT_DIM unchanged ({event_dim_after_restore}) — "
              f"train/test schemas happened to match, or restore was a no-op either way.")

    lens = [len(c.events) for c in test_cases]
    print(f"  Test trace length: min={min(lens)}, mean={np.mean(lens):.1f}, max={max(lens)}")

    too_short_hard = [c for c in test_cases if len(c.events) < 2]
    if too_short_hard:
        print(f"  WARNING: {len(too_short_hard)} test case(s) have < 2 events and will be "
              f"skipped entirely (cannot compute even a single slice for them).")

    print(f"  Length bucket edges (fixed) : {bucket_edges.tolist()}")
    for lbl in bucket_labels:
        n_in_bucket = sum(
            1 for c in test_cases
            if len(c.events) >= 2
            and assign_length_bucket(len(c.events), bucket_edges, bucket_labels) == lbl
        )
        print(f"    {lbl:<28} n_cases={n_in_bucket}")

    n_excluded_short = sum(
        1 for c in test_cases if 2 <= len(c.events) < args.min_case_length
    )
    if n_excluded_short:
        print(f"  {n_excluded_short} case(s) have 2 <= n < --min-case-length "
              f"({args.min_case_length}) and will get a single fixed-prefix pass "
              f"instead of the milestone sweep — reported under bucket 'excluded_short'.")

    section("3b / Pre-flight: cases per (length_bucket, milestone) cell")

    preview_cell_counts = {}   # (bucket_label, milestone) -> n_cases
    preview_short_count = 0
    for c in test_cases:
        n = len(c.events)
        if n < 2:
            continue
        if n < args.min_case_length:
            preview_short_count += 1
            continue
        b = assign_length_bucket(n, bucket_edges, bucket_labels)
        for milestone, _step in compute_milestone_slices(n, args.milestones, min_case_length=args.min_case_length):
            key = (b, milestone)
            preview_cell_counts[key] = preview_cell_counts.get(key, 0) + 1

    LOW_COUNT_THRESHOLD = 5  
    print(f"  {'bucket':<28} {'milestone':>10} {'n_cases':>9}")
    low_count_cells = []
    for bucket_label in bucket_labels:
        for milestone in args.milestones:
            n_cases = preview_cell_counts.get((bucket_label, milestone), 0)
            flag = "  <-- LOW" if n_cases < LOW_COUNT_THRESHOLD else ""
            print(f"  {bucket_label:<28} {milestone:>10} {n_cases:>9}{flag}")
            if n_cases < LOW_COUNT_THRESHOLD:
                low_count_cells.append((bucket_label, milestone, n_cases))
    if preview_short_count:
        print(f"  {'excluded_short':<28} {'(fixed pass)':>10} {preview_short_count:>9}")

    if low_count_cells:
        print(f"\n  WARNING: {len(low_count_cells)} cell(s) have fewer than "
              f"{LOW_COUNT_THRESHOLD} cases — their validity/proximity/plausibility "
              f"means in bucket_milestone_summary.csv will be noisy. Consider "
              f"reducing --n-buckets (fewer, larger buckets) or, if this cell is "
              f"consistently low across bucket counts, accepting it as a genuine "
              f"sparse region of your test set's length distribution and reporting "
              f"it with that caveat rather than treating the number as reliable.")

    if args.dry_run:
        print("\n  --dry-run set: stopping before the explain() loop. "
              "Re-run without --dry-run once the cell counts above look right.")
        return

    # ── 4. Phase 2 — run CF explanation at every (length-bucket, milestone) ─
    #      cut point, plus a separate fixed-prefix pass for too-short cases.
    section("4 / Phase 2 — length-bucket x milestone CF explanations over test set")

    explanations_dir = os.path.join(out_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    combined_summary = []   # one row per generated CF, across all test cases
    slice_level_summary = []  # one row per (case, slice) — surrogate precision
                               # is computed once per slice (not per CF), so it
                               # gets its own table to avoid either dropping it
                               # or duplicating/misweighting it across cf_k CFs
    n_cases_run = 0
    n_cases_failed = 0
    n_cf_total = 0
    n_slices_total = 0

    def run_one_slice(case, step, n, milestone_label, bucket_label, tag):
        """
        Runs explain() at one (case, step) cut point and appends results to
        combined_summary / writes the per-case JSON. milestone_label is the
        requested fraction (float) or None for the short-case fixed pass.
        tag is used in the output filename to keep milestone- and
        short-case-pass outputs from colliding.
        Returns True on success, False on failure (already logged).
        """
        nonlocal n_cases_run, n_cases_failed, n_cf_total

        K = args.editable_history_steps
        editable_steps = list(range(max(1, step - K), step)) if K > 0 else None
 
        try:
            explanation = lmdp.explain(
                case=case,
                step=step,
                surrogate_epochs=args.surrogate_epochs,
                verbose=False,
                k=args.cf_k,
                contrast_mode=args.contrast_mode,
                editable_steps=editable_steps,
                debug_attempts_dir=args.debug_attempts_dir,
                kpi_significance_threshold=args.kpi_significance_threshold,
                kpi_min_baseline=args.kpi_min_baseline
            )
        except Exception as e:
            logger.warning(f"  case={case.case_id} (n={n}, step={step}, {tag}) failed to explain: {e}")
            n_cases_failed += 1
            return False

        n_cases_run += 1
        n_found = explanation.get("n_found", len(explanation.get("multiple_explanations", [])))
        n_cf_total += n_found

        surr_prec = explanation.get("surrogate_precision", None)
        no_significant_kpi_cf = explanation.get("no_significant_kpi_cf", None)
        best_kpi_delta_pct = explanation.get("best_kpi_delta_pct", None)
        slice_level_summary.append({
            "case_id":                str(case.case_id),
            "trace_len":              n,
            "length_bucket":          bucket_label,
            "milestone":              milestone_label,
            "slice_idx":              step,
            "surrogate_accuracy":            surr_prec.get("accuracy") if surr_prec else None,
            "surrogate_precision_macro":     surr_prec.get("precision_macro") if surr_prec else None,
            "surrogate_precision_weighted":  surr_prec.get("precision_weighted") if surr_prec else None,
            "surrogate_n_holdout":           surr_prec.get("n_holdout") if surr_prec else None,
            "no_significant_kpi_cf":         no_significant_kpi_cf,
            "best_kpi_delta_pct":            best_kpi_delta_pct,
            "history_edit_multi_step":       explanation.get("history_edit_multi_step", False),
            "editable_steps_used":           explanation.get("editable_steps_used", [step]),
        })

        out_records = []
        for expl in explanation["multiple_explanations"]:
            cf_r = expl["cf_result"]

            # Manifold feasibility
            l_man, plaus_score = plausibility_score(cf_r.z_cf)

            record = {
                "target_rank":             expl["target_rank"],
                "contrast_type":           expl.get("contrast_type", ""),
                "contrast_mode_requested": expl.get("contrast_mode_requested", args.contrast_mode),
                "summary_text":            expl["summary_text"],
                "resource_valid":          expl.get("resource_valid", True),
                "resource_violations":     expl.get("resource_violations", []),
                "cf_diagnostics": {
                    "n_iterations":          cf_r.n_iterations,
                    "exit_reason":           cf_r.exit_reason,
                    "used_expulsion":        cf_r.used_expulsion,
                    "verified_by_blackbox":  cf_r.verified_by_blackbox,
                    "proximity_latent_l2sq": cf_r.proximity,
                    "sparsity":              cf_r.sparsity,
                    "original_action_idx":   cf_r.original_action_idx,
                    "cf_action_idx":         cf_r.cf_action_idx,
                    "l_man":                 l_man,
                    "plausibility_score":    plaus_score,
                },
                "kpi_original":            expl.get("kpi_original"),
                "kpi_counterfactual":      expl.get("kpi_counterfactual"),
                "kpi_delta_pct":          expl.get("kpi_delta_pct"),
                "kpi_significant":        expl.get("kpi_significant"),
                "kpi_rank":               expl.get("kpi_rank"),
                "changed_features": expl.get("changed_features", []),
                "changed_events": expl.get("changed_events", []),
                "history_edit_multi_step":       expl.get("history_edit_multi_step", False),
                "editable_steps_used":           expl.get("editable_steps_used", [step]),
            }
            out_records.append(record)

            combined_summary.append({
                "case_id":                 str(case.case_id),
                "trace_len":               n,
                "length_bucket":           bucket_label,
                "milestone":               milestone_label,
                "prefix_fraction_actual":  step / n,  
                "slice_idx":               step,
                "target_rank":             expl["target_rank"],
                "contrast_type":           expl.get("contrast_type", ""),
                "contrast_mode_requested": expl.get("contrast_mode_requested", args.contrast_mode),
                "verified_by_blackbox":    cf_r.verified_by_blackbox,
                "exit_reason":             cf_r.exit_reason,
                "used_expulsion":          cf_r.used_expulsion,
                # REVISED+_pres
                "ldc_final_violation":     cf_r.ldc_final_violation,
                "ldc_unavailable":         cf_r.ldc_unavailable,
                "n_iterations":            cf_r.n_iterations,
                "proximity_latent_l2":     float(np.sqrt(cf_r.proximity)),
                "sparsity":                cf_r.sparsity,
                "resource_valid":          expl.get("resource_valid", True),
                "l_man":                   l_man,
                "plausibility_score":      plaus_score, # Manifold plausibility score 
                "surrogate_accuracy":         surr_prec.get("accuracy") if surr_prec else None,
                "surrogate_precision_macro":  surr_prec.get("precision_macro") if surr_prec else None,
                "kpi_original":            expl.get("kpi_original"),
                "kpi_counterfactual":      expl.get("kpi_counterfactual"),
                "kpi_delta_pct":           expl.get("kpi_delta_pct"),
                "kpi_significant":         expl.get("kpi_significant"),
                "kpi_rank":                expl.get("kpi_rank"),
                "history_edit_multi_step":       explanation.get("history_edit_multi_step", False),
                "editable_steps_used":           explanation.get("editable_steps_used", [step]),
            })

        out_payload = {
            "case_id":                 str(case.case_id),
            "trace_len":               n,
            "length_bucket":           bucket_label,
            "milestone":               milestone_label,
            "slice_idx":               step,
            "explains_step":           step + 1,
            "contrast_mode_requested": args.contrast_mode,
            "cf_k_requested":          args.cf_k,
            "n_found":                 n_found,
            "n_requested":             explanation.get("n_requested", args.cf_k),
            "n_attempts":              explanation.get("n_attempts", None),
            "surrogate_precision":     surr_prec,
            "no_significant_kpi_cf":   no_significant_kpi_cf,
            "best_kpi_delta_pct":      best_kpi_delta_pct,
            "kpi_significance_threshold": args.kpi_significance_threshold,
            "counterfactuals":         out_records,
        }
        out_filename = f"case_{case.case_id}_step_{step}_{tag}_{args.contrast_mode}.json"
        with open(os.path.join(explanations_dir, out_filename), "w") as f:
            json.dump(out_payload, f, indent=2, default=str)
        return True

    short_case_ids = []

    for i, case in enumerate(test_cases):
        n = len(case.events)
        if n < 2:
            continue

        if n < args.min_case_length:
            short_case_ids.append(str(case.case_id))
            step = max(1, n - 1)
            run_one_slice(case, step, n, milestone_label=None,
                          bucket_label="excluded_short", tag="shortpass")
            n_slices_total += 1
        else:
            bucket_label = assign_length_bucket(n, bucket_edges, bucket_labels)
            slices = compute_milestone_slices(n, args.milestones, min_case_length=args.min_case_length)
            for milestone, step in slices:
                milestone_tag = f"m{milestone*100:.0f}pct"
                run_one_slice(case, step, n, milestone_label=milestone,
                              bucket_label=bucket_label, tag=milestone_tag)
                n_slices_total += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_cases)} test cases processed "
                  f"({n_slices_total} slices attempted, {n_cases_run} succeeded, "
                  f"{n_cases_failed} failed, {n_cf_total} CFs total so far)")

    # ── 5. Save combined summary CSV, surrogate-precision-per-slice CSV, ────
    #      bucket x milestone aggregate, and manifest
    section("5 / Saving combined summary")

    import csv

    slice_summary_csv_path = os.path.join(out_dir, "surrogate_precision_summary.csv")
    if slice_level_summary:
        with open(slice_summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(slice_level_summary[0].keys()))
            writer.writeheader()
            writer.writerows(slice_level_summary)
        print(f"  Saved surrogate precision per slice ({len(slice_level_summary)} rows) → {slice_summary_csv_path}")
    else:
        slice_summary_csv_path = None
        print("  No slices were run — no surrogate precision summary written.")

    summary_csv_path = os.path.join(out_dir, "combined_summary.csv")
    if combined_summary:
        fieldnames = list(combined_summary[0].keys())
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_summary)
        print(f"  Saved combined summary ({len(combined_summary)} CF rows) → {summary_csv_path}")
    else:
        print("  No CFs were generated across the entire test set — no summary CSV written.")

    agg_csv_path = os.path.join(out_dir, "bucket_milestone_summary.csv")
    if combined_summary:
        by_key = {}
        for row in combined_summary:
            key = (row["length_bucket"], row["milestone"])
            by_key.setdefault(key, []).append(row)

        slice_by_key = {}
        for row in slice_level_summary:
            key = (row["length_bucket"], row["milestone"])
            slice_by_key.setdefault(key, []).append(row)

        agg_rows = []
        for (bucket_label, milestone), rows in sorted(
            by_key.items(), key=lambda kv: (str(kv[0][0]), -1 if kv[0][1] is None else kv[0][1])
        ):
            case_ids = {r["case_id"] for r in rows}
            n_rows = len(rows)

            slice_rows = slice_by_key.get((bucket_label, milestone), [])
            surr_acc_vals  = [r["surrogate_accuracy"] for r in slice_rows if r["surrogate_accuracy"] is not None]
            surr_prec_vals = [r["surrogate_precision_macro"] for r in slice_rows if r["surrogate_precision_macro"] is not None]

            kpi_delta_vals  = [r["kpi_delta_pct"] for r in rows if r["kpi_delta_pct"] is not None]
            kpi_sig_vals    = [bool(r["kpi_significant"]) for r in rows if r["kpi_significant"] is not None]
            no_sig_kpi_vals = [bool(r["no_significant_kpi_cf"]) for r in slice_rows if r["no_significant_kpi_cf"] is not None]
            
            agg_rows.append({
                "length_bucket":          bucket_label,
                "milestone":              milestone,
                "n_cases":                len(case_ids),
                "n_cf_rows":              n_rows,
                "n_slices":               len(slice_rows),
                "mean_slice_idx":         float(np.mean([r["slice_idx"] for r in rows])),
                "mean_prefix_fraction":   float(np.mean([r["prefix_fraction_actual"] for r in rows])),
                "validity_rate":          float(np.mean([bool(r["verified_by_blackbox"]) for r in rows])),
                "used_expulsion_rate":    float(np.mean([bool(r["used_expulsion"]) for r in rows])),
                "resource_valid_rate":    float(np.mean([bool(r["resource_valid"]) for r in rows])),
                "mean_proximity_l2":      float(np.mean([r["proximity_latent_l2"] for r in rows])),
                "mean_sparsity":          float(np.mean([r["sparsity"] for r in rows])),
                "mean_n_iterations":      float(np.mean([r["n_iterations"] for r in rows])),
                # REVISED+_pres baseline
                "mean_ldc_final_violation": float(np.mean([r["ldc_final_violation"] for r in rows])),
                "ldc_unavailable_rate":      float(np.mean([bool(r["ldc_unavailable"]) for r in rows])),
                "mean_plausibility_score": float(np.mean([r["plausibility_score"] for r in rows])), # Manifold plausibility score 
                "mean_l_man":              float(np.mean([r["l_man"] for r in rows])),
                "mean_surrogate_accuracy":        float(np.mean(surr_acc_vals)) if surr_acc_vals else None,
                "mean_surrogate_precision_macro":  float(np.mean(surr_prec_vals)) if surr_prec_vals else None,
                "mean_kpi_delta_pct":        float(np.mean(kpi_delta_vals)) if kpi_delta_vals else None,
                "kpi_significant_rate":      float(np.mean(kpi_sig_vals)) if kpi_sig_vals else None,
                "no_significant_kpi_cf_rate": float(np.mean(no_sig_kpi_vals)) if no_sig_kpi_vals else None,
            })

        with open(agg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"  Saved bucket x milestone aggregate ({len(agg_rows)} cells) → {agg_csv_path}")

        for row in agg_rows:
            if row["used_expulsion_rate"] > 0.8:
                print(f"  WARNING: {row['length_bucket']} / milestone={row['milestone']} "
                      f"has used_expulsion_rate={row['used_expulsion_rate']:.2f} (>0.8) — "
                      f"consider reporting this cell's CFs separately from clean-convergence "
                      f"CFs, and/or check --cf-lr.")
    else:
        agg_csv_path = None
        print("  No CFs were generated — no bucket x milestone aggregate written.")

    manifest = {
        "train_csv":            args.train_csv,
        "test_csv":             args.test_csv,
        "milestones":           args.milestones,
        "n_buckets":            args.n_buckets,
        "bucket_method":        args.bucket_method,
        "length_bucket_edges":  bucket_edges.tolist(),
        "length_bucket_labels": bucket_labels,
        "min_case_length":      args.min_case_length,
        "n_short_cases_excluded_from_milestones": len(short_case_ids),
        "short_case_ids":       short_case_ids,
        "contrast_mode":        args.contrast_mode,
        "cf_k":                 args.cf_k,
        "gamma_conf":           args.gamma_conf,
        "declare_lambda_tdc":   args.declare_lambda_tdc,
        "lambda_res":           args.lambda_res,
        "cf_lambda_ldc":        args.cf_lambda_ldc,
        "kpi_significance_threshold": args.kpi_significance_threshold,
        "kpi_min_baseline":     args.kpi_min_baseline,
        "cf_enforce_ldc":       args.cf_enforce_ldc,
        "skip_bb_cv":           args.skip_bb_cv,
        "bb_cv_k":              args.bb_cv_k,
        "bb_cv_summary_path":   args.bb_cv_summary_path,
        "weights_dir":          weights_dir,
        "n_test_cases":         len(test_cases),
        "n_slices_attempted":   n_slices_total,
        "n_cases_run":          n_cases_run,
        "n_cases_failed":       n_cases_failed,
        "n_cf_total":           n_cf_total,
        "explanations_dir":     explanations_dir,
        "combined_summary_csv": summary_csv_path if combined_summary else None,
        "surrogate_precision_summary_csv": slice_summary_csv_path,
        "bucket_milestone_summary_csv": agg_csv_path,
    }
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    banner("DONE")
    print(f"  Test cases in set      : {len(test_cases)}")
    print(f"  Slices attempted       : {n_slices_total}  ({n_cases_run} succeeded, {n_cases_failed} failed)")
    print(f"  Short cases (< {args.min_case_length} events) : {len(short_case_ids)} (bucket='excluded_short')")
    print(f"  Total CFs generated    : {n_cf_total}")
    print(f"  Per-case JSON          : {explanations_dir}/")
    print(f"  Combined summary CSV   : {summary_csv_path if combined_summary else '(none)'}")
    print(f"  Surrogate precision CSV: {slice_summary_csv_path if slice_summary_csv_path else '(none)'}")
    print(f"  Bucket x milestone CSV : {agg_csv_path if agg_csv_path else '(none)'}")
    print(f"  Run manifest           : {manifest_path}")


if __name__ == "__main__":
    main()
