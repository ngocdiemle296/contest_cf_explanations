"""
Training for Offline Phase.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

from baseline.revised_plus.data import event_log
from baseline.revised_plus.models.blackbox import CatBoostPPM, TransitionSystem
from baseline.revised_plus.models.vae import LSTMVAE

logger = logging.getLogger(__name__)

def pad_sequence(seqs: List[np.ndarray], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [s.shape[0] for s in seqs]
    if max_len is None:
        max_len = max(lengths)
    B, D = len(seqs), seqs[0].shape[1]
    padded = np.zeros((B, max_len, D), dtype=np.float32)
    mask   = np.zeros((B, max_len), dtype=np.float32)
    for i, (s, l) in enumerate(zip(seqs, lengths)):
        padded[i, :l] = s
        mask[i, :l]   = 1.0
    return torch.tensor(padded), torch.tensor(lengths, dtype=torch.long), torch.tensor(mask)


class PPMTimeDataset(Dataset):
    """
    Builds (prefix, next-step target) samples for CatBoost black-box
    training, one sample per (case, prefix_length t) pair.
    """
    def __init__(self, cases: List[event_log.Case], min_prefix: int = 1):
        self.case_matrices: List[np.ndarray] = []
        self.index: List[Tuple[int, int, int, int, float]] = []

        for case in cases:
            events = case.events
            if len(events) <= min_prefix:
                continue   # no valid (prefix, target) pairs for this case

            full_mat = case.to_matrix().astype(np.float32)   
            self.case_matrices.append(full_mat)
            case_idx = len(self.case_matrices) - 1

            for t in range(min_prefix, len(events)):
                next_ev = events[t]
                act_idx = event_log.ACT2IDX.get(next_ev.activity, 0)
                res_idx = event_log.RES2IDX.get(next_ev.resource, 0)
                target_val = (
                    float(next_ev.remaining_time)
                    if next_ev.remaining_time is not None and not np.isnan(next_ev.remaining_time)
                    else 0.0
                )
                self.index.append((case_idx, t, act_idx, res_idx, target_val))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        case_idx, t, act_idx, res_idx, target_val = self.index[idx]
        trace_mat = self.case_matrices[case_idx][:t]   # view, not a copy
        return trace_mat, t, act_idx, res_idx, target_val


class VAEDataset(Dataset):

    def __init__(self, cases: List[event_log.Case], min_len: int = 2):
        self.case_matrices: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []   # (case_matrix_idx, t)

        for case in cases:
            if len(case.events) < min_len:
                continue

            full_mat = case.to_matrix().astype(np.float32)
            self.case_matrices.append(full_mat)
            case_idx = len(self.case_matrices) - 1

            for t in range(min_len, len(case.events) + 1):
                self.index.append((case_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        case_idx, t = self.index[idx]
        return self.case_matrices[case_idx][:t]   

def vae_collate(batch):
    return pad_sequence(batch)


class LengthBucketBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset: "VAEDataset", batch_size: int, bucket_width: int = 5, seed: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

        lengths = [dataset.case_matrices[c].shape[0] if t is None else t
                   for c, t in dataset.index]  
        lengths = [t for _, t in dataset.index]

        buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(lengths):
            key = length // bucket_width
            buckets.setdefault(key, []).append(idx)
        self.buckets = list(buckets.values())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        g = np.random.RandomState(self.seed + self._epoch)
        batches: List[List[int]] = []
        for bucket in self.buckets:
            b = bucket.copy()
            g.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batches.append(b[i:i + self.batch_size])
        order = np.arange(len(batches))
        g.shuffle(order)
        for i in order:
            yield batches[i]

    def __len__(self) -> int:
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets)


def cross_validate_black_box(cases: List[event_log.Case], k: int = 5,
                              depth: int = 6, learning_rate: float = 0.1,
                              iterations: int = 300, variance_power: float = 1.5,
                              catboost_thread_count: Optional[int] = 4,
                              seed: int = 42) -> Dict[str, float]:
    """
    Case-level k-fold CV for the black-box time predictor.
    """
    max_trace_len = max((len(c.events) for c in cases), default=1)
    ts = TransitionSystem.from_cases(cases)
    model_builder = CatBoostPPM(transition_system=ts, max_trace_len=max_trace_len,
                                 thread_count=catboost_thread_count,
                                 depth=depth, learning_rate=learning_rate,
                                 iterations=iterations, variance_power=variance_power)

    def _build_features_targets(case_subset):
        sub_dataset = PPMTimeDataset(case_subset)
        feats, targs = [], []
        for trace_mat, length, act_idx, res_idx, target_val in sub_dataset:
            feats.append(model_builder._build_feature_vector(trace_mat, length, act_idx, res_idx))
            targs.append(target_val)
        return np.stack(feats), np.asarray(targs, dtype=np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cases))
    folds = np.array_split(perm, k)

    fold_rmses, fold_maes, fold_negrates = [], [], []
    for fold_i in range(k):
        val_case_idx = set(folds[fold_i].tolist())
        train_c = [c for i, c in enumerate(cases) if i not in val_case_idx]
        val_c = [c for i, c in enumerate(cases) if i in val_case_idx]
        if not val_c or not train_c:
            continue

        X_tr, y_tr = _build_features_targets(train_c)
        X_va, y_va = _build_features_targets(val_c)

        fold_model = CatBoostPPM(transition_system=ts, max_trace_len=max_trace_len,
                                  thread_count=catboost_thread_count,
                                  depth=depth, learning_rate=learning_rate,
                                  iterations=iterations, variance_power=variance_power)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict_time(X_va)
        resid = preds - y_va
        rmse = float(np.sqrt((resid ** 2).mean()))
        mae = float(np.abs(resid).mean())
        neg_rate = 100.0 * float((preds < 0).mean())
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_negrates.append(neg_rate)
        logger.info(f"  fold {fold_i+1}/{k}: RMSE={rmse:.2f}  MAE={mae:.2f}  neg%={neg_rate:.2f}")

    summary = dict(
        rmse_mean=float(np.mean(fold_rmses)), rmse_std=float(np.std(fold_rmses)),
        mae_mean=float(np.mean(fold_maes)), mae_std=float(np.std(fold_maes)),
        neg_rate_mean=float(np.mean(fold_negrates)), neg_rate_std=float(np.std(fold_negrates)),
        k=k,
    )
    logger.info(
        f"BlackBox {k}-fold CV summary: "
        f"RMSE={summary['rmse_mean']:.2f}+/-{summary['rmse_std']:.2f}  "
        f"MAE={summary['mae_mean']:.2f}+/-{summary['mae_std']:.2f}  "
        f"neg%={summary['neg_rate_mean']:.2f}+/-{summary['neg_rate_std']:.2f}"
    )
    return summary


def train_black_box(cases: List[event_log.Case], device: torch.device,
                     batch_size: int = 64, lr: float = 1e-3,
                     catboost_thread_count: Optional[int] = 4,
                     run_cv: bool = True, cv_k: int = 5,
                     depth: int = 6, learning_rate: float = 0.1,
                     iterations: int = 300, variance_power: float = 1.5,
                     precomputed_cv_summary: Optional[Dict] = None) -> CatBoostPPM:

    logger.info("=== Training BlackBox PPM (CatBoost) ===")
    
    if event_log.EVENT_DIM == 0 or len(event_log.ACTIVITIES) == 0:
        all_acts = sorted(list(set(ev.activity for c in cases for ev in c.events)))
        all_ress = sorted(list(set(ev.resource for c in cases for ev in c.events)))
        event_log.set_domain(all_acts, all_ress)

    if not run_cv and precomputed_cv_summary is None:
        logger.warning(
            "run_cv=False but no precomputed_cv_summary was given — falling "
            "back to running CV rather than leaving model.cv_summary=None. "
            "Pass precomputed_cv_summary explicitly to actually skip computation."
        )
        run_cv = True

    cv_summary = None
    if run_cv:
        logger.info(f"Running {cv_k}-fold CV for honest generalization estimate before final fit...")
        cv_summary = cross_validate_black_box(
            cases, k=cv_k, depth=depth, learning_rate=learning_rate,
            iterations=iterations, variance_power=variance_power,
            catboost_thread_count=catboost_thread_count,
        )
    elif precomputed_cv_summary is not None:
        cv_summary = precomputed_cv_summary
        logger.info(
            f"Skipping CV (run_cv=False) — using precomputed_cv_summary instead: "
            f"RMSE={cv_summary.get('rmse_mean'):.2f}+/-{cv_summary.get('rmse_std'):.2f}  "
            f"MAE={cv_summary.get('mae_mean'):.2f}+/-{cv_summary.get('mae_std'):.2f}  "
            f"neg%={cv_summary.get('neg_rate_mean'):.2f}+/-{cv_summary.get('neg_rate_std'):.2f}  "
            f"(NOT recomputed on this run — verify data/hyperparams haven't changed)"
        )

    dataset = PPMTimeDataset(cases)
    max_trace_len = max((len(c.events) for c in cases), default=1)
    ts = TransitionSystem.from_cases(cases)
    model = CatBoostPPM(transition_system=ts, max_trace_len=max_trace_len,
                         thread_count=catboost_thread_count,
                         depth=depth, learning_rate=learning_rate,
                         iterations=iterations, variance_power=variance_power)

    feature_list, target_list = [], []
    for trace_mat, length, act_idx, res_idx, target_val in dataset:
        feature_list.append(model._build_feature_vector(trace_mat, length, act_idx, res_idx))
        target_list.append(target_val)

    features = np.stack(feature_list)
    targets = np.asarray(target_list, dtype=np.float32)
    
    logger.info(f"Dynamically configuring CatBoost for true feature space size: {features.shape[1]}")
    model.fit(features, targets)

    model.cv_summary = cv_summary
    return model


def train_vae(cases, device, latent_dim=32, hidden_dim=128, beta=1.0,
              n_epochs=20, batch_size=64, lr=1e-3,
              PM_ground_truth: Optional[torch.Tensor] = None, 
              gamma_conf: float = 0.0, # REVISED+_pres implementation so forcing DIFF-ERO loss to be 0
              declare_constraints: Optional[list] = None, 
              declare_lambda_tdc: float = 0.0,
              lambda_res: float = 0.0,
              use_scheduled_sampling=True, 
              use_length_bucketing: bool = False) -> LSTMVAE:
    logger.info("=== Training LSTM-VAE ===")
    dataset = VAEDataset(cases)
    
    use_workers = device.type == "cuda"
    if use_length_bucketing:
       
        batch_sampler = LengthBucketBatchSampler(dataset, batch_size=batch_size)
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=vae_collate,
            num_workers=4 if use_workers else 0,
            pin_memory=use_workers,
            persistent_workers=use_workers,
            prefetch_factor=4 if use_workers else None,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=vae_collate,
            num_workers=4 if use_workers else 0,
            pin_memory=use_workers,
            persistent_workers=use_workers,
            prefetch_factor=4 if use_workers else None,
            drop_last=False,
        )
        batch_sampler = None

    true_event_dim = cases[0].to_matrix(len(cases[0].events)).shape[1]
    model = LSTMVAE(event_dim=true_event_dim, hidden_dim=hidden_dim,
                     latent_dim=latent_dim, gamma_conf=gamma_conf,
                     lambda_res=lambda_res).to(device)
    
    if PM_ground_truth is not None:
        model.set_ground_truth_transitions(PM_ground_truth.to(device), gamma_conf)
        logger.info(f"  DIFF-ERO structural conformance active (γ={gamma_conf})")
    else:
        logger.info("  No conformance regularisation")

    if declare_lambda_tdc > 0.0:
        model.set_declare_constraints(declare_constraints, declare_lambda_tdc)
        logger.info(
            f"  REVISED+ TDC baseline active (λ_tdc={declare_lambda_tdc}, "
            f"{len(declare_constraints or [])} constraints)"
        )
    else:
        logger.info("  No TDC regularisation")

    if lambda_res > 0.0:
        logger.info(f"  REVISED+ resource-feasibility loss active (λ_res={lambda_res})")
    else:
        logger.info("  No resource-feasibility regularisation")
        
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)  
        model.train()

        total_loss  = torch.zeros((), device=device)
        total_recon = torch.zeros((), device=device)
        total_kl    = torch.zeros((), device=device)
        total_ero   = torch.zeros((), device=device)
        total_tdc   = torch.zeros((), device=device)
        total_res   = torch.zeros((), device=device)
        n = 0
        tf_prob = 1.0
        for x, lengths, mask in loader:
            x      = x.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)
            loss, recon, kl, conf_loss, tdc_loss, indep_res_loss, tf_prob = model.elbo_loss(x, lengths, mask, epoch=epoch, total_epochs=n_epochs, use_scheduled_sampling=use_scheduled_sampling)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            bs = x.size(0)
            total_loss  += loss.detach()  * bs
            total_recon += recon.detach() * bs
            total_kl    += kl.detach()    * bs
            total_ero   += conf_loss.detach() * bs
            total_tdc   += tdc_loss.detach() * bs
            total_res   += indep_res_loss.detach() * bs
            n += bs
        if epoch % 10 == 0:
            logger.info(f"  [sched-sampling] epoch={epoch} teacher_forcing_prob≈{tf_prob:.3f}")
        sched.step()
        logger.info(f"  Epoch {epoch+1:02d}/{n_epochs} | Loss={total_loss.item()/n:.4f} | Recon={total_recon.item()/n:.4f} | ERO={total_ero.item()/n:.4f} | TDC={total_tdc.item()/n:.4f} | Res={total_res.item()/n:.4f}")
    return model