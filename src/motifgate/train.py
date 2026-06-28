from __future__ import annotations

"""Training loop, evaluation, calibration and FDR-controlled operating points."""

import argparse
import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .constants import (LABEL_SMOOTHING, RC_AUGMENTATION)
from .data import (LabeledExample, TFBSDataset, compute_feature_stats, make_feature_matrix)
from .model import (make_cosine_warmup_scheduler, reverse_complement_onehot_batch, seq_pwm_match_loss)
from .pwm_io import (PWMIndex, compute_raw_prior_scores)
from .utils import (count_trainable_parameters, safe_auc_ap, safe_sigmoid_np)

def bootstrap_delta_ap(y: np.ndarray, prob_model: np.ndarray, prob_prior: np.ndarray,
                       n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap confidence interval for Δ(AP) = AP_model − AP_prior."""
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(seed)
    n = len(y)
    if n < 10:
        return {"delta_ap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}

    ap_model = average_precision_score(y, prob_model)
    ap_prior = average_precision_score(y, prob_prior)
    observed_delta = ap_model - ap_prior

    deltas = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            deltas[b] = 0.0
            continue
        ap_m = average_precision_score(yb, prob_model[idx])
        ap_p = average_precision_score(yb, prob_prior[idx])
        deltas[b] = ap_m - ap_p

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0.0))  # one-sided: P(Δ ≤ 0)

    return {
        "delta_ap": float(observed_delta),
        "ap_model": float(ap_model),
        "ap_prior": float(ap_prior),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    ys = []
    probs = []
    logits = []
    mids = []
    seqs = []
    sources = []
    tf_families = []
    tf_classes = []
    prior_zs = []
    baseline_components = []
    baseline_raws = []
    residual_components = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            seq_pad = batch["seq_pad"].to(device)
            feat = batch["feat"].to(device)
            pwm = batch["pwm"].to(device)
            pwm_pad = batch["pwm_pad"].to(device)
            out = model(X, seq_pad, feat, pwm, pwm_pad)
            ys.append(batch["y"].cpu().numpy().reshape(-1))
            probs.append(out["prob"].cpu().numpy().reshape(-1))
            logits.append(out["logit"].cpu().numpy().reshape(-1))
            prior_zs.append(batch["prior_z"].cpu().numpy().reshape(-1))
            bcomp = out.get("baseline_component", None)
            if bcomp is None:
                baseline_components.append(np.full((X.shape[0],), np.nan, dtype=np.float32))
            else:
                baseline_components.append(bcomp.detach().cpu().numpy().reshape(-1))
            braw = out.get("baseline_raw", None)
            if braw is None:
                baseline_raws.append(np.full((X.shape[0],), np.nan, dtype=np.float32))
            else:
                baseline_raws.append(braw.detach().cpu().numpy().reshape(-1))
            rcomp = out.get("residual_component", None)
            if rcomp is None:
                residual_components.append(np.full((X.shape[0],), np.nan, dtype=np.float32))
            else:
                residual_components.append(rcomp.detach().cpu().numpy().reshape(-1))
            mids.extend(batch["mids"])
            seqs.extend(batch["seqs"])
            sources.extend(batch["sources"])
            tf_families.extend(batch["tf_families"])
            tf_classes.extend(batch["tf_classes"])

    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    logit = np.concatenate(logits, axis=0) if logits else np.zeros((0,), dtype=np.float32)
    prior_z = np.concatenate(prior_zs, axis=0) if prior_zs else np.zeros((0,), dtype=np.float32)
    baseline_component = np.concatenate(baseline_components, axis=0) if baseline_components else np.zeros((0,), dtype=np.float32)
    baseline_raw = np.concatenate(baseline_raws, axis=0) if baseline_raws else np.zeros((0,), dtype=np.float32)
    residual_component = np.concatenate(residual_components, axis=0) if residual_components else np.zeros((0,), dtype=np.float32)
    auc, ap = safe_auc_ap(y, prob)
    return {
        "y": y.astype(np.float32),
        "prob": prob.astype(np.float32),
        "logit": logit.astype(np.float32),
        "prior_z": prior_z.astype(np.float32),
        "baseline_component": baseline_component.astype(np.float32),
        "baseline_raw": baseline_raw.astype(np.float32),
        "residual_component": residual_component.astype(np.float32),
        "mids": mids,
        "seqs": seqs,
        "sources": sources,
        "tf_families": tf_families,
        "tf_classes": tf_classes,
        "auc": auc,
        "ap": ap,
    }

def binary_nll_from_probs(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    p = np.clip(prob, 1e-8, 1.0 - 1e-8)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.mean(loss))

def brier_score_binary(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((prob - y_true) ** 2))

def ece_binary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(y_true.size)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(prob[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)

def fit_temperature_binary(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 50) -> float:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    if logits.size == 0 or len(np.unique(y_true)) < 2:
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_t / T, y_t)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(torch.exp(log_t).detach().cpu().item())
    except Exception:
        T = 1.0
    return float(min(max(T, 1e-3), 100.0))

def evaluate_at_threshold(prob: np.ndarray, y_true: np.ndarray, tau: float) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if prob.size == 0:
        return {
            "tau": float(tau),
            "selected": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "fdr": float("nan"),
            "coverage": float("nan"),
        }
    sel = prob >= float(tau)
    tp = int(np.sum(sel & (y_true == 1)))
    fp = int(np.sum(sel & (y_true == 0)))
    tn = int(np.sum((~sel) & (y_true == 0)))
    fn = int(np.sum((~sel) & (y_true == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fdr = fp / max(1, tp + fp)
    coverage = int(np.sum(sel)) / max(1, prob.size)
    return {
        "tau": float(tau),
        "selected": int(np.sum(sel)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fdr": float(fdr),
        "coverage": float(coverage),
    }

def find_tau_for_target_fdr(prob: np.ndarray, y_true: np.ndarray, target_fdr: float) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if prob.size == 0:
        return {
            "tau": 1.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "fdr": float("nan"),
            "coverage": float("nan"),
            "selected": 0,
        }

    order = np.argsort(-prob)
    p = prob[order]
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = np.maximum(tp + fp, 1)
    fdr = fp / denom
    ok = np.where(fdr <= float(target_fdr))[0]

    if ok.size == 0:
        tau = 1.0
        stats = evaluate_at_threshold(prob, y_true, tau)
        return stats

    k = int(ok[-1]) + 1
    tau = float(p[k - 1])
    stats = evaluate_at_threshold(prob, y_true, tau)
    return stats

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    use_aux_losses: bool,
    use_pwm: bool,
    use_prior_z: bool,
    lambda_prior: float,
    lambda_match: float,
    out_dir: str,
    label_smoothing: float = LABEL_SMOOTHING,
    rc_augmentation: bool = RC_AUGMENTATION,
    use_cosine_warmup: bool = False,  # manuscript config = constant LR
    test_loader: Optional[DataLoader] = None,          # test eval per epoch for logging
) -> Tuple[nn.Module, List[Dict], int]:
    best_path = os.path.join(out_dir, "best_model.pt")
    history: List[Dict] = []

    if getattr(model, "arch_name", "") == "pwmconv_exact" or count_trainable_parameters(model) == 0:
        t0 = time.time()
        train_eval = evaluate(model, train_eval_loader, device)
        val_eval = evaluate(model, val_loader, device)
        y_t = torch.tensor(train_eval["y"], dtype=torch.float32)
        logit_t = torch.tensor(train_eval["logit"], dtype=torch.float32)
        if y_t.numel() > 0:
            loss_main = float(F.binary_cross_entropy_with_logits(logit_t, y_t).detach().cpu().item())
        else:
            loss_main = float("nan")
        history.append({
            "epoch": 1,
            "loss": loss_main,
            "loss_main": loss_main,
            "loss_prior": 0.0,
            "loss_match": 0.0,
            "train_auc": train_eval["auc"],
            "train_ap": train_eval["ap"],
            "val_auc": val_eval["auc"],
            "val_ap": val_eval["ap"],
            "lr": 0.0,
            "seconds": float(time.time() - t0),
            "is_best_epoch": 1,
        })
        torch.save(model.state_dict(), best_path)
        return model, history, 1

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine warmup scheduler
    scheduler = None
    if use_cosine_warmup:
        steps_per_epoch = max(1, len(train_loader))
        scheduler = make_cosine_warmup_scheduler(opt, epochs, steps_per_epoch)

    # Label smoothing epsilon
    label_smooth_eps = label_smoothing

    # RC augmentation flag
    rc_aug = rc_augmentation
    best_val_ap = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_n = 0
        main_sum = 0.0
        prior_sum = 0.0
        match_sum = 0.0

        for batch in train_loader:
            X = batch["X"].to(device)
            seq_pad = batch["seq_pad"].to(device)
            feat = batch["feat"].to(device)
            pwm = batch["pwm"].to(device)
            pwm_pad = batch["pwm_pad"].to(device)
            y = batch["y"].to(device).view(-1)
            prior_z = batch["prior_z"].to(device).view(-1)

            # Reverse-complement augmentation — concatenate RC copies
            if rc_aug and X.shape[1] == 4:  # only if one-hot 4-channel
                X_rc = reverse_complement_onehot_batch(X)
                X = torch.cat([X, X_rc], dim=0)
                seq_pad = torch.cat([seq_pad, seq_pad], dim=0)
                feat = torch.cat([feat, feat], dim=0)
                pwm = torch.cat([pwm, pwm], dim=0)
                pwm_pad = torch.cat([pwm_pad, pwm_pad], dim=0)
                y = torch.cat([y, y], dim=0)
                prior_z = torch.cat([prior_z, prior_z], dim=0)

            out = model(X, seq_pad, feat, pwm, pwm_pad)

            # Label smoothing — soft targets
            if label_smooth_eps > 0:
                y_smooth = y * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
            else:
                y_smooth = y
            l_main = F.binary_cross_entropy_with_logits(out["logit"].view(-1), y_smooth)

            l_prior = torch.tensor(0.0, device=device)
            if use_aux_losses and use_prior_z:
                l_prior = F.mse_loss(out["prior_hat"].view(-1), prior_z)

            l_match = torch.tensor(0.0, device=device)
            if use_aux_losses and use_pwm:
                l_match = seq_pwm_match_loss(out["seq_emb"], out["pwm_emb"], y)

            loss = l_main + lambda_prior * l_prior + lambda_match * l_match

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # step scheduler per batch
            if scheduler is not None:
                scheduler.step()

            bs = int(y.size(0))
            # For logging, use original batch size (before RC aug doubling)
            orig_bs = bs // 2 if rc_aug and X.shape[1] == 4 else bs
            train_n += orig_bs
            train_loss += float(loss.detach().cpu()) * orig_bs
            main_sum += float(l_main.detach().cpu()) * orig_bs
            prior_sum += float(l_prior.detach().cpu()) * orig_bs
            match_sum += float(l_match.detach().cpu()) * orig_bs

        train_eval = evaluate(model, train_eval_loader, device)
        val_eval = evaluate(model, val_loader, device)

        # evaluate test per epoch for logging (NO model selection on test!)
        test_eval_ep = None
        if test_loader is not None:
            test_eval_ep = evaluate(model, test_loader, device)

        # compute prior_z baseline AUC/AP per split for logging
        train_prior_auc, train_prior_ap = safe_auc_ap(train_eval["y"], safe_sigmoid_np(train_eval["prior_z"]))
        val_prior_auc, val_prior_ap = safe_auc_ap(val_eval["y"], safe_sigmoid_np(val_eval["prior_z"]))
        test_prior_auc, test_prior_ap = (float("nan"), float("nan"))
        test_auc_ep, test_ap_ep = (float("nan"), float("nan"))
        if test_eval_ep is not None:
            test_prior_auc, test_prior_ap = safe_auc_ap(test_eval_ep["y"], safe_sigmoid_np(test_eval_ep["prior_z"]))
            test_auc_ep = test_eval_ep["auc"]
            test_ap_ep = test_eval_ep["ap"]

        # log residual_gate if present
        gate_val = float("nan")
        if hasattr(model, "residual_gate"):
            gate_val = float(model.residual_gate.detach().cpu().item())

        row = {
            "epoch": int(epoch),
            "loss": train_loss / max(1, train_n),
            "loss_main": main_sum / max(1, train_n),
            "loss_prior": prior_sum / max(1, train_n),
            "loss_match": match_sum / max(1, train_n),
            "train_auc": train_eval["auc"],
            "train_ap": train_eval["ap"],
            "val_auc": val_eval["auc"],
            "val_ap": val_eval["ap"],
            "test_auc": test_auc_ep,
            "test_ap": test_ap_ep,
            "prior_train_auc": train_prior_auc,
            "prior_train_ap": train_prior_ap,
            "prior_val_auc": val_prior_auc,
            "prior_val_ap": val_prior_ap,
            "prior_test_auc": test_prior_auc,
            "prior_test_ap": test_prior_ap,
            "lr": float(opt.param_groups[0]["lr"]),
            "residual_gate": gate_val,
            "seconds": float(time.time() - t0),
            "is_best_epoch": 0,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] loss={row['loss']:.4f} main={row['loss_main']:.4f} "
            f"prior_loss={row['loss_prior']:.4f} match={row['loss_match']:.4f} "
            f"| Train AUC={row['train_auc']:.4f} AP={row['train_ap']:.4f} "
            f"| Val AUC={row['val_auc']:.4f} AP={row['val_ap']:.4f} "
            f"| Test AUC={row['test_auc']:.4f} AP={row['test_ap']:.4f} "
            f"| Prior(val) AP={val_prior_ap:.4f} "
            f"gate={gate_val:.4f} lr={row['lr']:.2e} ({row['seconds']:.1f}s)"
        )

        if not math.isnan(row["val_ap"]) and row["val_ap"] > best_val_ap:
            best_val_ap = float(row["val_ap"])
            best_epoch = int(epoch)
            row["is_best_epoch"] = 1
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if os.path.isfile(best_path):
        try:
            state = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)

    return model, history, best_epoch

def make_datasets_and_stats(
    train_examples: List[LabeledExample],
    val_examples: List[LabeledExample],
    test_examples: List[LabeledExample],
    pwm_index: PWMIndex,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
) -> Tuple[TFBSDataset, TFBSDataset, TFBSDataset, Dict]:
    all_examples = train_examples + val_examples + test_examples
    seqs = [ex.seq for ex in all_examples]
    mids = [ex.motif_id for ex in all_examples]
    prior_raw = compute_raw_prior_scores(seqs, mids, pwm_index)

    n_tr = len(train_examples)
    n_va = len(val_examples)

    train_prior_raw = prior_raw[:n_tr]
    mu = float(train_prior_raw.mean()) if len(train_prior_raw) > 0 else 0.0
    sd = float(train_prior_raw.std() + 1e-6) if len(train_prior_raw) > 0 else 1.0
    prior_z = ((prior_raw - mu) / sd).astype(np.float32)

    feat_mat = make_feature_matrix(all_examples, prior_z, pwm_index, use_prior_z=use_prior_z, use_kmer_fcgr=use_kmer_fcgr)
    feat_norm = compute_feature_stats(feat_mat[:n_tr])

    ds_train = TFBSDataset(train_examples, prior_z[:n_tr], feat_mat[:n_tr], feat_norm, pwm_index)
    ds_val = TFBSDataset(val_examples, prior_z[n_tr:n_tr + n_va], feat_mat[n_tr:n_tr + n_va], feat_norm, pwm_index)
    ds_test = TFBSDataset(test_examples, prior_z[n_tr + n_va:], feat_mat[n_tr + n_va:], feat_norm, pwm_index)

    stats = {
        "prior_mu_train": float(mu),
        "prior_sd_train": float(sd),
        "feature_dim": int(feat_mat.shape[1]),
    }
    return ds_train, ds_val, ds_test, stats
