from __future__ import annotations

"""Interpretability: integrated gradients, IG-IC consistency, PWM reconstruction, prediction export."""

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

from .constants import (BASES)
from .data import (FeatureNormalizer, LabeledExample, collate_batch, make_feature_matrix)
from .pwm_io import (PWMIndex, best_alignment_score, compute_raw_prior_scores)
from .train import (binary_nll_from_probs, brier_score_binary, ece_binary, evaluate_at_threshold, find_tau_for_target_fdr, fit_temperature_binary)
from .utils import (canonical_seq, encode_onehot, pearson_corr, pwm_information_content, reverse_complement_pwm, safe_sigmoid_np, seq_to_idx, spearman_rho, write_csv)

def project_pwm_to_seq_matrix(seq: str, motif_id: str, pwm_index: PWMIndex) -> Dict:
    Ls = len(seq)
    pwm = pwm_index.get_full(motif_id)
    if pwm is None:
        proj = np.full((Ls, 4), 0.25, dtype=np.float32)
        return {
            "projected_pwm": proj,
            "ic": np.zeros((Ls,), dtype=np.float32),
            "mask": np.zeros((Ls,), dtype=bool),
            "score": float("nan"),
            "reverse_complemented": False,
            "mode_pwm_on_seq": True,
            "offset_seq": 0,
            "offset_pwm": 0,
            "pwm_length": 0,
        }

    log_pwm = np.log(np.clip(pwm, 1e-8, 1.0))
    score, off, rev, mode_pwm_on_seq = best_alignment_score(seq_to_idx(seq), log_pwm)
    pwm_use = reverse_complement_pwm(pwm) if rev else pwm.copy()
    Lp = int(pwm_use.shape[0])

    proj = np.full((Ls, 4), 0.25, dtype=np.float32)
    mask = np.zeros((Ls,), dtype=bool)

    if Ls >= Lp:
        start_seq = int(off if not rev else (Ls - off - Lp))
        start_seq = max(0, min(max(0, Ls - Lp), start_seq))
        end_seq = min(Ls, start_seq + Lp)
        use_len = max(0, end_seq - start_seq)
        if use_len > 0:
            proj[start_seq:end_seq] = pwm_use[:use_len]
            mask[start_seq:end_seq] = True
        offset_seq = start_seq
        offset_pwm = 0
    else:
        start_pwm = int(off if not rev else (Lp - off - Ls))
        start_pwm = max(0, min(max(0, Lp - Ls), start_pwm))
        end_pwm = min(Lp, start_pwm + Ls)
        use_len = max(0, end_pwm - start_pwm)
        if use_len == Ls:
            proj[:] = pwm_use[start_pwm:end_pwm]
            mask[:] = True
        else:
            local = pwm_use[start_pwm:end_pwm]
            proj[:local.shape[0]] = local
            mask[:local.shape[0]] = True
        offset_seq = 0
        offset_pwm = start_pwm

    ic = np.zeros((Ls,), dtype=np.float32)
    if np.any(mask):
        ic[mask] = pwm_information_content(proj[mask])

    return {
        "projected_pwm": proj.astype(np.float32),
        "ic": ic.astype(np.float32),
        "mask": mask,
        "score": float(score),
        "reverse_complemented": bool(rev),
        "mode_pwm_on_seq": bool(mode_pwm_on_seq),
        "offset_seq": int(offset_seq),
        "offset_pwm": int(offset_pwm),
        "pwm_length": int(Lp),
    }

def build_analysis_batch(
    seqs: List[str],
    motif_ids: List[str],
    tf_families: List[str],
    tf_classes: List[str],
    pwm_index: PWMIndex,
    feat_norm: FeatureNormalizer,
    prior_mu: float,
    prior_sd: float,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    if not seqs:
        empty = {
            "X": torch.zeros((0, 4, 0), dtype=torch.float32),
            "seq_pad": torch.zeros((0, 0), dtype=torch.bool),
            "pwm": torch.zeros((0, 0, 4), dtype=torch.float32),
            "pwm_pad": torch.zeros((0, 0), dtype=torch.bool),
            "feat": torch.zeros((0, 0), dtype=torch.float32),
            "y": torch.zeros((0, 1), dtype=torch.float32),
            "prior_z": torch.zeros((0, 1), dtype=torch.float32),
            "mids": [],
            "tf_families": [],
            "tf_classes": [],
            "seqs": [],
            "sources": [],
        }
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    examples = [
        LabeledExample(
            seq=str(seq).upper(),
            canon=canonical_seq(str(seq).upper()),
            motif_id=mid,
            tf_family=tfam,
            tf_class=tfcl,
            label=1,
            source="analysis",
        )
        for seq, mid, tfam, tfcl in zip(seqs, motif_ids, tf_families, tf_classes)
    ]

    prior_raw = compute_raw_prior_scores([ex.seq for ex in examples], motif_ids, pwm_index)
    sd = float(prior_sd) if abs(float(prior_sd)) > 1e-8 else 1.0
    prior_z = ((prior_raw - float(prior_mu)) / sd).astype(np.float32)
    feat_mat = make_feature_matrix(examples, prior_z, pwm_index, use_prior_z=use_prior_z, use_kmer_fcgr=use_kmer_fcgr)

    batch = []
    for ex, pz, feat_row in zip(examples, prior_z, feat_mat):
        X = encode_onehot(ex.seq)
        pwm = pwm_index.get_full(ex.motif_id)
        if pwm is None:
            pwm = np.full((len(ex.seq), 4), 0.25, dtype=np.float32)
        feat = feat_norm.apply(feat_row)
        batch.append({
            "X": torch.from_numpy(X),
            "seq_len": int(X.shape[1]),
            "feat": torch.from_numpy(feat.astype(np.float32)),
            "pwm": torch.from_numpy(pwm.astype(np.float32)),
            "pwm_len": int(pwm.shape[0]),
            "y": torch.tensor([1.0], dtype=torch.float32),
            "prior_z": torch.tensor([float(pz)], dtype=torch.float32),
            "mid": ex.motif_id,
            "tf_family": ex.tf_family,
            "tf_class": ex.tf_class,
            "source": ex.source,
            "seq": ex.seq,
        })

    return collate_batch(batch), prior_raw.astype(np.float32), prior_z.astype(np.float32)

def predict_sequences(
    model: nn.Module,
    seqs: List[str],
    motif_ids: List[str],
    tf_families: List[str],
    tf_classes: List[str],
    pwm_index: PWMIndex,
    feat_norm: FeatureNormalizer,
    prior_mu: float,
    prior_sd: float,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    logits_all = []
    probs_all = []
    model.eval()
    for s0 in range(0, len(seqs), max(1, batch_size)):
        s1 = min(len(seqs), s0 + max(1, batch_size))
        batch, _, _ = build_analysis_batch(
            seqs[s0:s1],
            motif_ids[s0:s1],
            tf_families[s0:s1],
            tf_classes[s0:s1],
            pwm_index=pwm_index,
            feat_norm=feat_norm,
            prior_mu=prior_mu,
            prior_sd=prior_sd,
            use_prior_z=use_prior_z,
            use_kmer_fcgr=use_kmer_fcgr,
        )
        X = batch["X"].to(device)
        seq_pad = batch["seq_pad"].to(device)
        feat = batch["feat"].to(device)
        pwm = batch["pwm"].to(device)
        pwm_pad = batch["pwm_pad"].to(device)
        with torch.no_grad():
            out = model(X, seq_pad, feat, pwm, pwm_pad)
        logits_all.append(out["logit"].detach().cpu().numpy().reshape(-1))
        probs_all.append(out["prob"].detach().cpu().numpy().reshape(-1))
    if not logits_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (
        np.concatenate(logits_all, axis=0).astype(np.float32),
        np.concatenate(probs_all, axis=0).astype(np.float32),
    )

def export_predictions_csv(
    path: str,
    eval_dict: Dict,
    split_name: str,
    calibrated_prob: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    baseline_scores: Optional[np.ndarray] = None,
) -> None:
    n = len(eval_dict["y"])
    if calibrated_prob is None:
        calibrated_prob = np.full((n,), np.nan, dtype=np.float32)
    if baseline_scores is None:
        baseline_scores = np.full((n,), np.nan, dtype=np.float32)
    model_base = np.asarray(eval_dict.get("baseline_component", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)
    model_base_raw = np.asarray(eval_dict.get("baseline_raw", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)
    model_resid = np.asarray(eval_dict.get("residual_component", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)

    rows = []
    for i in range(n):
        pred_tau = ""
        if tau is not None and np.isfinite(calibrated_prob[i]):
            pred_tau = int(calibrated_prob[i] >= tau)
        rows.append([
            split_name,
            eval_dict["mids"][i],
            eval_dict["tf_families"][i],
            eval_dict["tf_classes"][i],
            eval_dict["seqs"][i],
            eval_dict["sources"][i],
            int(eval_dict["y"][i]),
            float(baseline_scores[i]) if np.isfinite(baseline_scores[i]) else "",
            float(model_base[i]) if np.isfinite(model_base[i]) else "",
            float(model_base_raw[i]) if np.isfinite(model_base_raw[i]) else "",
            float(model_resid[i]) if np.isfinite(model_resid[i]) else "",
            float(eval_dict["logit"][i]),
            float(eval_dict["prob"][i]),
            float(calibrated_prob[i]) if np.isfinite(calibrated_prob[i]) else "",
            pred_tau,
        ])

    write_csv(
        path,
        ["split", "motif_id", "tf_family", "tf_class", "seq", "source", "label", "prior_baseline_score", "model_baseline_component", "model_baseline_raw", "model_residual_component", "logit", "prob", "prob_calibrated", "pred_tau"],
        rows,
    )

def select_test_indices_by_motif(eval_dict: Dict, label_value: int = 1, max_per_motif: int = 32) -> List[int]:
    by_mid: Dict[str, List[int]] = defaultdict(list)
    for i, (mid, y) in enumerate(zip(eval_dict["mids"], eval_dict["y"])):
        if int(y) == int(label_value):
            by_mid[mid].append(i)

    chosen = []
    for mid in sorted(by_mid):
        idx = by_mid[mid]
        idx.sort(key=lambda i: float(eval_dict["prob"][i]), reverse=True)
        chosen.extend(idx[:max(1, int(max_per_motif))])
    return chosen

def integrated_gradients_single(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    steps: int = 32,
) -> np.ndarray:
    X = batch["X"].to(device)
    seq_pad = batch["seq_pad"].to(device)
    feat = batch["feat"].to(device)
    pwm = batch["pwm"].to(device)
    pwm_pad = batch["pwm_pad"].to(device)

    baseline = torch.zeros_like(X)
    total_grad = torch.zeros_like(X)

    model.eval()
    for k in range(1, max(2, int(steps)) + 1):
        alpha = float(k) / float(max(2, int(steps)))
        x = (baseline + alpha * (X - baseline)).detach()
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x, seq_pad, feat, pwm, pwm_pad)["logit"].sum()
        grad = torch.autograd.grad(out, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / float(max(2, int(steps)))
    ig = ((X - baseline) * avg_grad).detach().cpu().numpy()[0]
    return ig.astype(np.float32)

def export_ig_and_consistency(
    exp_dir: str,
    model: nn.Module,
    test_eval: Dict,
    pwm_index: PWMIndex,
    feat_norm: FeatureNormalizer,
    prior_mu: float,
    prior_sd: float,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
    device: torch.device,
    ig_steps: int,
    ig_max_samples_per_motif: int,
) -> Tuple[int, int]:
    selected_idx = select_test_indices_by_motif(test_eval, label_value=1, max_per_motif=ig_max_samples_per_motif)

    attr_rows = []
    cons_rows = []
    for sample_rank, idx in enumerate(selected_idx, start=1):
        seq = test_eval["seqs"][idx]
        mid = test_eval["mids"][idx]
        tf_family = test_eval["tf_families"][idx]
        tf_class = test_eval["tf_classes"][idx]
        batch, _, _ = build_analysis_batch(
            [seq],
            [mid],
            [tf_family],
            [tf_class],
            pwm_index=pwm_index,
            feat_norm=feat_norm,
            prior_mu=prior_mu,
            prior_sd=prior_sd,
            use_prior_z=use_prior_z,
            use_kmer_fcgr=use_kmer_fcgr,
        )
        ig = integrated_gradients_single(model, batch, device=device, steps=ig_steps)
        pos_importance = np.sum(np.abs(ig), axis=0).astype(np.float32)

        align = project_pwm_to_seq_matrix(seq, mid, pwm_index)
        ic = align["ic"].astype(np.float32)
        rho = spearman_rho(pos_importance, ic)
        r = pearson_corr(pos_importance, ic)
        sample_id = f"{mid}__{sample_rank:05d}"

        cons_rows.append([
            sample_id,
            mid,
            tf_family,
            tf_class,
            int(test_eval["y"][idx]),
            seq,
            float(test_eval["logit"][idx]),
            float(test_eval["prob"][idx]),
            rho,
            r,
            float(np.mean(pos_importance)),
            float(np.max(pos_importance)) if pos_importance.size else float("nan"),
            float(np.mean(ic)) if ic.size else float("nan"),
            float(np.max(ic)) if ic.size else float("nan"),
            align["score"],
            int(align["reverse_complemented"]),
            int(align["mode_pwm_on_seq"]),
            align["offset_seq"],
            align["offset_pwm"],
            align["pwm_length"],
        ])

        for pos in range(ig.shape[1]):
            for bi, base in enumerate(BASES):
                attr_rows.append([
                    sample_id,
                    mid,
                    tf_family,
                    tf_class,
                    int(test_eval["y"][idx]),
                    seq,
                    pos + 1,
                    base,
                    int(seq[pos] == base),
                    float(ig[bi, pos]),
                    float(abs(ig[bi, pos])),
                    float(test_eval["logit"][idx]),
                    float(test_eval["prob"][idx]),
                ])

    write_csv(
        os.path.join(exp_dir, "ig_attributions.csv"),
        ["sample_id", "motif_id", "tf_family", "tf_class", "label", "seq", "position", "base", "onehot", "ig", "ig_abs", "logit", "prob"],
        attr_rows,
    )
    write_csv(
        os.path.join(exp_dir, "ig_pwm_consistency.csv"),
        ["sample_id", "motif_id", "tf_family", "tf_class", "label", "seq", "logit", "prob", "spearman_rho", "pearson_r", "mean_abs_ig", "max_abs_ig", "mean_ic", "max_ic", "alignment_score", "reverse_complemented", "mode_pwm_on_seq", "offset_seq", "offset_pwm", "pwm_length"],
        cons_rows,
    )
    return int(len(attr_rows)), int(len(cons_rows))

def export_pwm_reconstruction(
    exp_dir: str,
    model: nn.Module,
    test_eval: Dict,
    pwm_index: PWMIndex,
    feat_norm: FeatureNormalizer,
    prior_mu: float,
    prior_sd: float,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
    device: torch.device,
    fixed_len: int,
    max_samples_per_motif: int,
    pred_batch_size: int,
) -> Tuple[int, int]:
    by_mid: Dict[str, List[int]] = defaultdict(list)
    for i, (mid, y) in enumerate(zip(test_eval["mids"], test_eval["y"])):
        if int(y) == 1:
            by_mid[mid].append(i)

    summary_rows = []
    matrix_rows = []

    motif_items = sorted(by_mid.items(), key=lambda kv: kv[0])
    for motif_i, (mid, idxs) in enumerate(motif_items, start=1):
        idxs = sorted(idxs, key=lambda i: float(test_eval["prob"][i]), reverse=True)[:max(1, int(max_samples_per_motif))]
        if not idxs:
            continue

        target_acc = np.zeros((fixed_len, 4), dtype=np.float64)
        pred_acc = np.zeros((fixed_len, 4), dtype=np.float64)
        ic_acc = np.zeros((fixed_len,), dtype=np.float64)

        tf_family = test_eval["tf_families"][idxs[0]]
        tf_class = test_eval["tf_classes"][idxs[0]]

        for idx in idxs:
            seq = test_eval["seqs"][idx]
            align = project_pwm_to_seq_matrix(seq, mid, pwm_index)
            target_acc += align["projected_pwm"]
            ic_acc += align["ic"]

            mut_seqs = []
            mut_mids = []
            mut_fams = []
            mut_clss = []
            for pos in range(fixed_len):
                for base in BASES:
                    mseq = seq[:pos] + base + seq[pos + 1:]
                    mut_seqs.append(mseq)
                    mut_mids.append(mid)
                    mut_fams.append(tf_family)
                    mut_clss.append(tf_class)

            logits_mut, _ = predict_sequences(
                model=model,
                seqs=mut_seqs,
                motif_ids=mut_mids,
                tf_families=mut_fams,
                tf_classes=mut_clss,
                pwm_index=pwm_index,
                feat_norm=feat_norm,
                prior_mu=prior_mu,
                prior_sd=prior_sd,
                use_prior_z=use_prior_z,
                use_kmer_fcgr=use_kmer_fcgr,
                device=device,
                batch_size=pred_batch_size,
            )
            if logits_mut.size != fixed_len * 4:
                continue
            logits_mat = logits_mut.reshape(fixed_len, 4).astype(np.float64)
            logits_mat = logits_mat - logits_mat.max(axis=1, keepdims=True)
            pred_mat = np.exp(logits_mat)
            pred_mat = pred_mat / np.clip(pred_mat.sum(axis=1, keepdims=True), 1e-8, None)
            pred_acc += pred_mat

        n_used = max(1, len(idxs))
        target_mean = (target_acc / n_used).astype(np.float64)
        pred_mean = (pred_acc / n_used).astype(np.float64)
        ic_mean = (ic_acc / n_used).astype(np.float64)

        target_mean = np.clip(target_mean, 1e-8, 1.0)
        pred_mean = np.clip(pred_mean, 1e-8, 1.0)
        target_mean = target_mean / np.clip(target_mean.sum(axis=1, keepdims=True), 1e-8, None)
        pred_mean = pred_mean / np.clip(pred_mean.sum(axis=1, keepdims=True), 1e-8, None)

        active_mask = ic_mean > 0.05
        if not np.any(active_mask):
            active_mask = np.ones((fixed_len,), dtype=bool)

        t = target_mean[active_mask]
        p = pred_mean[active_mask]
        kl_tp = float(np.mean(np.sum(t * np.log(np.clip(t / p, 1e-8, 1e8)), axis=1)))
        kl_pt = float(np.mean(np.sum(p * np.log(np.clip(p / t, 1e-8, 1e8)), axis=1)))
        m = 0.5 * (t + p)
        js = float(0.5 * np.mean(np.sum(t * np.log(np.clip(t / m, 1e-8, 1e8)), axis=1)) + 0.5 * np.mean(np.sum(p * np.log(np.clip(p / m, 1e-8, 1e8)), axis=1)))
        mad = float(np.mean(np.abs(t - p)))

        summary_rows.append([
            mid,
            tf_family,
            tf_class,
            int(n_used),
            int(np.sum(active_mask)),
            kl_tp,
            kl_pt,
            js,
            mad,
            float(np.mean(ic_mean[active_mask])) if np.any(active_mask) else float("nan"),
        ])

        for pos in range(fixed_len):
            for bi, base in enumerate(BASES):
                matrix_rows.append([
                    mid,
                    tf_family,
                    tf_class,
                    pos + 1,
                    base,
                    float(target_mean[pos, bi]),
                    float(pred_mean[pos, bi]),
                    float(ic_mean[pos]),
                    int(n_used),
                ])

    write_csv(
        os.path.join(exp_dir, "pwm_reconstruction_summary.csv"),
        ["motif_id", "tf_family", "tf_class", "n_samples_used", "n_active_positions", "kl_target_to_pred", "kl_pred_to_target", "js_divergence", "mean_abs_diff", "mean_target_ic"],
        summary_rows,
    )
    write_csv(
        os.path.join(exp_dir, "pwm_reconstruction_matrix.csv"),
        ["motif_id", "tf_family", "tf_class", "position", "base", "target_prob", "pred_prob", "target_ic", "n_samples_used"],
        matrix_rows,
    )
    return int(len(summary_rows)), int(len(matrix_rows))

def export_calibration(
    exp_dir: str,
    val_eval: Dict,
    test_eval: Dict,
    target_fdr: float,
    min_cluster_n: int,
) -> Tuple[float, float, np.ndarray]:
    val_logits = np.asarray(val_eval["logit"], dtype=np.float64)
    val_y = np.asarray(val_eval["y"], dtype=np.int64)
    test_logits = np.asarray(test_eval["logit"], dtype=np.float64)
    test_y = np.asarray(test_eval["y"], dtype=np.int64)

    T_global = fit_temperature_binary(val_logits, val_y)
    val_prob_global = safe_sigmoid_np(val_logits / T_global)
    tau_global_stats = find_tau_for_target_fdr(val_prob_global, val_y, target_fdr)
    tau_global = float(tau_global_stats["tau"])
    test_prob_global = safe_sigmoid_np(test_logits / T_global)

    rows = []

    def field(eval_dict: Dict, cluster_type: str) -> List[str]:
        if cluster_type == "motif_id":
            return list(eval_dict["mids"])
        if cluster_type == "tf_family":
            return list(eval_dict["tf_families"])
        if cluster_type == "tf_class":
            return list(eval_dict["tf_classes"])
        raise ValueError(cluster_type)

    cluster_types = ["global", "motif_id", "tf_family", "tf_class"]
    for cluster_type in cluster_types:
        if cluster_type == "global":
            names = ["__ALL__"]
        else:
            names = sorted(set(field(test_eval, cluster_type)))

        for name in names:
            if cluster_type == "global":
                val_idx = np.arange(len(val_y), dtype=np.int64)
                test_idx = np.arange(len(test_y), dtype=np.int64)
            else:
                val_idx = np.asarray([i for i, x in enumerate(field(val_eval, cluster_type)) if x == name], dtype=np.int64)
                test_idx = np.asarray([i for i, x in enumerate(field(test_eval, cluster_type)) if x == name], dtype=np.int64)

            if test_idx.size == 0:
                continue

            use_local = (
                cluster_type != "global"
                and val_idx.size >= int(min_cluster_n)
                and len(np.unique(val_y[val_idx])) >= 2
            )

            if use_local:
                T = fit_temperature_binary(val_logits[val_idx], val_y[val_idx])
                val_prob = safe_sigmoid_np(val_logits[val_idx] / T)
                tau_stats = find_tau_for_target_fdr(val_prob, val_y[val_idx], target_fdr)
                tau_source = "cluster"
            else:
                T = T_global
                tau_stats = tau_global_stats
                tau_source = "global"
                val_prob = safe_sigmoid_np(val_logits[val_idx] / T) if val_idx.size else np.zeros((0,), dtype=np.float64)

            tau = float(tau_stats["tau"])
            test_prob = safe_sigmoid_np(test_logits[test_idx] / T)
            test_stats = evaluate_at_threshold(test_prob, test_y[test_idx], tau)

            rows.append([
                cluster_type,
                name,
                int(val_idx.size),
                int(test_idx.size),
                int(np.sum(val_y[val_idx] == 1)) if val_idx.size else 0,
                int(np.sum(test_y[test_idx] == 1)),
                float(T),
                float(tau),
                tau_source,
                float(target_fdr),
                float(tau_stats["fdr"]) if not math.isnan(tau_stats["fdr"]) else float("nan"),
                float(test_stats["fdr"]) if not math.isnan(test_stats["fdr"]) else float("nan"),
                float(test_stats["precision"]) if not math.isnan(test_stats["precision"]) else float("nan"),
                float(test_stats["recall"]) if not math.isnan(test_stats["recall"]) else float("nan"),
                float(test_stats["coverage"]) if not math.isnan(test_stats["coverage"]) else float("nan"),
                int(test_stats["selected"]),
                brier_score_binary(test_y[test_idx], test_prob),
                ece_binary(test_y[test_idx], test_prob, n_bins=10),
                binary_nll_from_probs(test_y[test_idx], test_prob),
            ])

    write_csv(
        os.path.join(exp_dir, "calibration.csv"),
        ["cluster_type", "cluster_name", "n_val", "n_test", "n_val_pos", "n_test_pos", "temperature", "tau", "tau_source", "target_fdr", "val_fdr_at_tau", "test_fdr_at_tau", "test_precision_at_tau", "test_recall_at_tau", "test_coverage_at_tau", "test_selected", "test_brier", "test_ece", "test_nll"],
        rows,
    )
    return float(T_global), float(tau_global), test_prob_global.astype(np.float32)
