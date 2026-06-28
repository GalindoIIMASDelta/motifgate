from __future__ import annotations

"""Paper-ready CSV/JSON exporters and summary builders."""

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

from .constants import (BATCH_SIZE, DROPOUT, D_MODEL, EPOCHS, FIXED_LEN, LABEL_SMOOTHING, LR, NEG_PER_POS, NUM_EXPERTS, N_HEADS, PATIENCE, RESIDUAL_GATE_INIT, WEIGHT_DECAY)
from .utils import (count_trainable_parameters, safe_auc_ap, write_csv, write_json)

def export_architecture_summary(model: nn.Module, args, exp_dir: str, feat_dim: int, prior_mu: float, prior_sd: float) -> None:
    """Export detailed architecture description as CSV and print to log."""
    rows = []
    arch_name = getattr(model, "arch_name", type(model).__name__)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_parameters(model)

    rows.append(["architecture", arch_name])
    rows.append(["total_parameters", int(total_params)])
    rows.append(["trainable_parameters", int(trainable_params)])
    rows.append(["d_model", int(getattr(args, "d_model", D_MODEL))])
    rows.append(["n_heads", int(getattr(args, "n_heads", N_HEADS))])
    rows.append(["num_experts", int(getattr(args, "num_experts", NUM_EXPERTS))])
    rows.append(["dropout", float(getattr(args, "dropout", DROPOUT))])
    rows.append(["feature_dim", int(feat_dim)])
    rows.append(["seq_channels", 4])
    rows.append(["fixed_len", int(getattr(args, "fixed_len", FIXED_LEN))])
    rows.append(["use_pwm", int(getattr(args, "use_pwm", 1))])
    rows.append(["use_cross_attention", int(getattr(args, "use_cross_attention", 1))])
    rows.append(["use_moe", int(getattr(args, "use_moe", 1))])
    rows.append(["use_se", int(getattr(args, "use_se", 1))])
    rows.append(["use_kmer_fcgr", int(getattr(args, "use_kmer_fcgr", 0))])
    rows.append(["use_aux_losses", int(getattr(args, "use_aux_losses", 1))])
    rows.append(["lambda_prior", float(getattr(args, "lambda_prior", 0.10))])
    rows.append(["lambda_match", float(getattr(args, "lambda_match", 0.05))])
    rows.append(["label_smoothing", float(getattr(args, "label_smoothing", LABEL_SMOOTHING))])
    rows.append(["rc_augmentation", int(getattr(args, "rc_augmentation", 1))])
    rows.append(["use_cosine_warmup", int(getattr(args, "use_cosine_warmup", 0))])
    rows.append(["residual_gate_init", float(getattr(args, "residual_gate_init", RESIDUAL_GATE_INIT))])
    rows.append(["prior_mu_train", float(prior_mu)])
    rows.append(["prior_sd_train", float(prior_sd)])
    rows.append(["epochs", int(getattr(args, "epochs", EPOCHS))])
    rows.append(["patience", int(getattr(args, "patience", PATIENCE))])
    rows.append(["lr", float(getattr(args, "lr", LR))])
    rows.append(["weight_decay", float(getattr(args, "weight_decay", WEIGHT_DECAY))])
    rows.append(["batch_size", int(getattr(args, "batch_size", BATCH_SIZE))])
    rows.append(["neg_per_pos", int(getattr(args, "neg_per_pos", NEG_PER_POS))])

    # Per-module layer detail
    rows.append(["---", "--- Module detail ---"])
    for name, module in model.named_modules():
        if name == "":
            continue
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params > 0:
            rows.append([f"module:{name}", f"params={n_params} type={type(module).__name__}"])

    write_csv(os.path.join(exp_dir, "architecture_summary.csv"), ["key", "value"], rows)

    # Print to log
    print("=" * 80)
    print(f"[ARCHITECTURE] {arch_name}  params={trainable_params:,}  d_model={getattr(args, 'd_model', D_MODEL)}")
    print(f"  SE={getattr(args, 'use_se', 1)}  MoE={getattr(args, 'use_moe', 1)}  "
          f"CrossAttn={getattr(args, 'use_cross_attention', 1)}  "
          f"kmer_fcgr={getattr(args, 'use_kmer_fcgr', 0)}  "
          f"LabelSmooth={getattr(args, 'label_smoothing', LABEL_SMOOTHING)}  "
          f"RC_aug={getattr(args, 'rc_augmentation', 1)}  "
          f"gate_init={getattr(args, 'residual_gate_init', RESIDUAL_GATE_INIT)}")
    print(f"  prior_mu={prior_mu:.4f}  prior_sd={prior_sd:.4f}  feat_dim={feat_dim}")
    print("=" * 80)

def export_worst_k_detailed(exp_dir: str, per_motif_rows: List[List], k: int, history: List[Dict]) -> None:
    """Enhanced worst-K motif export with training trajectory context."""
    worst = per_motif_rows[:max(1, k)]

    # Add columns for interpretation
    enhanced_rows = []
    for row in worst:
        # row format: [rank, motif_id, tf_family, tf_class, n_total, n_pos, n_neg,
        #              test_auc, test_ap, baseline_auc, baseline_ap, delta_auc, delta_ap]
        motif_id = row[1]
        n_pos = int(row[5])
        test_ap = float(row[8])
        baseline_ap = float(row[10])
        delta_ap = float(row[12])
        # Classification of difficulty
        if delta_ap > 0.02:
            verdict = "model_helps"
        elif delta_ap > -0.02:
            verdict = "tied_with_prior"
        else:
            verdict = "prior_wins"
        # Data scarcity flag
        scarcity = "low_data" if n_pos < 20 else ("medium_data" if n_pos < 100 else "sufficient_data")
        enhanced_rows.append(row + [verdict, scarcity])

    write_csv(
        os.path.join(exp_dir, "worst_k_motifs_enhanced.csv"),
        ["rank_test_ap", "motif_id", "tf_family", "tf_class", "n_total", "n_pos", "n_neg",
         "test_auc", "test_ap", "baseline_auc", "baseline_ap",
         "delta_auc_vs_baseline", "delta_ap_vs_baseline", "verdict", "data_regime"],
        enhanced_rows,
    )

def write_k_matched_exports(results_root: str, k_list: List[int], pool_rows: List[List], summary: Dict) -> None:
    header = [
        "keep",
        "raw_len",
        "motif_id",
        "tf_family",
        "tf_class",
        "source_file",
        "source_index",
        "canonical_raw",
        "raw_seq",
        "conflict_degree",
        "conflict_bucket_count",
    ] + [f"crop_K{k}" for k in k_list]
    write_csv(os.path.join(results_root, "k_matched_source_pool.csv"), header, pool_rows)

    summary_rows = []
    for k in k_list:
        summary_rows.append([
            k,
            summary.get("integrity", {}).get(f"K{k}_n_examples", 0),
            summary.get("integrity", {}).get(f"K{k}_n_unique", 0),
            summary.get("integrity", {}).get(f"K{k}_n_duplicates", 0),
        ])
    write_csv(
        os.path.join(results_root, "k_matched_integrity.csv"),
        ["fixed_len", "n_examples", "n_unique_crops", "n_duplicate_crops"],
        summary_rows,
    )

    write_csv(
        os.path.join(results_root, "k_matched_pool_summary.csv"),
        ["k_list", "n_input", "n_kept", "retention_frac", "matching_strategy", "mean_conflict_degree_input", "median_conflict_degree_input", "n_with_any_conflict"],
        [[
            ",".join(str(k) for k in k_list),
            summary.get("n_input", 0),
            summary.get("n_kept", 0),
            summary.get("retention_frac", 0.0),
            summary.get("matching_strategy", ""),
            summary.get("mean_conflict_degree_input", 0.0),
            summary.get("median_conflict_degree_input", 0.0),
            summary.get("n_with_any_conflict", 0),
        ]],
    )
    write_json(os.path.join(results_root, "k_matched_pool_summary.json"), summary)

def write_k_comparison_exports(results_root: str, metrics_records: List[Dict]) -> None:
    ks = sorted(set(int(m.get("fixed_len", -1)) for m in metrics_records if str(m.get("fixed_len", "")).strip() != ""))
    if len(ks) <= 1:
        return

    by_group_rows = []
    for m in metrics_records:
        if "fixed_len" not in m:
            continue
        by_group_rows.append([
            m["protocol"],
            m["group_by"],
            m["group_name"],
            m["fixed_len"],
            m["experiment_id"],
            m["best_epoch"],
            m["n_train_pos"],
            m["n_val_pos"],
            m["n_test_pos"],
            m["n_train_total"],
            m["n_val_total"],
            m["n_test_total"],
            m["train_auc"],
            m["train_ap"],
            m["val_auc"],
            m["val_ap"],
            m["test_auc"],
            m["test_ap"],
            m["prior_baseline_test_auc"],
            m["prior_baseline_test_ap"],
            m["delta_test_auc_vs_prior"],
            m["delta_test_ap_vs_prior"],
            m["temperature_global"],
            m["tau_global_fdr"],
            m["exp_dir"],
        ])
    write_csv(
        os.path.join(results_root, "k_comparison_by_group.csv"),
        [
            "protocol", "group_by", "group_name", "fixed_len", "experiment_id", "best_epoch",
            "n_train_pos", "n_val_pos", "n_test_pos", "n_train_total", "n_val_total", "n_test_total",
            "train_auc", "train_ap", "val_auc", "val_ap", "test_auc", "test_ap",
            "prior_baseline_test_auc", "prior_baseline_test_ap",
            "delta_test_auc_vs_prior", "delta_test_ap_vs_prior",
            "temperature_global", "tau_global_fdr", "exp_dir"
        ],
        by_group_rows,
    )

    summary_rows = []
    by_k = defaultdict(list)
    for m in metrics_records:
        if "fixed_len" in m:
            by_k[int(m["fixed_len"])].append(m)

    def _safe_mean(vals):
        vals = [float(v) for v in vals if not math.isnan(float(v))]
        return float(np.mean(vals)) if vals else float("nan")

    def _safe_median(vals):
        vals = [float(v) for v in vals if not math.isnan(float(v))]
        return float(np.median(vals)) if vals else float("nan")

    def _safe_wmean(vals, weights):
        arr = []
        wts = []
        for v, w in zip(vals, weights):
            v = float(v)
            if math.isnan(v):
                continue
            arr.append(v)
            wts.append(max(0.0, float(w)))
        if not arr:
            return float("nan")
        sw = sum(wts)
        if sw <= 0:
            return float(np.mean(arr))
        return float(np.average(np.asarray(arr, dtype=np.float64), weights=np.asarray(wts, dtype=np.float64)))

    for k in ks:
        items = by_k.get(k, [])
        if not items:
            continue
        weights = [m.get("n_test_pos", 0) for m in items]
        summary_rows.append([
            k,
            len(items),
            _safe_mean([m["test_auc"] for m in items]),
            _safe_median([m["test_auc"] for m in items]),
            _safe_wmean([m["test_auc"] for m in items], weights),
            _safe_mean([m["test_ap"] for m in items]),
            _safe_median([m["test_ap"] for m in items]),
            _safe_wmean([m["test_ap"] for m in items], weights),
            _safe_mean([m["prior_baseline_test_auc"] for m in items]),
            _safe_mean([m["prior_baseline_test_ap"] for m in items]),
            _safe_mean([m["delta_test_auc_vs_prior"] for m in items]),
            _safe_median([m["delta_test_auc_vs_prior"] for m in items]),
            _safe_mean([m["delta_test_ap_vs_prior"] for m in items]),
            _safe_median([m["delta_test_ap_vs_prior"] for m in items]),
            int(sum(int(m.get("n_test_pos", 0)) for m in items)),
            int(sum(int(m.get("n_test_total", 0)) for m in items)),
        ])
    write_csv(
        os.path.join(results_root, "k_comparison_summary.csv"),
        [
            "fixed_len", "n_experiments",
            "mean_test_auc", "median_test_auc", "weighted_mean_test_auc",
            "mean_test_ap", "median_test_ap", "weighted_mean_test_ap",
            "mean_prior_baseline_test_auc", "mean_prior_baseline_test_ap",
            "mean_delta_test_auc_vs_prior", "median_delta_test_auc_vs_prior",
            "mean_delta_test_ap_vs_prior", "median_delta_test_ap_vs_prior",
            "sum_n_test_pos", "sum_n_test_total"
        ],
        summary_rows,
    )

    pair_rows = []
    by_key = defaultdict(dict)
    for m in metrics_records:
        if "fixed_len" not in m:
            continue
        key = (m["protocol"], m["group_by"], m["group_name"])
        by_key[key][int(m["fixed_len"])] = m

    for key, sub in sorted(by_key.items()):
        ks_here = sorted(sub)
        for i in range(len(ks_here)):
            for j in range(i + 1, len(ks_here)):
                k1, k2 = ks_here[i], ks_here[j]
                m1, m2 = sub[k1], sub[k2]
                pair_rows.append([
                    key[0], key[1], key[2],
                    k1, k2,
                    m2["test_auc"] - m1["test_auc"] if not math.isnan(m2["test_auc"]) and not math.isnan(m1["test_auc"]) else float("nan"),
                    m2["test_ap"] - m1["test_ap"] if not math.isnan(m2["test_ap"]) and not math.isnan(m1["test_ap"]) else float("nan"),
                    m2["delta_test_auc_vs_prior"] - m1["delta_test_auc_vs_prior"] if not math.isnan(m2["delta_test_auc_vs_prior"]) and not math.isnan(m1["delta_test_auc_vs_prior"]) else float("nan"),
                    m2["delta_test_ap_vs_prior"] - m1["delta_test_ap_vs_prior"] if not math.isnan(m2["delta_test_ap_vs_prior"]) and not math.isnan(m1["delta_test_ap_vs_prior"]) else float("nan"),
                ])
    write_csv(
        os.path.join(results_root, "k_pairwise_deltas.csv"),
        ["protocol", "group_by", "group_name", "k_from", "k_to", "delta_test_auc", "delta_test_ap", "delta_delta_auc_vs_prior", "delta_delta_ap_vs_prior"],
        pair_rows,
    )

def prior_baseline_rows_for_split(split_name: str, eval_dict: Dict, model_auc: float, model_ap: float) -> List:
    scores = np.asarray(eval_dict["prior_z"], dtype=np.float64)
    auc, ap = safe_auc_ap(np.asarray(eval_dict["y"], dtype=np.float64), scores)
    y = np.asarray(eval_dict["y"], dtype=np.float64)
    return [
        split_name,
        int(len(y)),
        int(np.sum(y == 1)),
        int(np.sum(y == 0)),
        auc,
        ap,
        model_auc,
        model_ap,
        (float(model_auc) - float(auc)) if not math.isnan(model_auc) and not math.isnan(auc) else float("nan"),
        (float(model_ap) - float(ap)) if not math.isnan(model_ap) and not math.isnan(ap) else float("nan"),
    ]

def export_baseline_consistency_and_decomposition(exp_dir: str, split_evals: List[Tuple[str, Dict]]) -> Dict[str, Dict[str, float]]:
    consistency_rows = []
    decomposition_rows = []
    summary = {}
    for split_name, ev in split_evals:
        prior = np.asarray(ev.get("prior_z", []), dtype=np.float64).reshape(-1)
        bcomp = np.asarray(ev.get("baseline_component", []), dtype=np.float64).reshape(-1)
        braw = np.asarray(ev.get("baseline_raw", []), dtype=np.float64).reshape(-1)
        resid = np.asarray(ev.get("residual_component", []), dtype=np.float64).reshape(-1)
        y = np.asarray(ev.get("y", []), dtype=np.float64).reshape(-1)

        finite = np.isfinite(prior) & np.isfinite(bcomp)
        max_abs = float(np.max(np.abs(prior[finite] - bcomp[finite]))) if np.any(finite) else float("nan")
        mean_abs = float(np.mean(np.abs(prior[finite] - bcomp[finite]))) if np.any(finite) else float("nan")
        auc_b, ap_b = safe_auc_ap(y[finite], bcomp[finite]) if np.any(finite) else (float("nan"), float("nan"))
        auc_p, ap_p = safe_auc_ap(y[finite], prior[finite]) if np.any(finite) else (float("nan"), float("nan"))
        summary[split_name] = {
            "max_abs_diff_prior_vs_model_baseline": max_abs,
            "mean_abs_diff_prior_vs_model_baseline": mean_abs,
            "model_baseline_auc": auc_b,
            "model_baseline_ap": ap_b,
            "prior_auc": auc_p,
            "prior_ap": ap_p,
        }
        consistency_rows.append([
            split_name,
            int(y.size),
            max_abs,
            mean_abs,
            auc_b,
            ap_b,
            auc_p,
            ap_p,
        ])

        for label_name, mask in [
            ("all", np.ones_like(y, dtype=bool)),
            ("pos", y == 1),
            ("neg", y == 0),
        ]:
            if y.size == 0 or not np.any(mask):
                continue
            decomposition_rows.append([
                split_name,
                label_name,
                int(np.sum(mask)),
                float(np.mean(prior[mask])) if prior.size else float("nan"),
                float(np.mean(bcomp[mask])) if bcomp.size else float("nan"),
                float(np.mean(braw[mask])) if braw.size else float("nan"),
                float(np.mean(resid[mask])) if resid.size else float("nan"),
                float(np.mean(np.abs(resid[mask]))) if resid.size else float("nan"),
                float(np.std(resid[mask])) if resid.size else float("nan"),
                float(np.mean(ev["logit"][mask])) if y.size else float("nan"),
                float(np.mean(ev["prob"][mask])) if y.size else float("nan"),
            ])

    write_csv(
        os.path.join(exp_dir, "baseline_consistency.csv"),
        ["split", "n", "max_abs_diff_prior_vs_model_baseline", "mean_abs_diff_prior_vs_model_baseline", "model_baseline_auc", "model_baseline_ap", "prior_auc", "prior_ap"],
        consistency_rows,
    )
    write_csv(
        os.path.join(exp_dir, "model_decomposition.csv"),
        ["split", "label_subset", "n", "mean_prior_z", "mean_model_baseline_component", "mean_model_baseline_raw", "mean_model_residual_component", "mean_abs_model_residual_component", "std_model_residual_component", "mean_model_logit", "mean_model_prob"],
        decomposition_rows,
    )
    return summary

def make_per_motif_test_metrics(eval_dict: Dict, baseline_scores: np.ndarray, min_pos_for_report: int = 10) -> List[List]:
    """Filter motifs with n_pos < min_pos_for_report (default 10) from report.
    Motifs with 1-3 positives in test produce meaningless AP/AUC."""
    rows = []
    by_mid_idx: Dict[str, List[int]] = defaultdict(list)
    for i, mid in enumerate(eval_dict["mids"]):
        by_mid_idx[mid].append(i)

    for mid in sorted(by_mid_idx):
        idx = np.asarray(by_mid_idx[mid], dtype=np.int64)
        y = np.asarray(eval_dict["y"])[idx]
        n_pos = int(np.sum(y == 1))
        # skip motifs with insufficient positives for reliable metrics
        if n_pos < min_pos_for_report:
            continue
        p = np.asarray(eval_dict["prob"])[idx]
        b = np.asarray(baseline_scores)[idx]
        auc, ap = safe_auc_ap(y, p)
        b_auc, b_ap = safe_auc_ap(y, b)
        tf_family = eval_dict["tf_families"][idx[0]]
        tf_class = eval_dict["tf_classes"][idx[0]]
        rows.append([
            mid,
            tf_family,
            tf_class,
            int(len(idx)),
            n_pos,
            int(np.sum(y == 0)),
            auc,
            ap,
            b_auc,
            b_ap,
            (float(auc) - float(b_auc)) if not math.isnan(auc) and not math.isnan(b_auc) else float("nan"),
            (float(ap) - float(b_ap)) if not math.isnan(ap) and not math.isnan(b_ap) else float("nan"),
        ])

    rows.sort(key=lambda r: (float("inf") if math.isnan(r[7]) else r[7], float("inf") if math.isnan(r[11]) else r[11], r[0]))
    ranked = []
    for rank, row in enumerate(rows, start=1):
        ranked.append([rank] + row)
    return ranked

def build_summary_row(metrics: Dict) -> List:
    _dab = metrics.get("delta_ap_bootstrap", {})
    return [
        metrics["experiment_id"],
        metrics.get("fixed_len", ""),
        metrics["protocol"],
        metrics["group_by"],
        metrics["group_name"],
        metrics.get("model_arch", ""),
        metrics.get("trainable_params", ""),
        metrics.get("total_params", ""),
        metrics.get("residual_gate_final", ""),
        metrics["best_epoch"],
        metrics["n_train_pos"],
        metrics["n_val_pos"],
        metrics["n_test_pos"],
        metrics["n_train_total"],
        metrics["n_val_total"],
        metrics["n_test_total"],
        metrics["train_auc"],
        metrics["train_ap"],
        metrics["val_auc"],
        metrics["val_ap"],
        metrics["test_auc"],
        metrics["test_ap"],
        metrics["prior_baseline_train_auc"],
        metrics["prior_baseline_train_ap"],
        metrics["prior_baseline_val_auc"],
        metrics["prior_baseline_val_ap"],
        metrics["prior_baseline_test_auc"],
        metrics["prior_baseline_test_ap"],
        metrics["delta_test_auc_vs_prior"],
        metrics["delta_test_ap_vs_prior"],
        _dab.get("ci_lo", float("nan")),
        _dab.get("ci_hi", float("nan")),
        _dab.get("p_value", float("nan")),
        metrics["temperature_global"],
        metrics["tau_global_fdr"],
        metrics["exp_dir"],
    ]

def summary_header() -> List[str]:
    return [
        "experiment_id",
        "fixed_len",
        "protocol",
        "group_by",
        "group_name",
        "model_arch",
        "trainable_params",
        "total_params",
        "residual_gate_final",
        "best_epoch",
        "n_train_pos",
        "n_val_pos",
        "n_test_pos",
        "n_train_total",
        "n_val_total",
        "n_test_total",
        "train_auc",
        "train_ap",
        "val_auc",
        "val_ap",
        "test_auc",
        "test_ap",
        "prior_baseline_train_auc",
        "prior_baseline_train_ap",
        "prior_baseline_val_auc",
        "prior_baseline_val_ap",
        "prior_baseline_test_auc",
        "prior_baseline_test_ap",
        "delta_test_auc_vs_prior",
        "delta_test_ap_vs_prior",
        "delta_ap_ci_lo",
        "delta_ap_ci_hi",
        "delta_ap_p_value",
        "temperature_global",
        "tau_global_fdr",
        "exp_dir",
    ]

def history_complete_header() -> List[str]:
    return [
        "experiment_id",
        "fixed_len",
        "protocol",
        "group_by",
        "group_name",
        "model_arch",
        "epoch",
        "loss",
        "loss_main",
        "loss_prior",
        "loss_match",
        "train_auc",
        "train_ap",
        "val_auc",
        "val_ap",
        "test_auc",
        "test_ap",
        "prior_train_auc",
        "prior_train_ap",
        "prior_val_auc",
        "prior_val_ap",
        "prior_test_auc",
        "prior_test_ap",
        "residual_gate",
        "lr",
        "seconds",
        "is_best_epoch",
    ]

def history_rows_with_meta(metrics: Dict, history: List[Dict]) -> List[List]:
    rows = []
    for h in history:
        rows.append([
            metrics["experiment_id"],
            metrics.get("fixed_len", ""),
            metrics["protocol"],
            metrics["group_by"],
            metrics["group_name"],
            metrics.get("model_arch", ""),
            h["epoch"],
            h["loss"],
            h["loss_main"],
            h["loss_prior"],
            h["loss_match"],
            h["train_auc"],
            h["train_ap"],
            h["val_auc"],
            h["val_ap"],
            h.get("test_auc", float("nan")),
            h.get("test_ap", float("nan")),
            h.get("prior_train_auc", float("nan")),
            h.get("prior_train_ap", float("nan")),
            h.get("prior_val_auc", float("nan")),
            h.get("prior_val_ap", float("nan")),
            h.get("prior_test_auc", float("nan")),
            h.get("prior_test_ap", float("nan")),
            h.get("residual_gate", float("nan")),
            h["lr"],
            h["seconds"],
            h["is_best_epoch"],
        ])
    return rows

def write_global_history_exports(results_root: str, metrics_records: List[Dict], history_complete_rows: List[List]) -> None:
    s_header = summary_header()
    s_rows = [build_summary_row(m) for m in metrics_records]
    write_csv(os.path.join(results_root, "training_history_summary.csv"), s_header, s_rows)
    write_csv(os.path.join(results_root, "training_history_complete.csv"), history_complete_header(), history_complete_rows)

    family_rows = [build_summary_row(m) for m in metrics_records if m.get("group_by") == "family"]
    class_rows = [build_summary_row(m) for m in metrics_records if m.get("group_by") == "class"]
    family_complete = [r for r in history_complete_rows if r[3] == "family"]
    class_complete = [r for r in history_complete_rows if r[3] == "class"]

    write_csv(os.path.join(results_root, "training_history_family_summary.csv"), s_header, family_rows)
    write_csv(os.path.join(results_root, "training_history_family_complete.csv"), history_complete_header(), family_complete)
    write_csv(os.path.join(results_root, "training_history_class_summary.csv"), s_header, class_rows)
    write_csv(os.path.join(results_root, "training_history_class_complete.csv"), history_complete_header(), class_complete)

    prior_rows = []
    for m in metrics_records:
        prior_rows.append([
            m["experiment_id"],
            m.get("fixed_len", ""),
            m["protocol"],
            m["group_by"],
            m["group_name"],
            m["prior_baseline_train_auc"],
            m["prior_baseline_train_ap"],
            m["prior_baseline_val_auc"],
            m["prior_baseline_val_ap"],
            m["prior_baseline_test_auc"],
            m["prior_baseline_test_ap"],
            m["delta_test_auc_vs_prior"],
            m["delta_test_ap_vs_prior"],
        ])
    write_csv(
        os.path.join(results_root, "prior_baseline_global_summary.csv"),
        ["experiment_id", "fixed_len", "protocol", "group_by", "group_name", "baseline_train_auc", "baseline_train_ap", "baseline_val_auc", "baseline_val_ap", "baseline_test_auc", "baseline_test_ap", "delta_test_auc_vs_prior", "delta_test_ap_vs_prior"],
        prior_rows,
    )

    integrity_rows = []
    for m in metrics_records:
        s = m.get("split_integrity_summary", {})
        integrity_rows.append([
            m["experiment_id"],
            m.get("fixed_len", ""),
            m["protocol"],
            m["group_by"],
            m["group_name"],
            s.get("canon_overlap_train_val", ""),
            s.get("canon_overlap_train_test", ""),
            s.get("canon_overlap_val_test", ""),
            s.get("motif_overlap_train_val", ""),
            s.get("motif_overlap_train_test", ""),
            s.get("motif_overlap_val_test", ""),
            s.get("group_overlap_train_val", ""),
            s.get("group_overlap_train_test", ""),
            s.get("group_overlap_val_test", ""),
        ])
    write_csv(
        os.path.join(results_root, "split_integrity_summary.csv"),
        ["experiment_id", "fixed_len", "protocol", "group_by", "group_name", "canon_overlap_train_val", "canon_overlap_train_test", "canon_overlap_val_test", "motif_overlap_train_val", "motif_overlap_train_test", "motif_overlap_val_test", "group_overlap_train_val", "group_overlap_train_test", "group_overlap_val_test"],
        integrity_rows,
    )

    index_rows = []
    for m in metrics_records:
        index_rows.append([
            m["experiment_id"],
            m.get("fixed_len", ""),
            m["protocol"],
            m["group_by"],
            m["group_name"],
            m.get("model_arch", ""),
            m["exp_dir"],
            m["negative_sampler"].get("background_mode_resolved", ""),
            m["negative_sampler"].get("pool_size", ""),
            m["best_epoch"],
            m["test_auc"],
            m["test_ap"],
        ])
    write_csv(
        os.path.join(results_root, "experiment_index.csv"),
        ["experiment_id", "fixed_len", "protocol", "group_by", "group_name", "model_arch", "exp_dir", "background_mode_resolved", "negative_pool_size", "best_epoch", "test_auc", "test_ap"],
        index_rows,
    )
