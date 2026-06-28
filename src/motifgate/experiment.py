from __future__ import annotations

"""Single-experiment orchestration (run_single_experiment)."""

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

from .constants import (LABEL_SMOOTHING, RESIDUAL_GATE_INIT)
from .data import (SplitBundle, TrainOnlyNegativeSampler, audit_split_integrity, collate_batch, examples_to_manifest_rows, export_split_leak_audit, make_binary_split, make_split_inventory_rows, make_weighted_sampler)
from .exports import (export_architecture_summary, export_baseline_consistency_and_decomposition, export_worst_k_detailed, make_per_motif_test_metrics, prior_baseline_rows_for_split)
from .interpret import (export_calibration, export_ig_and_consistency, export_predictions_csv, export_pwm_reconstruction)
from .model import (build_model)
from .pwm_io import (PWMIndex)
from .train import (bootstrap_delta_ap, evaluate, make_datasets_and_stats, train_model)
from .utils import (count_parameters, count_trainable_parameters, ensure_dir, safe_name, safe_sigmoid_np, write_csv, write_json)

def run_single_experiment(
    experiment_id: str,
    protocol: str,
    group_by: str,
    group_name: str,
    split_bundle: SplitBundle,
    pwm_index: PWMIndex,
    args,
    results_root: str,
    seed: int,
    global_positive_examples: Optional[List] = None,  # ALL dedup examples for cross-family pool
) -> Tuple[Dict, List[Dict]]:
    experiment_uid = f"{experiment_id}__{safe_name(getattr(args, 'model_arch', 'clean_hybrid'))}"
    exp_dir = os.path.join(results_root, safe_name(experiment_uid))
    ensure_dir(exp_dir)

    integrity_rows, integrity_summary = audit_split_integrity(split_bundle, protocol=protocol, group_by=group_by)
    write_csv(
        os.path.join(exp_dir, "split_integrity_audit.csv"),
        ["check_name", "value", "expected_zero", "status", "notes"],
        integrity_rows,
    )

    # Use global positive examples (all families) for cross-family negative pool.
    _all_pos_for_pool = global_positive_examples if global_positive_examples else (
        split_bundle.train_pos + split_bundle.val_pos + split_bundle.test_pos
    )

    # group_pos_canons = ALL canonical sequences that are positive in this group.
    # Any negative matching these would be cross-label leakage.
    _group_pos_canons = {ex.canon for ex in split_bundle.train_pos + split_bundle.val_pos + split_bundle.test_pos}

    sampler = TrainOnlyNegativeSampler(
        train_pos=split_bundle.train_pos,
        all_positive_examples=_all_pos_for_pool,
        pwm_index=pwm_index,
        group_by=group_by,
        fixed_len=args.fixed_len,
        background_mode=args.background_mode,
        background_fasta=args.background_fasta,
        background_sites_dir=args.background_sites_dir,
        background_stride=args.background_stride,
        background_limit=args.background_limit,
        gc_tol_frac=args.neg_gc_tol,
        candidate_tries=args.neg_candidate_tries,
        cap_q=args.neg_cap_q,
        seed=seed + 17,
        group_pos_canons=_group_pos_canons,
    )

    train_bin, train_neg_summary = make_binary_split(split_bundle.train_pos, sampler, args.neg_per_pos, "train", seed + 101)
    val_bin, val_neg_summary = make_binary_split(split_bundle.val_pos, sampler, args.neg_per_pos, "val", seed + 202)
    test_bin, test_neg_summary = make_binary_split(split_bundle.test_pos, sampler, args.neg_per_pos, "test", seed + 303)

    manifest_rows = []
    manifest_rows.extend(examples_to_manifest_rows(train_bin, "train"))
    manifest_rows.extend(examples_to_manifest_rows(val_bin, "val"))
    manifest_rows.extend(examples_to_manifest_rows(test_bin, "test"))
    write_csv(
        os.path.join(exp_dir, "split_manifest.csv"),
        ["split", "seq", "canonical_seq", "motif_id", "tf_family", "tf_class", "label", "source"],
        manifest_rows,
    )

    inv_summary_rows = []
    inv_motif_rows = []
    for split_name, split_examples in [("train", train_bin), ("val", val_bin), ("test", test_bin)]:
        s_rows, m_rows = make_split_inventory_rows(split_examples, split_name)
        inv_summary_rows.extend(s_rows)
        inv_motif_rows.extend(m_rows)

    write_csv(
        os.path.join(exp_dir, "split_inventory.csv"),
        ["split", "label", "source", "n"],
        inv_summary_rows,
    )
    write_csv(
        os.path.join(exp_dir, "split_inventory_by_motif.csv"),
        ["split", "motif_id", "tf_family", "tf_class", "label", "source", "n"],
        inv_motif_rows,
    )

    total_pos = float(len(split_bundle.train_pos) + len(split_bundle.val_pos) + len(split_bundle.test_pos))
    split_balance_meta = split_bundle.meta.get("site_balanced_partition", {})
    split_balance_rows = [
        ["train", float(args.train_frac), float(split_balance_meta.get("target_train_sites", args.train_frac * total_pos)), int(len(split_bundle.train_pos)), float(len(split_bundle.train_pos) / max(1.0, total_pos))],
        ["val", float(args.val_frac), float(split_balance_meta.get("target_val_sites", args.val_frac * total_pos)), int(len(split_bundle.val_pos)), float(len(split_bundle.val_pos) / max(1.0, total_pos))],
        ["test", float(args.test_frac), float(split_balance_meta.get("target_test_sites", args.test_frac * total_pos)), int(len(split_bundle.test_pos)), float(len(split_bundle.test_pos) / max(1.0, total_pos))],
    ]
    write_csv(
        os.path.join(exp_dir, "split_balance.csv"),
        ["split", "target_frac", "target_pos_sites", "actual_pos_sites", "actual_frac"],
        split_balance_rows,
    )
    negative_audit_rows = [
        [train_neg_summary["split_name"], train_neg_summary["n_pos"], train_neg_summary["expected_n_neg"], train_neg_summary["n_neg"], train_neg_summary["exact_negative_quota_ok"], train_neg_summary["n_negative_sampling_failures"], json.dumps(train_neg_summary["neg_source_counts"], ensure_ascii=False)],
        [val_neg_summary["split_name"], val_neg_summary["n_pos"], val_neg_summary["expected_n_neg"], val_neg_summary["n_neg"], val_neg_summary["exact_negative_quota_ok"], val_neg_summary["n_negative_sampling_failures"], json.dumps(val_neg_summary["neg_source_counts"], ensure_ascii=False)],
        [test_neg_summary["split_name"], test_neg_summary["n_pos"], test_neg_summary["expected_n_neg"], test_neg_summary["n_neg"], test_neg_summary["exact_negative_quota_ok"], test_neg_summary["n_negative_sampling_failures"], json.dumps(test_neg_summary["neg_source_counts"], ensure_ascii=False)],
    ]
    write_csv(
        os.path.join(exp_dir, "negative_sampling_audit.csv"),
        ["split", "n_pos", "expected_n_neg", "actual_n_neg", "exact_negative_quota_ok", "n_negative_sampling_failures", "neg_source_counts_json"],
        negative_audit_rows,
    )

    ds_train, ds_val, ds_test, dataset_stats = make_datasets_and_stats(
        train_bin,
        val_bin,
        test_bin,
        pwm_index=pwm_index,
        use_prior_z=bool(args.use_prior_z),
        use_kmer_fcgr=bool(args.use_kmer_fcgr),
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(ds_train.examples),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    train_eval_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(args.device)
    model = build_model(
        args=args,
        feat_dim=dataset_stats["feature_dim"],
        prior_mu=dataset_stats["prior_mu_train"],
        prior_sd=dataset_stats["prior_sd_train"],
    ).to(device)

    # Export architecture summary CSV + print
    export_architecture_summary(
        model, args, exp_dir,
        feat_dim=dataset_stats["feature_dim"],
        prior_mu=dataset_stats["prior_mu_train"],
        prior_sd=dataset_stats["prior_sd_train"],
    )

    # Verify zero data leakage
    export_split_leak_audit(exp_dir, train_bin, val_bin, test_bin)

    print("=" * 100)
    print(
        f"[EXP] {experiment_uid} | protocol={protocol} group_by={group_by} group_name={group_name} "
        f"| pos train/val/test={len(split_bundle.train_pos)}/{len(split_bundle.val_pos)}/{len(split_bundle.test_pos)} "
        f"| total train/val/test={len(ds_train)}/{len(ds_val)}/{len(ds_test)} "
        f"| background={sampler.background_mode_resolved} "
        f"| arch={getattr(model, 'arch_name', 'unknown')} trainable_params={count_trainable_parameters(model)}"
    )

    model, history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_aux_losses=bool(args.use_aux_losses),
        use_pwm=bool(args.use_pwm),
        use_prior_z=bool(args.use_prior_z),
        lambda_prior=args.lambda_prior,
        lambda_match=args.lambda_match,
        out_dir=exp_dir,
        label_smoothing=float(getattr(args, 'label_smoothing', LABEL_SMOOTHING)),
        rc_augmentation=bool(getattr(args, 'rc_augmentation', 1)),
        use_cosine_warmup=bool(getattr(args, 'use_cosine_warmup', 0)),
        test_loader=test_loader,
    )

    train_eval = evaluate(model, train_eval_loader, device)
    val_eval = evaluate(model, val_loader, device)
    test_eval = evaluate(model, test_loader, device)

    baseline_consistency_summary = export_baseline_consistency_and_decomposition(
        exp_dir=exp_dir,
        split_evals=[("train", train_eval), ("val", val_eval), ("test", test_eval)],
    )

    T_global, tau_global, test_prob_cal = export_calibration(
        exp_dir=exp_dir,
        val_eval=val_eval,
        test_eval=test_eval,
        target_fdr=args.calibration_target_fdr,
        min_cluster_n=args.calibration_min_cluster_n,
    )

    export_predictions_csv(
        os.path.join(exp_dir, "train_predictions.csv"),
        train_eval,
        split_name="train",
        baseline_scores=train_eval["prior_z"],
    )
    export_predictions_csv(
        os.path.join(exp_dir, "val_predictions.csv"),
        val_eval,
        split_name="val",
        baseline_scores=val_eval["prior_z"],
    )
    export_predictions_csv(
        os.path.join(exp_dir, "test_predictions.csv"),
        test_eval,
        split_name="test",
        baseline_scores=test_eval["prior_z"],
    )
    export_predictions_csv(
        os.path.join(exp_dir, "test_predictions_calibrated.csv"),
        test_eval,
        split_name="test",
        calibrated_prob=test_prob_cal,
        tau=tau_global,
        baseline_scores=test_eval["prior_z"],
    )

    prior_baseline_rows = [
        prior_baseline_rows_for_split("train", train_eval, train_eval["auc"], train_eval["ap"]),
        prior_baseline_rows_for_split("val", val_eval, val_eval["auc"], val_eval["ap"]),
        prior_baseline_rows_for_split("test", test_eval, test_eval["auc"], test_eval["ap"]),
    ]
    write_csv(
        os.path.join(exp_dir, "prior_baseline.csv"),
        ["split", "n_total", "n_pos", "n_neg", "baseline_auc", "baseline_ap", "model_auc", "model_ap", "delta_auc_model_minus_baseline", "delta_ap_model_minus_baseline"],
        prior_baseline_rows,
    )

    # Bootstrap confidence interval for Δ(AP)
    _boot_n = int(getattr(args, "bootstrap_n", 2000))
    test_prior_prob = safe_sigmoid_np(test_eval["prior_z"])
    delta_ap_result = bootstrap_delta_ap(
        test_eval["y"], test_eval["prob"], test_prior_prob,
        n_boot=_boot_n, seed=args.seed + 777,
    )
    write_csv(
        os.path.join(exp_dir, "delta_ap_bootstrap.csv"),
        ["metric", "value"],
        [
            ["delta_ap", delta_ap_result["delta_ap"]],
            ["ap_model", delta_ap_result["ap_model"]],
            ["ap_prior", delta_ap_result["ap_prior"]],
            ["ci_lo_2.5", delta_ap_result["ci_lo"]],
            ["ci_hi_97.5", delta_ap_result["ci_hi"]],
            ["p_value_one_sided", delta_ap_result["p_value"]],
            ["n_bootstrap", _boot_n],
        ],
    )
    print(f"  [Δ(AP)] model={delta_ap_result['ap_model']:.4f} prior={delta_ap_result['ap_prior']:.4f} "
          f"Δ={delta_ap_result['delta_ap']:.4f} CI=[{delta_ap_result['ci_lo']:.4f}, {delta_ap_result['ci_hi']:.4f}] "
          f"p={delta_ap_result['p_value']:.4f}")

    # Export residual_gate value
    if hasattr(model, "residual_gate"):
        gate_final = float(model.residual_gate.detach().cpu().item())
        write_csv(
            os.path.join(exp_dir, "residual_gate.csv"),
            ["metric", "value"],
            [["residual_gate_final", gate_final], ["residual_gate_init", float(getattr(args, "residual_gate_init", RESIDUAL_GATE_INIT))]],
        )
        print(f"  [gate] residual_gate = {gate_final:.4f}")

    per_motif_rows = make_per_motif_test_metrics(test_eval, baseline_scores=np.asarray(test_eval["prior_z"], dtype=np.float64))
    write_csv(
        os.path.join(exp_dir, "per_motif_test_metrics.csv"),
        ["rank_test_ap", "motif_id", "tf_family", "tf_class", "n_total", "n_pos", "n_neg", "test_auc", "test_ap", "baseline_auc", "baseline_ap", "delta_auc_vs_baseline", "delta_ap_vs_baseline"],
        per_motif_rows,
    )
    worst_rows = per_motif_rows[:max(1, int(args.worst_k))]
    write_csv(
        os.path.join(exp_dir, "worst_k_motifs_test.csv"),
        ["rank_test_ap", "motif_id", "tf_family", "tf_class", "n_total", "n_pos", "n_neg", "test_auc", "test_ap", "baseline_auc", "baseline_ap", "delta_auc_vs_baseline", "delta_ap_vs_baseline"],
        worst_rows,
    )
    # Enhanced worst-K with verdict and data regime
    export_worst_k_detailed(exp_dir, per_motif_rows, int(args.worst_k), history)
    write_csv(
        os.path.join(exp_dir, "test_metrics_by_motif.csv"),
        ["rank_test_ap", "motif_id", "tf_family", "tf_class", "n_total", "n_pos", "n_neg", "test_auc", "test_ap", "baseline_auc", "baseline_ap", "delta_auc_vs_baseline", "delta_ap_vs_baseline"],
        per_motif_rows,
    )

    pwm_summary_n, pwm_matrix_n = export_pwm_reconstruction(
        exp_dir=exp_dir,
        model=model,
        test_eval=test_eval,
        pwm_index=pwm_index,
        feat_norm=ds_train.feat_norm,
        prior_mu=dataset_stats["prior_mu_train"],
        prior_sd=dataset_stats["prior_sd_train"],
        use_prior_z=bool(args.use_prior_z),
        use_kmer_fcgr=bool(args.use_kmer_fcgr),
        device=device,
        fixed_len=args.fixed_len,
        max_samples_per_motif=args.pwm_recon_max_samples_per_motif,
        pred_batch_size=max(64, args.batch_size),
    )

    if int(args.export_ig) == 1:
        ig_attr_n, ig_cons_n = export_ig_and_consistency(
            exp_dir=exp_dir,
            model=model,
            test_eval=test_eval,
            pwm_index=pwm_index,
            feat_norm=ds_train.feat_norm,
            prior_mu=dataset_stats["prior_mu_train"],
            prior_sd=dataset_stats["prior_sd_train"],
            use_prior_z=bool(args.use_prior_z),
            use_kmer_fcgr=bool(args.use_kmer_fcgr),
            device=device,
            ig_steps=args.ig_steps,
            ig_max_samples_per_motif=args.ig_max_samples_per_motif,
        )
    else:
        write_csv(
            os.path.join(exp_dir, "ig_attributions.csv"),
            ["sample_id", "motif_id", "tf_family", "tf_class", "label", "seq", "position", "base", "onehot", "ig", "ig_abs", "logit", "prob"],
            [],
        )
        write_csv(
            os.path.join(exp_dir, "ig_pwm_consistency.csv"),
            ["sample_id", "motif_id", "tf_family", "tf_class", "label", "seq", "logit", "prob", "spearman_rho", "pearson_r", "mean_abs_ig", "max_abs_ig", "mean_ic", "max_ic", "alignment_score", "reverse_complemented", "mode_pwm_on_seq", "offset_seq", "offset_pwm", "pwm_length"],
            [],
        )
        ig_attr_n, ig_cons_n = 0, 0

    _hist_header = [
        "epoch", "loss", "loss_main", "loss_prior", "loss_match",
        "train_auc", "train_ap", "val_auc", "val_ap", "test_auc", "test_ap",
        "prior_train_auc", "prior_train_ap", "prior_val_auc", "prior_val_ap",
        "prior_test_auc", "prior_test_ap",
        "residual_gate", "lr", "seconds", "is_best_epoch",
    ]
    def _hist_row(h):
        return [
            h["epoch"], h["loss"], h["loss_main"], h["loss_prior"], h["loss_match"],
            h["train_auc"], h["train_ap"], h["val_auc"], h["val_ap"],
            h.get("test_auc", float("nan")), h.get("test_ap", float("nan")),
            h.get("prior_train_auc", float("nan")), h.get("prior_train_ap", float("nan")),
            h.get("prior_val_auc", float("nan")), h.get("prior_val_ap", float("nan")),
            h.get("prior_test_auc", float("nan")), h.get("prior_test_ap", float("nan")),
            h.get("residual_gate", float("nan")), h["lr"], h["seconds"], h["is_best_epoch"],
        ]

    write_csv(
        os.path.join(exp_dir, "training_history_group.csv"),
        _hist_header,
        [_hist_row(h) for h in history],
    )
    write_csv(
        os.path.join(exp_dir, "training_history.csv"),
        _hist_header,
        [_hist_row(h) for h in history],
    )

    baseline_train_auc = float(prior_baseline_rows[0][4])
    baseline_train_ap = float(prior_baseline_rows[0][5])
    baseline_val_auc = float(prior_baseline_rows[1][4])
    baseline_val_ap = float(prior_baseline_rows[1][5])
    baseline_test_auc = float(prior_baseline_rows[2][4])
    baseline_test_ap = float(prior_baseline_rows[2][5])

    residual_gate_final = float(model.residual_gate.detach().cpu().item()) if hasattr(model, "residual_gate") else 0.0
    metrics = {
        "experiment_id": experiment_uid,
        "fixed_len": int(args.fixed_len),
        "protocol": protocol,
        "group_by": group_by,
        "group_name": group_name,
        "model_arch": str(getattr(model, "arch_name", getattr(args, "model_arch", "unknown"))),
        "trainable_params": int(count_trainable_parameters(model)),
        "total_params": int(count_parameters(model)),
        "residual_gate_final": residual_gate_final,
        "exp_dir": exp_dir,
        "best_epoch": int(best_epoch),
        "train_auc": float(train_eval["auc"]),
        "train_ap": float(train_eval["ap"]),
        "val_auc": float(val_eval["auc"]),
        "val_ap": float(val_eval["ap"]),
        "test_auc": float(test_eval["auc"]),
        "test_ap": float(test_eval["ap"]),
        "prior_baseline_train_auc": baseline_train_auc,
        "prior_baseline_train_ap": baseline_train_ap,
        "prior_baseline_val_auc": baseline_val_auc,
        "prior_baseline_val_ap": baseline_val_ap,
        "prior_baseline_test_auc": baseline_test_auc,
        "prior_baseline_test_ap": baseline_test_ap,
        "delta_test_auc_vs_prior": (float(test_eval["auc"]) - baseline_test_auc) if not math.isnan(test_eval["auc"]) and not math.isnan(baseline_test_auc) else float("nan"),
        "delta_test_ap_vs_prior": (float(test_eval["ap"]) - baseline_test_ap) if not math.isnan(test_eval["ap"]) and not math.isnan(baseline_test_ap) else float("nan"),
        "temperature_global": float(T_global),
        "tau_global_fdr": float(tau_global),
        "n_train_pos": int(sum(ex.label == 1 for ex in train_bin)),
        "n_val_pos": int(sum(ex.label == 1 for ex in val_bin)),
        "n_test_pos": int(sum(ex.label == 1 for ex in test_bin)),
        "n_train_total": int(len(train_bin)),
        "n_val_total": int(len(val_bin)),
        "n_test_total": int(len(test_bin)),
        "dataset_stats": dataset_stats,
        "negative_sampler": sampler.stats,
        "negative_summaries": {
            "train": train_neg_summary,
            "val": val_neg_summary,
            "test": test_neg_summary,
        },
        "split_meta": split_bundle.meta,
        "split_integrity_summary": integrity_summary,
        "baseline_consistency_summary": baseline_consistency_summary,
        "delta_ap_bootstrap": delta_ap_result,
        "export_counts": {
            "per_motif_rows": int(len(per_motif_rows)),
            "worst_k_rows": int(len(worst_rows)),
            "pwm_reconstruction_summary_rows": int(pwm_summary_n),
            "pwm_reconstruction_matrix_rows": int(pwm_matrix_n),
            "ig_attribution_rows": int(ig_attr_n),
            "ig_consistency_rows": int(ig_cons_n),
        },
    }
    write_json(os.path.join(exp_dir, "metrics.json"), metrics)
    return metrics, history
