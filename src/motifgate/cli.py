from __future__ import annotations

"""Command-line entry point and argument parser."""

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

from .constants import (BACKGROUND_LIMIT, BACKGROUND_STRIDE, BATCH_SIZE, DEVICE, DROPOUT, D_MODEL, EPOCHS, FIXED_LEN, JASPAR_PWM_DIR, LABEL_SMOOTHING, LR, NEG_CANDIDATE_TRIES, NEG_CAP_Q, NEG_GC_TOL_FRAC, NEG_PER_POS, NUM_EXPERTS, N_HEADS, PATIENCE, RESIDUAL_GATE_INIT, SEED, SITES_DIR, TEST_FRAC, TRAIN_FRAC, TRANSFAC_DIR, VAL_FRAC, WEIGHT_DECAY)
from .data import (PositiveExample, RawPositiveExample, build_conflict_free_matched_subset_across_k, crop_split_bundle_to_fixed_len, deduplicate_examples_global, deduplicate_raw_examples_global, group_value, load_positive_examples, load_raw_positive_examples_min_len, split_group_heldout, split_motif_heldout_in_group, split_site_level_seen_motifs)
from .experiment import (run_single_experiment)
from .exports import (build_summary_row, history_rows_with_meta, summary_header, write_global_history_exports, write_k_comparison_exports, write_k_matched_exports)
from .pwm_io import (JasparPWMIndex, PWMIndex, TransfacIndex)
from .utils import (ensure_dir, parse_k_list_arg, safe_name, set_seed, stable_int, write_csv, write_json)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MotifGate: a PWM-aware residual learning model for TFBS prediction")
    p.add_argument("--wdir", default=".", help="Working directory containing Shared/")
    p.add_argument("--device", default=DEVICE, help="cpu or cuda")
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--fixed_len", type=int, default=FIXED_LEN)
    p.add_argument("--compare_k_list", default="", help="Optional comma-separated K list for fair matched-source comparison, e.g. 7,8,9")

    p.add_argument("--protocol", choices=["site_level", "motif_heldout", "group_heldout"], default="site_level")
    p.add_argument("--group_by", choices=["family", "class"], default="family")
    p.add_argument("--target_group", default="", help="Optional single family/class to run")
    p.add_argument("--max_groups", type=int, default=0, help="Optional cap on number of groups for site_level/motif_heldout")

    p.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--val_frac", type=float, default=VAL_FRAC)
    p.add_argument("--test_frac", type=float, default=TEST_FRAC)

    p.add_argument("--min_group_pos", type=int, default=15)
    p.add_argument("--min_sites_per_motif", type=int, default=3)
    p.add_argument("--min_motifs_per_group", type=int, default=1)
    p.add_argument("--min_groups_for_group_heldout", type=int, default=3)
    p.add_argument("--min_pos_per_split", type=int, default=10, help="Minimum number of POSITIVE examples required in each split; groups that cannot satisfy this are skipped/invalidated")
    p.add_argument("--split_balance_trials", type=int, default=2048, help="Randomized search trials for site-balanced motif/group held-out assignment")

    p.add_argument("--background_mode", choices=["auto", "genomic_fasta", "other_tf_train"], default="other_tf_train")
    p.add_argument("--background_fasta", default="", help="Preferred biologically plausible background FASTA")
    p.add_argument("--background_sites_dir", default="", help="Alternative background directory with FASTA/.sites files")
    p.add_argument("--background_stride", type=int, default=BACKGROUND_STRIDE)
    p.add_argument("--background_limit", type=int, default=BACKGROUND_LIMIT)

    p.add_argument("--neg_per_pos", type=int, default=NEG_PER_POS)
    p.add_argument("--neg_cap_q", type=float, default=NEG_CAP_Q)
    p.add_argument("--neg_gc_tol", type=float, default=NEG_GC_TOL_FRAC)
    p.add_argument("--neg_candidate_tries", type=int, default=NEG_CANDIDATE_TRIES)

    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--n_heads", type=int, default=N_HEADS)
    p.add_argument("--num_experts", type=int, default=NUM_EXPERTS)

    p.add_argument("--model_arch", choices=["clean_hybrid", "pwmconv_exact", "pwmconv_residual"], default="pwmconv_residual", help="clean_hybrid: legacy v31.3-style model; pwmconv_exact: exact baseline-as-model; pwmconv_residual: exact baseline + learned residual")
    p.add_argument("--residual_gate_init", type=float, default=RESIDUAL_GATE_INIT, help="Initial scalar gate for the residual branch; default 0.1")
    p.add_argument("--freeze_residual_gate", type=int, default=0, help="Ablation: if 1, keep the residual gate fixed at its init value (not trained)")

    p.add_argument("--use_pwm", type=int, default=1)
    p.add_argument("--use_prior_z", type=int, default=1, help="v32 auxiliary-only switch: 1 enables prior_z regression loss; prior_z is never used as model input")
    p.add_argument("--use_cross_attention", type=int, default=1)
    p.add_argument("--use_kmer_fcgr", type=int, default=0, help="default OFF; k-mer/FCGR uninformative at K<=9")
    p.add_argument("--use_moe", type=int, default=1)
    p.add_argument("--use_aux_losses", type=int, default=1)
    p.add_argument("--lambda_prior", type=float, default=0.10)
    p.add_argument("--lambda_match", type=float, default=0.05)

    # new flags
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING, help="label smoothing epsilon (0 to disable)")
    p.add_argument("--rc_augmentation", type=int, default=1, help="reverse-complement augmentation during training (1=on, 0=off)")
    p.add_argument("--use_se", type=int, default=1, help="Squeeze-and-Excitation block on CNN (1=on, 0=off)")
    p.add_argument("--use_cosine_warmup", type=int, default=0, help="cosine-annealing LR with linear warmup (1=on, 0=off)")
    p.add_argument("--bootstrap_n", type=int, default=2000, help="number of bootstrap resamples for delta-AP CI")

    p.add_argument("--calibration_target_fdr", type=float, default=0.10)
    p.add_argument("--calibration_min_cluster_n", type=int, default=25)
    p.add_argument("--export_ig", type=int, default=1)
    p.add_argument("--ig_steps", type=int, default=32)
    p.add_argument("--ig_max_samples_per_motif", type=int, default=32)
    p.add_argument("--pwm_recon_max_samples_per_motif", type=int, default=256)
    p.add_argument("--worst_k", type=int, default=20)

    p.add_argument("--results_root", default="", help="Optional explicit results directory")
    p.add_argument("--run_name", default="motifgate", help="Optional run-name suffix")
    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    args.device = args.device if isinstance(args.device, str) else str(args.device)

    if args.use_pwm == 0:
        args.use_cross_attention = 0
    if args.use_aux_losses == 0:
        args.lambda_prior = 0.0
        args.lambda_match = 0.0
    if args.use_prior_z == 0:
        args.lambda_prior = 0.0

    compare_k_list = parse_k_list_arg(args.compare_k_list)
    compare_mode = len(compare_k_list) >= 2

    wdir = args.wdir
    transfac_dir = os.path.join(wdir, TRANSFAC_DIR)
    sites_dir = os.path.join(wdir, SITES_DIR)
    jaspar_pwm_dir = os.path.join(wdir, JASPAR_PWM_DIR)

    ti = TransfacIndex(transfac_dir)
    ji = JasparPWMIndex(jaspar_pwm_dir)
    pwm_index = PWMIndex(ti, ji)

    print(f"[Device] {args.device}")
    print(f"[prior_z] auxiliary-only mode; used as external baseline and optional auxiliary target, never as input feature")
    print(f"[TRANSFAC] {transfac_dir}")
    print(f"[SITES] {sites_dir}")
    print(f"[PWM exact/base] transfac={len(ti.pwm_exact)}/{len(ti.pwm_base)} | jaspar={len(ji.pwm_exact)}/{len(ji.pwm_base)}")
    if compare_mode:
        print(f"[K-compare] fair matched-source mode enabled for K={compare_k_list} using shared raw sites and shared split assignments")

    results_root = args.results_root
    if not results_root:
        if compare_mode:
            suffix = f"Kcmp_{'_'.join(str(k) for k in compare_k_list)}_{args.protocol}_{args.group_by}"
        else:
            suffix = f"K{args.fixed_len}_{args.protocol}_{args.group_by}_last"
        if args.run_name:
            suffix += f"_{safe_name(args.run_name)}"
        results_root = os.path.join(wdir, "Shared", "Results", suffix)
    ensure_dir(results_root)

    config_to_write = vars(args).copy()
    config_to_write["script_version"] = "1.0.0"
    config_to_write["compare_k_list_resolved"] = compare_k_list
    config_to_write["compare_mode"] = bool(compare_mode)
    write_json(os.path.join(results_root, "config.json"), config_to_write)

    dedup_header = [
        "canonical_seq",
        "kept_motif",
        "dropped_motif",
        "kept_tf_family",
        "dropped_tf_family",
        "kept_tf_class",
        "dropped_tf_class",
    ]

    summaries: List[List] = []
    metrics_records: List[Dict] = []
    history_complete_rows: List[List] = []

    if compare_mode:
        raw_examples = load_raw_positive_examples_min_len(sites_dir, pwm_index, min_len=max(compare_k_list))
        dedup_examples, dedup_report, dedup_dropped_rows = deduplicate_raw_examples_global(raw_examples)
        write_json(os.path.join(results_root, "dedup_report.json"), dedup_report)
        write_csv(os.path.join(results_root, "dedup_dropped.csv"), dedup_header, dedup_dropped_rows)

        matched_examples, pool_rows, matched_summary = build_conflict_free_matched_subset_across_k(dedup_examples, compare_k_list)
        write_k_matched_exports(results_root, compare_k_list, pool_rows, matched_summary)

        print(f"[Raw positives eligible len>={max(compare_k_list)}] {len(raw_examples)}")
        print(f"[Raw positives dedup canonical/global] {len(dedup_examples)}")
        print(f"[Matched raw positives conflict-free across K] {len(matched_examples)}")
        print(f"[Matched pool summary] {matched_summary}")

        if not matched_examples:
            raise RuntimeError("Matched K-comparison subset is empty after conflict filtering.")

        if args.protocol in {"site_level", "motif_heldout"}:
            by_group: Dict[str, List[RawPositiveExample]] = defaultdict(list)
            for ex in matched_examples:
                g = group_value(ex, args.group_by)
                if g:
                    by_group[g].append(ex)

            group_items = [(g, exs) for g, exs in by_group.items() if len(exs) >= args.min_group_pos]
            group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

            if args.target_group:
                group_items = [(g, exs) for g, exs in group_items if g == args.target_group]
            if args.max_groups and args.max_groups > 0:
                group_items = group_items[:args.max_groups]

            print(f"[Groups selected] {len(group_items)}")

            for gi, (gname, gexamples) in enumerate(group_items, start=1):
                seed = args.seed + stable_int(f"{args.protocol}|{args.group_by}|{gname}")
                if args.protocol == "site_level":
                    split_bundle_raw = split_site_level_seen_motifs(
                        examples=gexamples,
                        seed=seed,
                        train_frac=args.train_frac,
                        val_frac=args.val_frac,
                        test_frac=args.test_frac,
                        min_sites_per_motif=args.min_sites_per_motif,
                        min_pos_per_split=args.min_pos_per_split,
                    )
                else:
                    split_bundle_raw = split_motif_heldout_in_group(
                        examples=gexamples,
                        seed=seed,
                        train_frac=args.train_frac,
                        val_frac=args.val_frac,
                        test_frac=args.test_frac,
                        min_motifs_per_group=args.min_motifs_per_group,
                        min_pos_per_split=args.min_pos_per_split,
                        split_balance_trials=args.split_balance_trials,
                    )
                if split_bundle_raw is None:
                    print(f"[skip] {gname} - could not build valid raw split")
                    continue

                for k in compare_k_list:
                    args_k = argparse.Namespace(**vars(args))
                    args_k.fixed_len = int(k)
                    split_bundle_k = crop_split_bundle_to_fixed_len(split_bundle_raw, k)
                    experiment_id = f"{args.protocol}_{args.group_by}_{safe_name(gname)}_K{k}"
                    metrics, history = run_single_experiment(
                        experiment_id=experiment_id,
                        protocol=args.protocol,
                        group_by=args.group_by,
                        group_name=gname,
                        split_bundle=split_bundle_k,
                        pwm_index=pwm_index,
                        args=args_k,
                        results_root=results_root,
                        seed=seed + 1000 * int(k),
                        global_positive_examples=matched_examples,
                    )
                    metrics["compare_mode"] = True
                    metrics["compare_k_list"] = list(compare_k_list)
                    metrics["matched_raw_source_n"] = int(len(matched_examples))
                    metrics_records.append(metrics)
                    summaries.append(build_summary_row(metrics))
                    history_complete_rows.extend(history_rows_with_meta(metrics, history))
                    write_csv(os.path.join(results_root, "summary.csv"), summary_header(), summaries)

        else:
            split_bundle_raw = split_group_heldout(
                examples=matched_examples,
                group_by=args.group_by,
                seed=args.seed + stable_int(f"group_heldout|{args.group_by}|matched"),
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                min_groups=args.min_groups_for_group_heldout,
                min_pos_per_split=args.min_pos_per_split,
                split_balance_trials=args.split_balance_trials,
            )
            if split_bundle_raw is None:
                raise RuntimeError("Could not build group-heldout split for matched K-comparison mode.")

            for k in compare_k_list:
                args_k = argparse.Namespace(**vars(args))
                args_k.fixed_len = int(k)
                split_bundle_k = crop_split_bundle_to_fixed_len(split_bundle_raw, k)
                experiment_id = f"group_heldout_{args.group_by}_K{k}"
                metrics, history = run_single_experiment(
                    experiment_id=experiment_id,
                    protocol=args.protocol,
                    group_by=args.group_by,
                    group_name="__GLOBAL__",
                    split_bundle=split_bundle_k,
                    pwm_index=pwm_index,
                    args=args_k,
                    results_root=results_root,
                    seed=args.seed + 7001 + 1000 * int(k),
                    global_positive_examples=matched_examples,
                )
                metrics["compare_mode"] = True
                metrics["compare_k_list"] = list(compare_k_list)
                metrics["matched_raw_source_n"] = int(len(matched_examples))
                metrics_records.append(metrics)
                summaries.append(build_summary_row(metrics))
                history_complete_rows.extend(history_rows_with_meta(metrics, history))
                write_csv(os.path.join(results_root, "summary.csv"), summary_header(), summaries)

    else:
        raw_examples = load_positive_examples(sites_dir, pwm_index, fixed_len=args.fixed_len)
        dedup_examples, dedup_report, dedup_dropped_rows = deduplicate_examples_global(raw_examples)

        write_json(os.path.join(results_root, "dedup_report.json"), dedup_report)
        write_csv(os.path.join(results_root, "dedup_dropped.csv"), dedup_header, dedup_dropped_rows)

        print(f"[Positives raw] {len(raw_examples)}")
        print(f"[Positives dedup canonical/global] {len(dedup_examples)}")
        print(f"[Dedup report] {dedup_report}")

        if args.protocol in {"site_level", "motif_heldout"}:
            by_group: Dict[str, List[PositiveExample]] = defaultdict(list)
            for ex in dedup_examples:
                g = group_value(ex, args.group_by)
                if g:
                    by_group[g].append(ex)

            group_items = [(g, exs) for g, exs in by_group.items() if len(exs) >= args.min_group_pos]
            group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

            if args.target_group:
                group_items = [(g, exs) for g, exs in group_items if g == args.target_group]
            if args.max_groups and args.max_groups > 0:
                group_items = group_items[:args.max_groups]

            print(f"[Groups selected] {len(group_items)}")

            for gi, (gname, gexamples) in enumerate(group_items, start=1):
                seed = args.seed + stable_int(f"{args.protocol}|{args.group_by}|{gname}")
                if args.protocol == "site_level":
                    split_bundle = split_site_level_seen_motifs(
                        examples=gexamples,
                        seed=seed,
                        train_frac=args.train_frac,
                        val_frac=args.val_frac,
                        test_frac=args.test_frac,
                        min_sites_per_motif=args.min_sites_per_motif,
                        min_pos_per_split=args.min_pos_per_split,
                    )
                else:
                    split_bundle = split_motif_heldout_in_group(
                        examples=gexamples,
                        seed=seed,
                        train_frac=args.train_frac,
                        val_frac=args.val_frac,
                        test_frac=args.test_frac,
                        min_motifs_per_group=args.min_motifs_per_group,
                        min_pos_per_split=args.min_pos_per_split,
                        split_balance_trials=args.split_balance_trials,
                    )
                if split_bundle is None:
                    print(f"[skip] {gname} - could not build valid split")
                    continue

                print(f"\n[{gi}/{len(group_items)}] Processing: {gname}  pos={len(gexamples)}")
                experiment_id = f"{args.protocol}_{args.group_by}_{safe_name(gname)}"
                metrics, history = run_single_experiment(
                    experiment_id=experiment_id,
                    protocol=args.protocol,
                    group_by=args.group_by,
                    group_name=gname,
                    split_bundle=split_bundle,
                    pwm_index=pwm_index,
                    args=args,
                    results_root=results_root,
                    seed=seed,
                    global_positive_examples=dedup_examples,
                )
                metrics_records.append(metrics)
                summaries.append(build_summary_row(metrics))
                history_complete_rows.extend(history_rows_with_meta(metrics, history))

                write_csv(os.path.join(results_root, "summary.csv"), summary_header(), summaries)

        else:
            split_bundle = split_group_heldout(
                examples=dedup_examples,
                group_by=args.group_by,
                seed=args.seed + stable_int(f"group_heldout|{args.group_by}"),
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                min_groups=args.min_groups_for_group_heldout,
                min_pos_per_split=args.min_pos_per_split,
                split_balance_trials=args.split_balance_trials,
            )
            if split_bundle is None:
                raise RuntimeError("Could not build group-heldout split.")

            experiment_id = f"group_heldout_{args.group_by}"
            metrics, history = run_single_experiment(
                experiment_id=experiment_id,
                protocol=args.protocol,
                group_by=args.group_by,
                group_name="__GLOBAL__",
                split_bundle=split_bundle,
                pwm_index=pwm_index,
                args=args,
                results_root=results_root,
                seed=args.seed + 7001,
                global_positive_examples=dedup_examples,
            )
            metrics_records.append(metrics)
            summaries.append(build_summary_row(metrics))
            history_complete_rows.extend(history_rows_with_meta(metrics, history))
            write_csv(os.path.join(results_root, "summary.csv"), summary_header(), summaries)

    write_json(os.path.join(results_root, "all_metrics.json"), metrics_records)
    write_global_history_exports(results_root, metrics_records, history_complete_rows)
    if compare_mode:
        write_k_comparison_exports(results_root, metrics_records)

    print(f"\n[OK] Results exported to: {results_root}")

if __name__ == "__main__":
    main()
