from __future__ import annotations

"""Positive/negative example handling, dedup, leakage-controlled splits, datasets and hard-negative sampling."""

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

from .constants import (BASES, FIXED_LEN)
from .pwm_io import (PWMIndex, best_alignment_score, extract_fixed_len_windows, load_background_sequences_from_dir, load_fasta_sequences, parse_sites_file)
from .utils import (basic_scalar_features, canonical_seq, center_crop, encode_onehot, estimate_base_probs, fcgr_features, gc_count, kmer_hash_features, sample_random_seq, seq_to_idx, split_counts, write_csv)

def export_split_leak_audit(exp_dir: str, train_examples, val_examples, test_examples) -> None:
    """Verify zero data leakage between splits — export audit CSV.
    
    IMPORTANT: For positives, ANY overlap is a real leak and must be zero.
    For negatives, some canonical overlap is acceptable because the same 
    background sequence can legitimately serve as negative for different 
    positives across splits. We report both separately.
    """
    # Separate positives and negatives
    train_pos_canons = set(ex.canon for ex in train_examples if ex.label == 1)
    val_pos_canons = set(ex.canon for ex in val_examples if ex.label == 1)
    test_pos_canons = set(ex.canon for ex in test_examples if ex.label == 1)

    train_neg_canons = set(ex.canon for ex in train_examples if ex.label == 0)
    val_neg_canons = set(ex.canon for ex in val_examples if ex.label == 0)
    test_neg_canons = set(ex.canon for ex in test_examples if ex.label == 0)

    train_all_canons = set(ex.canon for ex in train_examples)
    val_all_canons = set(ex.canon for ex in val_examples)
    test_all_canons = set(ex.canon for ex in test_examples)

    # Critical checks: positive overlap (must be zero)
    pos_tr_va = train_pos_canons & val_pos_canons
    pos_tr_te = train_pos_canons & test_pos_canons
    pos_va_te = val_pos_canons & test_pos_canons

    # Informational: negative overlap (acceptable)
    neg_tr_va = train_neg_canons & val_neg_canons
    neg_tr_te = train_neg_canons & test_neg_canons
    neg_va_te = val_neg_canons & test_neg_canons

    # Cross-label check: train negative that is test positive (very bad)
    train_neg_is_test_pos = train_neg_canons & test_pos_canons
    train_neg_is_val_pos = train_neg_canons & val_pos_canons
    val_neg_is_test_pos = val_neg_canons & test_pos_canons

    rows = [
        ["POSITIVE_train_val_overlap", len(pos_tr_va), 0, "PASS" if len(pos_tr_va) == 0 else "FAIL"],
        ["POSITIVE_train_test_overlap", len(pos_tr_te), 0, "PASS" if len(pos_tr_te) == 0 else "FAIL"],
        ["POSITIVE_val_test_overlap", len(pos_va_te), 0, "PASS" if len(pos_va_te) == 0 else "FAIL"],
        ["negative_train_val_overlap", len(neg_tr_va), "acceptable", "INFO"],
        ["negative_train_test_overlap", len(neg_tr_te), "acceptable", "INFO"],
        ["negative_val_test_overlap", len(neg_va_te), "acceptable", "INFO"],
        ["CROSSLABEL_train_neg_is_test_pos", len(train_neg_is_test_pos), 0, "PASS" if len(train_neg_is_test_pos) == 0 else "FAIL"],
        ["CROSSLABEL_train_neg_is_val_pos", len(train_neg_is_val_pos), 0, "PASS" if len(train_neg_is_val_pos) == 0 else "FAIL"],
        ["CROSSLABEL_val_neg_is_test_pos", len(val_neg_is_test_pos), 0, "PASS" if len(val_neg_is_test_pos) == 0 else "FAIL"],
        ["train_pos_count", len(train_pos_canons), "", ""],
        ["val_pos_count", len(val_pos_canons), "", ""],
        ["test_pos_count", len(test_pos_canons), "", ""],
        ["train_neg_count", len(train_neg_canons), "", ""],
        ["val_neg_count", len(val_neg_canons), "", ""],
        ["test_neg_count", len(test_neg_canons), "", ""],
    ]
    write_csv(
        os.path.join(exp_dir, "split_leak_audit.csv"),
        ["check", "value", "expected", "status"],
        rows,
    )

    has_real_leak = (len(pos_tr_va) > 0 or len(pos_tr_te) > 0 or len(pos_va_te) > 0 or
                     len(train_neg_is_test_pos) > 0 or len(train_neg_is_val_pos) > 0 or len(val_neg_is_test_pos) > 0)

    if has_real_leak:
        print(f"  [WARNING] REAL data leakage! pos: tr∩va={len(pos_tr_va)} tr∩te={len(pos_tr_te)} va∩te={len(pos_va_te)} "
              f"| cross-label: tr_neg∩te_pos={len(train_neg_is_test_pos)} tr_neg∩va_pos={len(train_neg_is_val_pos)}")
    else:
        neg_info = f"neg_shared: tr∩va={len(neg_tr_va)} tr∩te={len(neg_tr_te)} va∩te={len(neg_va_te)}" if (len(neg_tr_va) + len(neg_tr_te) + len(neg_va_te)) > 0 else "neg_shared=0"
        print(f"  [AUDIT] Positives clean, no cross-label leak. {neg_info}")

@dataclass
class PositiveExample:
    seq: str
    canon: str
    motif_id: str
    tf_family: str
    tf_class: str


def group_value(ex: PositiveExample, group_by: str) -> str:
    if group_by == "family":
        return ex.tf_family or ""
    if group_by == "class":
        return ex.tf_class or ""
    raise ValueError(f"Unsupported group_by: {group_by}")

def load_positive_examples(
    sites_dir: str,
    pwm_index: PWMIndex,
    fixed_len: int,
) -> List[PositiveExample]:
    files = sorted(glob.glob(os.path.join(sites_dir, "*.sites")))
    if not files:
        raise FileNotFoundError(f"No *.sites files found in {sites_dir}")
    examples: List[PositiveExample] = []
    for fp in files:
        motif_id = os.path.basename(fp).split(".sites")[0]
        pwm = pwm_index.get_full(motif_id)
        if pwm is None:
            continue
        meta = pwm_index.get_meta(motif_id) or {}
        tf_family = meta.get("tf_family", "")
        tf_class = meta.get("tf_class", "")
        for seq in parse_sites_file(fp):
            if len(seq) != fixed_len:
                continue
            c = canonical_seq(seq)
            examples.append(
                PositiveExample(
                    seq=c,
                    canon=c,
                    motif_id=motif_id,
                    tf_family=tf_family,
                    tf_class=tf_class,
                )
            )
    return examples

def deduplicate_examples_global(examples: List[PositiveExample]) -> Tuple[List[PositiveExample], Dict, List[List[str]]]:
    by_canon: Dict[str, List[PositiveExample]] = defaultdict(list)
    for ex in examples:
        by_canon[ex.canon].append(ex)

    kept: List[PositiveExample] = []
    dropped_rows: List[List[str]] = []
    motif_collision_count = 0
    same_motif_dup_count = 0

    for canon, group in sorted(by_canon.items(), key=lambda kv: kv[0]):
        motif_counts = Counter(ex.motif_id for ex in group)
        ordered = sorted(
            group,
            key=lambda ex: (-motif_counts[ex.motif_id], ex.motif_id, ex.tf_family, ex.tf_class),
        )
        chosen_ref = ordered[0]
        kept.append(
            PositiveExample(
                seq=canon,
                canon=canon,
                motif_id=chosen_ref.motif_id,
                tf_family=chosen_ref.tf_family,
                tf_class=chosen_ref.tf_class,
            )
        )
        motif_set = sorted(set(ex.motif_id for ex in group))
        if len(motif_set) > 1:
            motif_collision_count += 1
        elif len(group) > 1:
            same_motif_dup_count += 1
        for ex in ordered[1:]:
            dropped_rows.append(
                [
                    canon,
                    chosen_ref.motif_id,
                    ex.motif_id,
                    chosen_ref.tf_family,
                    ex.tf_family,
                    chosen_ref.tf_class,
                    ex.tf_class,
                ]
            )

    report = {
        "n_input_examples": int(len(examples)),
        "n_kept_after_global_canonical_dedup": int(len(kept)),
        "n_dropped": int(len(examples) - len(kept)),
        "n_canonical_keys": int(len(by_canon)),
        "n_cross_motif_canonical_collisions": int(motif_collision_count),
        "n_same_motif_duplicate_keys": int(same_motif_dup_count),
    }
    return kept, report, dropped_rows

@dataclass
class RawPositiveExample:
    seq: str              # original orientation sequence from the source .sites record
    canon: str            # canonical full-length sequence for raw deduplication
    motif_id: str
    tf_family: str
    tf_class: str
    source_file: str = ""
    source_index: int = -1


def crop_raw_example_to_fixed_len(ex: RawPositiveExample, fixed_len: int) -> PositiveExample:
    crop = canonical_seq(center_crop(ex.seq, fixed_len))
    return PositiveExample(
        seq=crop,
        canon=crop,
        motif_id=ex.motif_id,
        tf_family=ex.tf_family,
        tf_class=ex.tf_class,
    )

def crop_split_bundle_to_fixed_len(split_bundle: SplitBundle, fixed_len: int) -> SplitBundle:
    train_pos = [crop_raw_example_to_fixed_len(ex, fixed_len) for ex in split_bundle.train_pos]
    val_pos = [crop_raw_example_to_fixed_len(ex, fixed_len) for ex in split_bundle.val_pos]
    test_pos = [crop_raw_example_to_fixed_len(ex, fixed_len) for ex in split_bundle.test_pos]
    meta = dict(split_bundle.meta)
    meta["cropped_fixed_len"] = int(fixed_len)
    meta["crop_mode"] = "center_left_then_canonicalize"
    return SplitBundle(train_pos=train_pos, val_pos=val_pos, test_pos=test_pos, meta=meta)

def load_raw_positive_examples_min_len(
    sites_dir: str,
    pwm_index: PWMIndex,
    min_len: int,
) -> List[RawPositiveExample]:
    files = sorted(glob.glob(os.path.join(sites_dir, "*.sites")))
    if not files:
        raise FileNotFoundError(f"No *.sites files found in {sites_dir}")
    examples: List[RawPositiveExample] = []
    for fp in files:
        motif_id = os.path.basename(fp).split(".sites")[0]
        pwm = pwm_index.get_full(motif_id)
        if pwm is None:
            continue
        meta = pwm_index.get_meta(motif_id) or {}
        tf_family = meta.get("tf_family", "")
        tf_class = meta.get("tf_class", "")
        seqs = parse_sites_file(fp)
        for si, seq in enumerate(seqs):
            seq = str(seq).upper()
            if len(seq) < min_len:
                continue
            c = canonical_seq(seq)
            examples.append(
                RawPositiveExample(
                    seq=seq,
                    canon=c,
                    motif_id=motif_id,
                    tf_family=tf_family,
                    tf_class=tf_class,
                    source_file=os.path.basename(fp),
                    source_index=int(si),
                )
            )
    return examples

def deduplicate_raw_examples_global(examples: List[RawPositiveExample]) -> Tuple[List[RawPositiveExample], Dict, List[List[str]]]:
    by_canon: Dict[str, List[RawPositiveExample]] = defaultdict(list)
    for ex in examples:
        by_canon[ex.canon].append(ex)

    kept: List[RawPositiveExample] = []
    dropped_rows: List[List[str]] = []
    motif_collision_count = 0
    same_motif_dup_count = 0

    for canon, group in sorted(by_canon.items(), key=lambda kv: kv[0]):
        motif_counts = Counter(ex.motif_id for ex in group)
        ordered = sorted(
            group,
            key=lambda ex: (-motif_counts[ex.motif_id], ex.motif_id, ex.tf_family, ex.tf_class, ex.source_file, ex.source_index),
        )
        chosen_ref = ordered[0]
        kept.append(
            RawPositiveExample(
                seq=chosen_ref.seq,
                canon=canon,
                motif_id=chosen_ref.motif_id,
                tf_family=chosen_ref.tf_family,
                tf_class=chosen_ref.tf_class,
                source_file=chosen_ref.source_file,
                source_index=chosen_ref.source_index,
            )
        )
        motif_set = sorted(set(ex.motif_id for ex in group))
        if len(motif_set) > 1:
            motif_collision_count += 1
        elif len(group) > 1:
            same_motif_dup_count += 1

        for ex in ordered[1:]:
            dropped_rows.append([
                canon,
                chosen_ref.motif_id,
                ex.motif_id,
                chosen_ref.tf_family,
                ex.tf_family,
                chosen_ref.tf_class,
                ex.tf_class,
            ])

    report = {
        "n_input": int(len(examples)),
        "n_unique_canonical": int(len(kept)),
        "n_dropped": int(len(examples) - len(kept)),
        "n_cross_motif_collisions": int(motif_collision_count),
        "n_same_motif_duplicates": int(same_motif_dup_count),
    }
    return kept, report, dropped_rows

def build_conflict_free_matched_subset_across_k(
    raw_examples: List[RawPositiveExample],
    k_list: List[int],
) -> Tuple[List[RawPositiveExample], List[List], Dict]:
    if not raw_examples:
        return [], [], {
            "k_list": list(k_list),
            "n_input": 0,
            "n_kept": 0,
            "retention_frac": 0.0,
            "matching_strategy": "greedy_conflict_free",
        }
    if not k_list:
        raise ValueError("k_list must be non-empty")

    k_list = sorted(set(int(k) for k in k_list))
    bucket_to_indices: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    site_buckets: List[List[Tuple[int, str]]] = []
    crop_maps: List[Dict[int, str]] = []

    for i, ex in enumerate(raw_examples):
        crop_map = {}
        buckets = []
        for k in k_list:
            crop = canonical_seq(center_crop(ex.seq, k))
            crop_map[k] = crop
            key = (k, crop)
            buckets.append(key)
            bucket_to_indices[key].append(i)
        crop_maps.append(crop_map)
        site_buckets.append(buckets)

    degrees = []
    conflict_bucket_counts = []
    for i in range(len(raw_examples)):
        deg = 0
        cbc = 0
        for key in site_buckets[i]:
            m = len(bucket_to_indices[key])
            if m > 1:
                deg += (m - 1)
                cbc += 1
        degrees.append(int(deg))
        conflict_bucket_counts.append(int(cbc))

    order = sorted(
        range(len(raw_examples)),
        key=lambda i: (
            degrees[i],
            raw_examples[i].motif_id,
            raw_examples[i].tf_family,
            raw_examples[i].tf_class,
            len(raw_examples[i].seq),
            raw_examples[i].canon,
            raw_examples[i].source_file,
            raw_examples[i].source_index,
        ),
    )

    removed = np.zeros((len(raw_examples),), dtype=bool)
    keep = np.zeros((len(raw_examples),), dtype=bool)
    kept_indices: List[int] = []

    for i in order:
        if removed[i]:
            continue
        keep[i] = True
        kept_indices.append(i)
        removed[i] = True
        for key in site_buckets[i]:
            for j in bucket_to_indices[key]:
                removed[j] = True

    kept_examples = [raw_examples[i] for i in kept_indices]

    rows = []
    for i, ex in enumerate(raw_examples):
        row = [
            int(keep[i]),
            len(ex.seq),
            ex.motif_id,
            ex.tf_family,
            ex.tf_class,
            ex.source_file,
            ex.source_index,
            ex.canon,
            ex.seq,
            degrees[i],
            conflict_bucket_counts[i],
        ]
        for k in k_list:
            row.append(crop_maps[i][k])
        rows.append(row)

    integrity = {}
    for k in k_list:
        crops = [canonical_seq(center_crop(ex.seq, k)) for ex in kept_examples]
        n_unique = len(set(crops))
        integrity[f"K{k}_n_examples"] = int(len(crops))
        integrity[f"K{k}_n_unique"] = int(n_unique)
        integrity[f"K{k}_n_duplicates"] = int(len(crops) - n_unique)

    summary = {
        "k_list": list(k_list),
        "n_input": int(len(raw_examples)),
        "n_kept": int(len(kept_examples)),
        "retention_frac": float(len(kept_examples) / max(1, len(raw_examples))),
        "matching_strategy": "greedy_conflict_free",
        "mean_conflict_degree_input": float(np.mean(degrees)) if degrees else 0.0,
        "median_conflict_degree_input": float(np.median(degrees)) if degrees else 0.0,
        "n_with_any_conflict": int(sum(d > 0 for d in degrees)),
        "integrity": integrity,
    }
    return kept_examples, rows, summary

@dataclass
class SplitBundle:
    train_pos: List[PositiveExample]
    val_pos: List[PositiveExample]
    test_pos: List[PositiveExample]
    meta: Dict


def _target_counts(total_weight: int, train_frac: float, val_frac: float, test_frac: float) -> np.ndarray:
    return np.asarray(
        [float(train_frac) * float(total_weight), float(val_frac) * float(total_weight), float(test_frac) * float(total_weight)],
        dtype=np.float64,
    )

def _assignment_state_from_map(assign_map: Dict[str, int], weight_map: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(3, dtype=np.float64)
    ent_counts = np.zeros(3, dtype=np.int64)
    for name, split_idx in assign_map.items():
        counts[int(split_idx)] += float(weight_map[name])
        ent_counts[int(split_idx)] += 1
    return counts, ent_counts

def _assignment_objective(
    counts: np.ndarray,
    ent_counts: np.ndarray,
    targets: np.ndarray,
    min_entities_each: int,
    min_pos_each: int,
) -> float:
    denom = np.maximum(targets, 1.0)
    score = float(np.sum(((counts - targets) / denom) ** 2))
    if min_entities_each > 0:
        score += 10.0 * float(np.sum(np.maximum(0, int(min_entities_each) - ent_counts) ** 2))
    if min_pos_each > 0:
        score += 100.0 * float(np.sum(np.maximum(0.0, float(min_pos_each) - counts) ** 2) / max(1.0, float(np.sum(targets))))
    return score

def _is_valid_assignment(
    counts: np.ndarray,
    ent_counts: np.ndarray,
    min_entities_each: int,
    min_pos_each: int,
) -> bool:
    if np.any(ent_counts < int(min_entities_each)):
        return False
    if np.any(counts < float(min_pos_each)):
        return False
    return True

def _local_improve_assignment(
    assign_map: Dict[str, int],
    weight_map: Dict[str, int],
    targets: np.ndarray,
    min_entities_each: int,
    min_pos_each: int,
) -> Dict[str, int]:
    names = sorted(assign_map)
    improved = True
    while improved:
        improved = False
        base_counts, base_ent = _assignment_state_from_map(assign_map, weight_map)
        base_score = _assignment_objective(base_counts, base_ent, targets, min_entities_each, min_pos_each)

        # Single-entity moves
        for name in names:
            s0 = int(assign_map[name])
            for s1 in range(3):
                if s1 == s0:
                    continue
                cand = dict(assign_map)
                cand[name] = s1
                counts, ent = _assignment_state_from_map(cand, weight_map)
                if not _is_valid_assignment(counts, ent, min_entities_each, min_pos_each):
                    continue
                score = _assignment_objective(counts, ent, targets, min_entities_each, min_pos_each)
                if score + 1e-12 < base_score:
                    assign_map = cand
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Pairwise swaps
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]
                sa = int(assign_map[a])
                sb = int(assign_map[b])
                if sa == sb:
                    continue
                cand = dict(assign_map)
                cand[a], cand[b] = sb, sa
                counts, ent = _assignment_state_from_map(cand, weight_map)
                if not _is_valid_assignment(counts, ent, min_entities_each, min_pos_each):
                    continue
                score = _assignment_objective(counts, ent, targets, min_entities_each, min_pos_each)
                if score + 1e-12 < base_score:
                    assign_map = cand
                    improved = True
                    break
            if improved:
                break
    return assign_map

def _balanced_partition_exact(
    names: List[str],
    weight_map: Dict[str, int],
    targets: np.ndarray,
    min_entities_each: int,
    min_pos_each: int,
) -> Optional[Dict[str, int]]:
    order = sorted(names, key=lambda nm: (-int(weight_map[nm]), nm))
    best = {"score": float("inf"), "assign": None}
    n = len(order)

    def rec(i: int, counts: np.ndarray, ent_counts: np.ndarray, assign: Dict[str, int]) -> None:
        remaining_entities = n - i
        need_entities = int(np.sum(np.maximum(0, int(min_entities_each) - ent_counts)))
        if need_entities > remaining_entities:
            return

        if i == n:
            if not _is_valid_assignment(counts, ent_counts, min_entities_each, min_pos_each):
                return
            score = _assignment_objective(counts, ent_counts, targets, min_entities_each, min_pos_each)
            if score < best["score"]:
                best["score"] = score
                best["assign"] = dict(assign)
            return

        name = order[i]
        w = float(weight_map[name])

        # Try more deficient bins first.
        split_order = sorted(range(3), key=lambda s: ((counts[s] - targets[s]) / max(targets[s], 1.0), ent_counts[s]))
        for s in split_order:
            ent_after = ent_counts.copy()
            ent_after[s] += 1
            remaining_after = n - i - 1
            need_after = int(np.sum(np.maximum(0, int(min_entities_each) - ent_after)))
            if need_after > remaining_after:
                continue
            counts_after = counts.copy()
            counts_after[s] += w
            assign[name] = s
            rec(i + 1, counts_after, ent_after, assign)
            del assign[name]

    rec(0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.int64), {})
    return best["assign"]

def _balanced_partition_randomized(
    names: List[str],
    weight_map: Dict[str, int],
    seed: int,
    targets: np.ndarray,
    min_entities_each: int,
    min_pos_each: int,
    n_trials: int,
) -> Optional[Dict[str, int]]:
    rng = np.random.RandomState(seed)
    n_trials = max(64, int(n_trials))
    best = {"score": float("inf"), "assign": None}
    names = list(names)

    for _ in range(n_trials):
        order = list(names)
        rng.shuffle(order)
        order.sort(key=lambda nm: (-int(weight_map[nm]), rng.rand()))
        counts = np.zeros(3, dtype=np.float64)
        ent_counts = np.zeros(3, dtype=np.int64)
        assign: Dict[str, int] = {}

        for i, name in enumerate(order):
            w = float(weight_map[name])
            remaining_after = len(order) - i - 1
            feasible: List[int] = []
            for s in range(3):
                ent_after = ent_counts.copy()
                ent_after[s] += 1
                need_after = int(np.sum(np.maximum(0, int(min_entities_each) - ent_after)))
                if need_after <= remaining_after:
                    feasible.append(s)
            if not feasible:
                feasible = [0, 1, 2]

            best_s = None
            best_s_cost = None
            for s in feasible:
                counts_after = counts.copy()
                counts_after[s] += w
                ent_after = ent_counts.copy()
                ent_after[s] += 1
                cost = _assignment_objective(counts_after, ent_after, targets, min_entities_each, min_pos_each)
                # Encourage filling deficits first.
                deficit = np.maximum(0.0, targets - counts_after)
                cost += 0.02 * float(np.sum(deficit / np.maximum(targets, 1.0)))
                if best_s_cost is None or cost < best_s_cost:
                    best_s_cost = cost
                    best_s = s

            assign[name] = int(best_s)
            counts[best_s] += w
            ent_counts[best_s] += 1

        assign = _local_improve_assignment(assign, weight_map, targets, min_entities_each, min_pos_each)
        counts, ent_counts = _assignment_state_from_map(assign, weight_map)
        if not _is_valid_assignment(counts, ent_counts, min_entities_each, min_pos_each):
            continue
        score = _assignment_objective(counts, ent_counts, targets, min_entities_each, min_pos_each)
        if score < best["score"]:
            best["score"] = score
            best["assign"] = dict(assign)

    return best["assign"]

def balanced_partition_entities_by_sites(
    entity_to_examples: Dict[str, List[PositiveExample]],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_entities_each: int = 1,
    min_pos_each: int = 1,
    n_trials: int = 1024,
) -> Optional[Tuple[set, set, set, Dict]]:
    names = sorted([name for name, items in entity_to_examples.items() if items])
    if len(names) < 3:
        return None

    weight_map = {name: int(len(entity_to_examples[name])) for name in names}
    total_weight = int(sum(weight_map.values()))
    if total_weight <= 0:
        return None
    if total_weight < 3 * int(min_pos_each):
        return None
    if len(names) < 3 * int(min_entities_each):
        return None

    targets = _target_counts(total_weight, train_frac, val_frac, test_frac)

    if len(names) <= 12:
        assign = _balanced_partition_exact(
            names=names,
            weight_map=weight_map,
            targets=targets,
            min_entities_each=min_entities_each,
            min_pos_each=min_pos_each,
        )
    else:
        assign = _balanced_partition_randomized(
            names=names,
            weight_map=weight_map,
            seed=seed,
            targets=targets,
            min_entities_each=min_entities_each,
            min_pos_each=min_pos_each,
            n_trials=n_trials,
        )

    if assign is None:
        return None

    counts, ent_counts = _assignment_state_from_map(assign, weight_map)
    if not _is_valid_assignment(counts, ent_counts, min_entities_each, min_pos_each):
        return None

    train_set = {name for name, s in assign.items() if int(s) == 0}
    val_set = {name for name, s in assign.items() if int(s) == 1}
    test_set = {name for name, s in assign.items() if int(s) == 2}

    if not train_set or not val_set or not test_set:
        return None

    meta = {
        "target_train_sites": float(targets[0]),
        "target_val_sites": float(targets[1]),
        "target_test_sites": float(targets[2]),
        "actual_train_sites": int(counts[0]),
        "actual_val_sites": int(counts[1]),
        "actual_test_sites": int(counts[2]),
        "actual_train_frac": float(counts[0] / max(1.0, float(total_weight))),
        "actual_val_frac": float(counts[1] / max(1.0, float(total_weight))),
        "actual_test_frac": float(counts[2] / max(1.0, float(total_weight))),
        "n_train_entities": int(ent_counts[0]),
        "n_val_entities": int(ent_counts[1]),
        "n_test_entities": int(ent_counts[2]),
        "entity_weight_map": {k: int(v) for k, v in sorted(weight_map.items())},
        "partition_objective": float(_assignment_objective(counts, ent_counts, targets, min_entities_each, min_pos_each)),
    }
    return train_set, val_set, test_set, meta

def split_site_level_seen_motifs(
    examples: List[PositiveExample],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_sites_per_motif: int,
    min_pos_per_split: int,
) -> Optional[SplitBundle]:
    rng = np.random.RandomState(seed)
    by_motif: Dict[str, List[PositiveExample]] = defaultdict(list)
    for ex in examples:
        by_motif[ex.motif_id].append(ex)

    train_pos: List[PositiveExample] = []
    val_pos: List[PositiveExample] = []
    test_pos: List[PositiveExample] = []
    dropped_small = {}

    for motif_id in sorted(by_motif):
        items = list(by_motif[motif_id])
        if len(items) < min_sites_per_motif:
            dropped_small[motif_id] = len(items)
            continue
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_tr, n_va, n_te = split_counts(len(items), train_frac, val_frac, test_frac, min_each=1)
        if min(n_tr, n_va, n_te) <= 0:
            dropped_small[motif_id] = len(items)
            continue
        train_pos.extend(items[i] for i in idx[:n_tr])
        val_pos.extend(items[i] for i in idx[n_tr:n_tr + n_va])
        test_pos.extend(items[i] for i in idx[n_tr + n_va:])

    if not train_pos or not val_pos or not test_pos:
        return None
    if min(len(train_pos), len(val_pos), len(test_pos)) < int(min_pos_per_split):
        return None

    total = len(train_pos) + len(val_pos) + len(test_pos)
    meta = {
        "protocol": "site_level",
        "n_motifs_after_filter": int(len(set(ex.motif_id for ex in train_pos + val_pos + test_pos))),
        "n_dropped_small_motifs": int(len(dropped_small)),
        "dropped_small_motifs": dropped_small,
        "target_train_sites": float(train_frac * total),
        "target_val_sites": float(val_frac * total),
        "target_test_sites": float(test_frac * total),
        "actual_train_sites": int(len(train_pos)),
        "actual_val_sites": int(len(val_pos)),
        "actual_test_sites": int(len(test_pos)),
        "actual_train_frac": float(len(train_pos) / max(1, total)),
        "actual_val_frac": float(len(val_pos) / max(1, total)),
        "actual_test_frac": float(len(test_pos) / max(1, total)),
        "min_pos_per_split_enforced": int(min_pos_per_split),
    }
    return SplitBundle(train_pos=train_pos, val_pos=val_pos, test_pos=test_pos, meta=meta)

def split_motif_heldout_in_group(
    examples: List[PositiveExample],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_motifs_per_group: int,
    min_pos_per_split: int,
    split_balance_trials: int,
) -> Optional[SplitBundle]:
    by_motif: Dict[str, List[PositiveExample]] = defaultdict(list)
    for ex in examples:
        by_motif[ex.motif_id].append(ex)

    motifs = sorted(by_motif)
    if len(motifs) < int(min_motifs_per_group):
        return None

    part = balanced_partition_entities_by_sites(
        entity_to_examples=by_motif,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        min_entities_each=1,
        min_pos_each=min_pos_per_split,
        n_trials=split_balance_trials,
    )
    if part is None:
        return None

    motifs_train, motifs_val, motifs_test, bal_meta = part
    train_pos = [ex for ex in examples if ex.motif_id in motifs_train]
    val_pos = [ex for ex in examples if ex.motif_id in motifs_val]
    test_pos = [ex for ex in examples if ex.motif_id in motifs_test]

    if not train_pos or not val_pos or not test_pos:
        return None
    if min(len(train_pos), len(val_pos), len(test_pos)) < int(min_pos_per_split):
        return None

    meta = {
        "protocol": "motif_heldout",
        "n_train_motifs": int(len(motifs_train)),
        "n_val_motifs": int(len(motifs_val)),
        "n_test_motifs": int(len(motifs_test)),
        "train_motifs": sorted(motifs_train),
        "val_motifs": sorted(motifs_val),
        "test_motifs": sorted(motifs_test),
        "min_pos_per_split_enforced": int(min_pos_per_split),
        "site_balanced_partition": bal_meta,
    }
    return SplitBundle(train_pos=train_pos, val_pos=val_pos, test_pos=test_pos, meta=meta)

def split_group_heldout(
    examples: List[PositiveExample],
    group_by: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_groups: int = 5,
    min_pos_per_split: int = 1,
    split_balance_trials: int = 2048,
) -> Optional[SplitBundle]:
    by_group: Dict[str, List[PositiveExample]] = defaultdict(list)
    for ex in examples:
        g = group_value(ex, group_by)
        if g:
            by_group[g].append(ex)
    groups = sorted(g for g in by_group if by_group[g])
    if len(groups) < int(min_groups):
        return None

    part = balanced_partition_entities_by_sites(
        entity_to_examples=by_group,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        min_entities_each=1,
        min_pos_each=min_pos_per_split,
        n_trials=split_balance_trials,
    )
    if part is None:
        return None

    groups_train, groups_val, groups_test, bal_meta = part
    train_pos = [ex for ex in examples if group_value(ex, group_by) in groups_train]
    val_pos = [ex for ex in examples if group_value(ex, group_by) in groups_val]
    test_pos = [ex for ex in examples if group_value(ex, group_by) in groups_test]

    if not train_pos or not val_pos or not test_pos:
        return None
    if min(len(train_pos), len(val_pos), len(test_pos)) < int(min_pos_per_split):
        return None

    meta = {
        "protocol": "group_heldout",
        "group_by": group_by,
        "train_groups": sorted(groups_train),
        "val_groups": sorted(groups_val),
        "test_groups": sorted(groups_test),
        "min_pos_per_split_enforced": int(min_pos_per_split),
        "site_balanced_partition": bal_meta,
    }
    return SplitBundle(train_pos=train_pos, val_pos=val_pos, test_pos=test_pos, meta=meta)

@dataclass
class PoolItem:
    seq: str
    canon: str
    source_label: str
    source_motif: str
    source_group: str


class CapEstimator:
    def __init__(self, q: float, pwm_index: PWMIndex, group_by: str):
        self.q = float(q)
        self.pwm_index = pwm_index
        self.group_by = group_by
        self.motif_caps: Dict[str, float] = {}
        self.group_caps: Dict[str, float] = {}
        self.global_cap: Optional[float] = None

    def fit(self, train_pos: List[PositiveExample]) -> None:
        by_motif: Dict[str, List[float]] = defaultdict(list)
        by_group: Dict[str, List[float]] = defaultdict(list)
        global_scores: List[float] = []

        for ex in train_pos:
            pwm = self.pwm_index.get_full(ex.motif_id)
            if pwm is None:
                continue
            log_pwm = np.log(np.clip(pwm, 1e-8, 1.0))
            score, _, _, _ = best_alignment_score(seq_to_idx(ex.seq), log_pwm)
            sc = float(score)
            by_motif[ex.motif_id].append(sc)
            g = group_value(ex, self.group_by)
            if g:
                by_group[g].append(sc)
            global_scores.append(sc)

        for mid, vals in by_motif.items():
            arr = np.asarray(vals, dtype=np.float32)
            self.motif_caps[mid] = float(np.quantile(arr, self.q)) if len(arr) >= 2 else float(arr.min())
        for g, vals in by_group.items():
            arr = np.asarray(vals, dtype=np.float32)
            self.group_caps[g] = float(np.quantile(arr, self.q)) if len(arr) >= 2 else float(arr.min())
        if global_scores:
            arr = np.asarray(global_scores, dtype=np.float32)
            self.global_cap = float(np.quantile(arr, self.q)) if len(arr) >= 2 else float(arr.min())
        else:
            self.global_cap = None

    def cap_for(self, motif_id: str, group_name: str) -> Optional[float]:
        if motif_id in self.motif_caps:
            return self.motif_caps[motif_id]
        if group_name in self.group_caps:
            return self.group_caps[group_name]
        return self.global_cap

class HardNegativeSampler:
    """Real-TFBS hard-negative sampler for MotifGate.

    NEGATIVE POOL (Methods 3.6): default `other_tf_train` mode draws from the globally
    deduplicated set of positive TFBS across ALL motifs/families/classes (real binding
    sites of OTHER TFs). A real site for TF-A is a plausible hard negative for TF-B.

    REAL-ONLY GUARANTEE: MotifGate never generates synthetic, shuffled, or mutated
    negatives. When the real pool cannot fill the 1:2 quota, the split is built with
    the maximum real negatives available and the realized ratio is logged.

    ANTI-LEAKAGE: candidates positive anywhere in the current group (group_pos_canons)
    are rejected, and negatives are not reused within a split.

    Legacy alias TrainOnlyNegativeSampler kept for backward compatibility.
    """

    def __init__(
        self,
        train_pos: List[PositiveExample],
        all_positive_examples: List[PositiveExample],
        pwm_index: PWMIndex,
        group_by: str,
        fixed_len: int,
        background_mode: str,
        background_fasta: Optional[str],
        background_sites_dir: Optional[str],
        background_stride: int,
        background_limit: int,
        gc_tol_frac: float,
        candidate_tries: int,
        cap_q: float,
        seed: int,
        group_pos_canons: Optional[set] = None,  # ALL canons that are positive in this group's train/val/test
    ):
        self.train_pos = train_pos
        self.all_positive_examples = all_positive_examples
        self.pwm_index = pwm_index
        self.group_by = group_by
        self.fixed_len = fixed_len
        self.gc_tol = max(1, int(round(gc_tol_frac * fixed_len)))
        self.candidate_tries = int(candidate_tries)
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed)
        self.base_probs = estimate_base_probs([ex.seq for ex in train_pos]) if train_pos else np.full(4, 0.25, dtype=np.float32)
        self.train_pos_canon = {ex.canon for ex in train_pos}
        self.all_pos_canon = {ex.canon for ex in all_positive_examples}
        # group_pos_canons = canons of ALL positives in this group (train+val+test).
        # Any negative candidate that matches a group positive would be cross-label leakage.
        self.group_pos_canons = group_pos_canons if group_pos_canons is not None else self.train_pos_canon
        # Per-motif canon sets (for motif-level filtering when group-level isn't needed)
        self.canons_by_motif: Dict[str, set] = defaultdict(set)
        for ex in all_positive_examples:
            self.canons_by_motif[ex.motif_id].add(ex.canon)
        self.background_mode_requested = background_mode
        self.background_mode_resolved = self._resolve_background_mode(background_mode, background_fasta, background_sites_dir)
        self.cap_estimator = CapEstimator(cap_q, pwm_index, group_by)
        self.cap_estimator.fit(train_pos)
        self.log_pwm_cache: Dict[str, Optional[np.ndarray]] = {}
        self.used_negative_canon_global: set = set()
        self.used_negative_canon_by_split: Dict[str, set] = defaultdict(set)
        self.stats = {
            "background_mode_requested": background_mode,
            "background_mode_resolved": self.background_mode_resolved,
            "pool_size": 0,
            "rejected_high_prior": 0,
            "rejected_same_motif_proxy": 0,
            "rejected_used_negative": 0,
            "rejected_any_positive_overlap": 0,
            "negative_shortfall": 0,
            "pool_source_counts": {},
        }
        self.pool = self._build_pool(background_fasta, background_sites_dir, background_stride, background_limit)
        self.buckets = self._build_gc_buckets(self.pool)

    def _resolve_background_mode(self, background_mode: str, background_fasta: Optional[str], background_sites_dir: Optional[str]) -> str:
        mode = background_mode.strip().lower()
        if mode == "auto":
            if background_fasta or background_sites_dir:
                return "genomic_background"
            return "other_tf_train"
        if mode == "genomic_fasta":
            return "genomic_background"
        if mode == "other_tf_train":
            return "other_tf_train"
        raise ValueError(f"Unsupported background_mode: {background_mode}")

    def _build_pool(
        self,
        background_fasta: Optional[str],
        background_sites_dir: Optional[str],
        background_stride: int,
        background_limit: int,
    ) -> List[PoolItem]:
        items: List[PoolItem] = []
        if self.background_mode_resolved == "genomic_background":
            seqs: List[str] = []
            if background_fasta:
                raw = load_fasta_sequences(background_fasta)
                seqs.extend(extract_fixed_len_windows(raw, self.fixed_len, background_stride, background_limit))
            if background_sites_dir:
                remain = max(0, background_limit - len(seqs)) if background_limit > 0 else background_limit
                seqs.extend(load_background_sequences_from_dir(background_sites_dir, self.fixed_len, background_stride, remain))
            seen = set()
            for seq in seqs:
                canon = canonical_seq(seq)
                if canon in seen:
                    continue
                seen.add(canon)
                if canon in self.all_pos_canon:
                    self.stats["rejected_any_positive_overlap"] += 1
                    continue
                items.append(
                    PoolItem(
                        seq=canon,
                        canon=canon,
                        source_label="genomic_background",
                        source_motif="__GENOMIC__",
                        source_group="__GENOMIC__",
                    )
                )
        else:
            # Use ALL positive examples (cross-family) as negative pool,
            # not just train_pos from the current group. This gives a much larger
            # and more diverse pool of biologically real sequences as negatives.
            # A binding site for TF-A in family-X is a legitimate hard negative
            # for TF-B in family-Y.
            seen = set()
            for ex in self.all_positive_examples:
                canon = ex.canon
                if canon in seen:
                    continue
                seen.add(canon)
                items.append(
                    PoolItem(
                        seq=ex.seq,
                        canon=canon,
                        source_label="other_tf_train",
                        source_motif=ex.motif_id,
                        source_group=group_value(ex, self.group_by),
                    )
                )

        self.stats["pool_size"] = int(len(items))
        self.stats["pool_source_counts"] = dict(Counter(x.source_label for x in items))
        return items

    def _build_gc_buckets(self, pool: List[PoolItem]) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, item in enumerate(pool):
            buckets[gc_count(item.seq)].append(i)
        return buckets

    def _get_log_pwm(self, motif_id: str) -> Optional[np.ndarray]:
        if motif_id not in self.log_pwm_cache:
            pwm = self.pwm_index.get_full(motif_id)
            self.log_pwm_cache[motif_id] = None if pwm is None else np.log(np.clip(pwm, 1e-8, 1.0))
        return self.log_pwm_cache[motif_id]

    def _is_forbidden_canon(self, canon: str, target_motif_id: str = "") -> bool:
        """Reject if canon is a positive of ANY motif in the current group.
        This prevents cross-label leakage where a train negative is a test positive."""
        return canon in self.group_pos_canons

    def _draw_pool_candidate(self, target: PositiveExample, rng: np.random.RandomState,
                             exclude: Optional[set] = None) -> Optional[PoolItem]:
        exclude = exclude or set()
        g0 = gc_count(target.seq)
        bucket_ids = [g0]
        for d in range(1, self.gc_tol + 1):
            if g0 - d >= 0:
                bucket_ids.append(g0 - d)
            if g0 + d <= self.fixed_len:
                bucket_ids.append(g0 + d)

        for _ in range(max(16, self.candidate_tries * 2)):
            bid = bucket_ids[int(rng.randint(0, len(bucket_ids)))]
            candidates = self.buckets.get(bid, [])
            if not candidates:
                continue
            idx = candidates[int(rng.randint(0, len(candidates)))]
            item = self.pool[idx]
            # For other_tf_train mode, pool items ARE positives of other motifs.
            # Only reject if: (a) same canonical as target, or (b) same motif as target.
            # The old check "item.canon in self.all_pos_canon" rejected ALL pool items
            # because the pool IS built from positives — causing infinite fallback loops.
            if item.canon == target.canon:
                continue
            if item.canon in exclude:
                continue
            if item.source_label == "other_tf_train" and item.source_motif == target.motif_id:
                self.stats["rejected_same_motif_proxy"] += 1
                continue
            return item
        return None

    def sample_negative_for_target(self, target: PositiveExample, split_name: str, local_rng: np.random.RandomState) -> Tuple[Optional[str], str]:
        """Return one REAL TFBS hard negative for target, or (None, "none").

        MotifGate uses ONLY real TFBS sequences (binding sites of other motifs) as
        negatives. No synthetic, shuffled, or mutated sequences are ever generated.
        If the real pool cannot supply an admissible, previously-unused negative,
        this returns (None, "none") and the caller records the shortfall.
        """
        cap = self.cap_estimator.cap_for(target.motif_id, group_value(target, self.group_by))
        log_pwm = self._get_log_pwm(target.motif_id)
        used_in_split = self.used_negative_canon_by_split[split_name]

        best_seq = None
        best_src = None
        best_score = -1e18

        for _ in range(max(16, self.candidate_tries)):
            item = self._draw_pool_candidate(target, local_rng, exclude=used_in_split)
            if item is None:
                continue
            seq = item.seq
            if seq == target.canon or seq in used_in_split:
                continue
            if seq in self.group_pos_canons:
                continue
            if log_pwm is not None and cap is not None:
                sc, _, _, _ = best_alignment_score(seq_to_idx(seq), log_pwm)
                sc = float(sc)
                if sc >= cap:
                    self.stats["rejected_high_prior"] += 1
                    continue
                if sc > best_score:
                    best_score = sc
                    best_seq = seq
                    best_src = item.source_label
            else:
                best_seq = seq
                best_src = item.source_label
                break

        if best_seq is None:
            self.stats["negative_shortfall"] = int(self.stats.get("negative_shortfall", 0) + 1)
            return None, "none"

        canon = canonical_seq(best_seq)
        if canon in self.group_pos_canons:
            self.stats["negative_shortfall"] = int(self.stats.get("negative_shortfall", 0) + 1)
            return None, "none"

        self.used_negative_canon_global.add(canon)
        self.used_negative_canon_by_split[split_name].add(canon)
        self.stats[f"{split_name}_negatives"] = int(self.stats.get(f"{split_name}_negatives", 0) + 1)
        return canon, best_src

@dataclass
class LabeledExample:
    seq: str
    canon: str
    motif_id: str
    tf_family: str
    tf_class: str
    label: int
    source: str


def make_binary_split(
    positives: List[PositiveExample],
    sampler: TrainOnlyNegativeSampler,
    neg_per_pos: int,
    split_name: str,
    seed: int,
    require_exact: bool = False,
) -> Tuple[List[LabeledExample], Dict]:
    """Assemble a labeled split (positives + REAL TFBS hard negatives).

    MotifGate uses ONLY real binding sites of other motifs as negatives; it never
    generates synthetic, shuffled, or mutated sequences. Target ratio 1:neg_per_pos.
    When the real pool cannot supply enough admissible, previously-unused sequences,
    the split is built with the maximum real negatives available and the realized
    ratio is logged. require_exact (default False) is diagnostic only.
    """
    rng = np.random.RandomState(seed)
    out: List[LabeledExample] = []
    neg_stats = Counter()
    n_requested_neg = 0
    n_missing_neg = 0

    for ex in positives:
        out.append(
            LabeledExample(
                seq=ex.seq,
                canon=ex.canon,
                motif_id=ex.motif_id,
                tf_family=ex.tf_family,
                tf_class=ex.tf_class,
                label=1,
                source="positive",
            )
        )
        for _ in range(int(neg_per_pos)):
            n_requested_neg += 1
            neg_seq, src = None, "none"
            for _attempt in range(16):
                neg_seq, src = sampler.sample_negative_for_target(ex, split_name=split_name, local_rng=rng)
                if neg_seq is not None:
                    break
            if neg_seq is None:
                # No real negative available — never synthesize. Record the shortfall.
                n_missing_neg += 1
                continue

            out.append(
                LabeledExample(
                    seq=neg_seq,
                    canon=canonical_seq(neg_seq),
                    motif_id=ex.motif_id,
                    tf_family=ex.tf_family,
                    tf_class=ex.tf_class,
                    label=0,
                    source=src,
                )
            )
            neg_stats[src] += 1

    idx = np.arange(len(out))
    rng.shuffle(idx)
    out = [out[i] for i in idx]
    n_pos = int(sum(x.label == 1 for x in out))
    n_neg = int(sum(x.label == 0 for x in out))
    expected_n_neg = int(n_pos * int(neg_per_pos))
    realized_ratio = (n_neg / n_pos) if n_pos > 0 else 0.0
    quota_ok = int(n_neg == expected_n_neg)
    if not quota_ok:
        print(
            f"[neg-quota] split={split_name}: the 1:{int(neg_per_pos)} positive-negative "
            f"ratio could not be completed using only real TFBS negatives. "
            f"Built {n_neg} real negatives for {n_pos} positives "
            f"(expected {expected_n_neg}; missing {n_missing_neg}); "
            f"realized ratio 1:{realized_ratio:.2f}. "
            f"MotifGate uses ONLY real TFBS negatives (no synthetic shuffling)."
        )
        if require_exact:
            raise RuntimeError(
                f"require_exact=True but only {n_neg}/{expected_n_neg} real negatives "
                f"could be built for split={split_name}."
            )
    summary = {
        "split_name": split_name,
        "n_total": int(len(out)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "expected_n_neg": expected_n_neg,
        "exact_negative_quota_ok": quota_ok,
        "realized_pos_neg_ratio": float(realized_ratio),
        "n_missing_negatives": int(n_missing_neg),
        "negative_source": "real_tfbs_only",
        "neg_source_counts": dict(neg_stats),
    }
    return out, summary

class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

def compute_feature_stats(train_feats: np.ndarray) -> FeatureNormalizer:
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    return FeatureNormalizer(mean, std)

def make_feature_matrix(
    examples: List[LabeledExample],
    prior_z: np.ndarray,
    pwm_index: PWMIndex,
    use_prior_z: bool,
    use_kmer_fcgr: bool,
) -> np.ndarray:
    rows = []
    for ex, pz in zip(examples, prior_z):
        pwm = pwm_index.get_full(ex.motif_id)
        pwm_len = float(0 if pwm is None else pwm.shape[0])
        parts = [
            basic_scalar_features(ex.seq),
            np.asarray([pwm_len], dtype=np.float32),
        ]
        # prior_z is intentionally NOT concatenated into the input feature vector.
        # It is kept only as an external baseline and, optionally, as an auxiliary regression target.
        if use_kmer_fcgr:
            parts.append(kmer_hash_features(ex.seq, ks=(4, 5, 6), bins=64))
            parts.append(fcgr_features(ex.seq, k=4, pool=2))
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.stack(rows, axis=0)

class TFBSDataset(Dataset):
    def __init__(
        self,
        examples: List[LabeledExample],
        prior_z: np.ndarray,
        feat_mat: np.ndarray,
        feat_norm: FeatureNormalizer,
        pwm_index: PWMIndex,
    ):
        self.examples = examples
        self.prior_z = prior_z.astype(np.float32)
        self.feat_mat = feat_mat.astype(np.float32)
        self.feat_norm = feat_norm
        self.pwm_index = pwm_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        X = encode_onehot(ex.seq)
        pwm = self.pwm_index.get_full(ex.motif_id)
        if pwm is None:
            pwm = np.full((FIXED_LEN, 4), 0.25, dtype=np.float32)
        feat = self.feat_norm.apply(self.feat_mat[i])
        return {
            "X": torch.from_numpy(X),
            "seq_len": int(X.shape[1]),
            "feat": torch.from_numpy(feat),
            "pwm": torch.from_numpy(pwm.astype(np.float32)),
            "pwm_len": int(pwm.shape[0]),
            "y": torch.tensor([float(ex.label)], dtype=torch.float32),
            "prior_z": torch.tensor([float(self.prior_z[i])], dtype=torch.float32),
            "mid": ex.motif_id,
            "tf_family": ex.tf_family,
            "tf_class": ex.tf_class,
            "source": ex.source,
            "seq": ex.seq,
        }

def collate_batch(batch: List[dict]) -> dict:
    B = len(batch)
    C = batch[0]["X"].shape[0]
    Lmax = max(b["seq_len"] for b in batch)
    Pmax = max(b["pwm_len"] for b in batch)
    Fdim = batch[0]["feat"].shape[0]

    X = torch.zeros((B, C, Lmax), dtype=torch.float32)
    seq_pad = torch.ones((B, Lmax), dtype=torch.bool)
    pwm = torch.zeros((B, Pmax, 4), dtype=torch.float32)
    pwm_pad = torch.ones((B, Pmax), dtype=torch.bool)
    feat = torch.zeros((B, Fdim), dtype=torch.float32)
    y = torch.zeros((B, 1), dtype=torch.float32)
    prior_z = torch.zeros((B, 1), dtype=torch.float32)

    mids = []
    tf_families = []
    tf_classes = []
    seqs = []
    sources = []

    for i, b in enumerate(batch):
        L = b["seq_len"]
        P = b["pwm_len"]
        X[i, :, :L] = b["X"]
        seq_pad[i, :L] = False
        pwm[i, :P] = b["pwm"]
        pwm_pad[i, :P] = False
        feat[i] = b["feat"]
        y[i] = b["y"]
        prior_z[i] = b["prior_z"]
        mids.append(b["mid"])
        tf_families.append(b["tf_family"])
        tf_classes.append(b["tf_class"])
        seqs.append(b["seq"])
        sources.append(b["source"])

    return {
        "X": X,
        "seq_pad": seq_pad,
        "pwm": pwm,
        "pwm_pad": pwm_pad,
        "feat": feat,
        "y": y,
        "prior_z": prior_z,
        "mids": mids,
        "tf_families": tf_families,
        "tf_classes": tf_classes,
        "seqs": seqs,
        "sources": sources,
    }

def make_weighted_sampler(examples: List[LabeledExample]) -> WeightedRandomSampler:
    counts = Counter(ex.motif_id for ex in examples)
    weights = torch.tensor([1.0 / counts[ex.motif_id] for ex in examples], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)

def make_split_inventory_rows(examples: List[LabeledExample], split_name: str) -> Tuple[List[List], List[List]]:
    summary_counter = Counter((split_name, int(ex.label), ex.source) for ex in examples)
    motif_counter = Counter((split_name, ex.motif_id, ex.tf_family, ex.tf_class, int(ex.label), ex.source) for ex in examples)

    summary_rows = [
        [sp, lab, src, int(n)]
        for (sp, lab, src), n in sorted(summary_counter.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]
    motif_rows = [
        [sp, mid, tfam, tfcl, lab, src, int(n)]
        for (sp, mid, tfam, tfcl, lab, src), n in sorted(motif_counter.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][4], kv[0][5]))
    ]
    return summary_rows, motif_rows

def audit_split_integrity(split_bundle: SplitBundle, protocol: str, group_by: str) -> Tuple[List[List], Dict]:
    def set_of(items: List[PositiveExample], key_fn):
        return {key_fn(x) for x in items if key_fn(x) is not None}

    train_c = set_of(split_bundle.train_pos, lambda x: x.canon)
    val_c = set_of(split_bundle.val_pos, lambda x: x.canon)
    test_c = set_of(split_bundle.test_pos, lambda x: x.canon)

    train_m = set_of(split_bundle.train_pos, lambda x: x.motif_id)
    val_m = set_of(split_bundle.val_pos, lambda x: x.motif_id)
    test_m = set_of(split_bundle.test_pos, lambda x: x.motif_id)

    train_g = set_of(split_bundle.train_pos, lambda x: group_value(x, group_by))
    val_g = set_of(split_bundle.val_pos, lambda x: group_value(x, group_by))
    test_g = set_of(split_bundle.test_pos, lambda x: group_value(x, group_by))

    checks = [
        ("canon_overlap_train_val", len(train_c & val_c), True, "canonical sequences shared across train/val"),
        ("canon_overlap_train_test", len(train_c & test_c), True, "canonical sequences shared across train/test"),
        ("canon_overlap_val_test", len(val_c & test_c), True, "canonical sequences shared across val/test"),
        ("motif_overlap_train_val", len(train_m & val_m), protocol != "site_level", "motif IDs shared across train/val"),
        ("motif_overlap_train_test", len(train_m & test_m), protocol != "site_level", "motif IDs shared across train/test"),
        ("motif_overlap_val_test", len(val_m & test_m), protocol != "site_level", "motif IDs shared across val/test"),
        ("group_overlap_train_val", len(train_g & val_g), protocol == "group_heldout", f"{group_by} labels shared across train/val"),
        ("group_overlap_train_test", len(train_g & test_g), protocol == "group_heldout", f"{group_by} labels shared across train/test"),
        ("group_overlap_val_test", len(val_g & test_g), protocol == "group_heldout", f"{group_by} labels shared across val/test"),
    ]

    rows = []
    summary = {}
    for name, value, expected_zero, note in checks:
        status = 1 if ((value == 0) if expected_zero else True) else 0
        rows.append([name, int(value), int(expected_zero), int(status), note])
        summary[name] = int(value)
        summary[f"{name}_expected_zero"] = int(expected_zero)
        summary[f"{name}_status"] = int(status)
    return rows, summary

def examples_to_manifest_rows(examples: List[LabeledExample], split_name: str) -> List[List]:
    rows = []
    for ex in examples:
        rows.append([
            split_name,
            ex.seq,
            ex.canon,
            ex.motif_id,
            ex.tf_family,
            ex.tf_class,
            int(ex.label),
            ex.source,
        ])
    return rows


# Backward-compatibility alias (see HardNegativeSampler docstring).
TrainOnlyNegativeSampler = HardNegativeSampler
