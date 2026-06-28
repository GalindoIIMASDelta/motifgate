#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
motifgate_external_unibind_validation_recursive_groupmodels.py

External validation builder for MotifGate using UniBind TFBS FASTA files.

This version supports the actual UniBind directory structure:

Homo_sapiens_TFBS_per_TF_FASTA/
├── AR/
│   ├── ERP001226....AR.MA0007.3.damo.fa
│   └── ...
├── ARNT/
│   └── ...
├── CTCF/
│   └── ...

It also supports MotifGate experiment roots with one model per class/family:

Results/K7_site_level_class_last/
├── config.json
├── summary.csv
├── training_history_class_summary.csv
├── site_level_class_AP2_EREBP__pwmconv_residual/
│   ├── best_model.pt
│   ├── split_manifest.csv
│   └── test_predictions.csv
├── site_level_class_.../
│   ├── best_model.pt
│   ├── split_manifest.csv
│   └── test_predictions.csv

Main features:
1) Recursively scans UniBind FASTA files.
2) Infers TF symbol from parent folder.
3) Extracts motif_id directly from UniBind filename, e.g. MA0007.3.
4) Builds positives from UniBind TFBS sequences.
5) Builds hard negatives from other-TF UniBind sites with GC matching.
6) Computes PWM-prior metrics when --motifgate_py and --wdir are provided.
7) Supports:
   A) one single MotifGate model via --motifgate_exp_dir
   B) recursive group-specific models via --motifgate_exp_root and --model_group_by class/family

Recommended K-specific use:
  For a K=7 class experiment:
    --k_list 7
    --motifgate_exp_root Results/K7_site_level_class_last
    --model_group_by class

  For a K=7 family experiment:
    --k_list 7
    --motifgate_exp_root Results/K7_site_level_family_last
    --model_group_by family
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from motifgate_validation_utils import (
    EXAMPLE_FIELDS,
    Example,
    attach_and_export_metrics,
    canonical,
    center_crop,
    ensure_dir,
    examples_to_rows,
    gc_count,
    load_fasta_records,
    load_mapping_csv,
    metric_ap_auc,
    pwm_raw_scores_with_base,
    summarize_metrics,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------
# Basic parsing helpers
# ---------------------------------------------------------------------

def normalize_tf_symbol(x: str) -> str:
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x)
    return x


def sanitize_group_name(x: str) -> str:
    x = str(x).strip()
    x = x.replace("/", "_")
    x = x.replace("\\", "_")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def extract_motif_id_from_unibind_filename(path: Path) -> str:
    """
    Extract JASPAR motif ID from UniBind FASTA filename.

    Example:
      EXP000407.LNCaP_prostate_carcinoma.AR.MA0007.3.damo.fa
      -> MA0007.3
    """
    m = re.search(r"(MA\d+\.\d+)", path.name)
    return m.group(1) if m else ""


def infer_tf_from_unibind_path(path: Path, root_dir: Path) -> str:
    """
    Infer TF symbol from UniBind folder structure.

    Expected:
      root_dir/AR/file.fa       -> AR
      root_dir/CTCF/file.fa     -> CTCF

    Fallback:
      parse first token from filename.
    """
    try:
        rel = path.relative_to(root_dir)
        if len(rel.parts) >= 2:
            return normalize_tf_symbol(rel.parts[0])
    except Exception:
        pass

    stem = path.name
    stem = re.sub(r"\.(fa|fasta|fas|fa\.gz|fasta\.gz|fas\.gz)$", "", stem, flags=re.I)
    token = re.split(r"[._| -]", stem)[0]
    return normalize_tf_symbol(token)


def find_unibind_fasta_files(unibind_fasta_dir: Path) -> List[Path]:
    patterns = [
        "*.fa",
        "*.fasta",
        "*.fas",
        "*.fa.gz",
        "*.fasta.gz",
        "*.fas.gz",
    ]

    fasta_files: List[Path] = []
    for pat in patterns:
        fasta_files.extend(unibind_fasta_dir.rglob(pat))

    return sorted(set(fasta_files))


def read_csv_dict_local(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# Mapping / metadata helpers
# ---------------------------------------------------------------------

def resolve_mapping(tf: str, motif_id: str, mapping: Dict[str, dict]) -> dict:
    """
    Resolve metadata from mapping CSV.

    Priority:
    1) Exact motif_id match.
    2) TF symbol match.
    3) Empty metadata.
    """
    motif_key = str(motif_id).strip().upper()
    tf_key = normalize_tf_symbol(tf)

    if motif_key and motif_key in mapping:
        return mapping[motif_key]

    if tf_key and tf_key in mapping:
        return mapping[tf_key]

    return {}


def infer_group_value_from_split_manifest(
    split_manifest: Path,
    model_group_by: str,
) -> str:
    """
    Infer which class/family a trained model directory represents.

    For class-specific models:
      model_group_by = "class"  -> field tf_class

    For family-specific models:
      model_group_by = "family" -> field tf_family
    """
    rows = read_csv_dict_local(split_manifest)

    if model_group_by == "class":
        field = "tf_class"
    elif model_group_by == "family":
        field = "tf_family"
    else:
        raise ValueError("model_group_by must be 'class' or 'family'.")

    values = sorted({
        (r.get(field) or "").strip()
        for r in rows
        if (r.get(field) or "").strip()
    })

    if len(values) == 0:
        raise RuntimeError(
            f"No group values found in {split_manifest} using field {field}"
        )

    if len(values) > 1:
        print(
            f"[warning] Multiple {field} values in {split_manifest}: {values}. "
            f"Using first value: {values[0]}"
        )

    return values[0]


def get_example_group_value(ex: Example, model_group_by: str) -> str:
    if model_group_by == "class":
        return str(ex.tf_class).strip()
    if model_group_by == "family":
        return str(ex.tf_family).strip()
    raise ValueError("model_group_by must be 'class' or 'family'.")


# ---------------------------------------------------------------------
# UniBind positive construction
# ---------------------------------------------------------------------

def build_positive_examples_from_fasta(
    fasta_files: Sequence[Path],
    unibind_root_dir: Path,
    mapping: Dict[str, dict],
    k_list: Sequence[int],
    species: str,
    max_per_file: int,
    seed: int,
) -> List[Example]:
    """
    Build positive examples from UniBind FASTA files using streaming.

    Important:
    This version does NOT load the whole FASTA into memory.
    It stops after max_per_file records per FASTA file.
    """
    rng = random.Random(seed)
    positives: List[Example] = []

    skipped_no_motif = 0
    skipped_empty = 0
    n_records_total = 0
    n_records_used = 0

    for file_i, fp in enumerate(sorted(fasta_files), start=1):
        tf_symbol = infer_tf_from_unibind_path(fp, unibind_root_dir)
        motif_id_from_file = extract_motif_id_from_unibind_filename(fp)

        meta = resolve_mapping(tf_symbol, motif_id_from_file, mapping)

        if motif_id_from_file:
            motif_id = motif_id_from_file
            motif_source = "filename"
        else:
            motif_id = (
                meta.get("motif_id")
                or meta.get("matrix_id")
                or meta.get("jaspar_id")
                or ""
            ).strip()
            motif_source = "mapping_csv"

        if not motif_id:
            print(f"[skip] No motif_id found for TF={tf_symbol} file={fp.name}")
            skipped_no_motif += 1
            continue

        tf_family = meta.get("tf_family", meta.get("family", ""))
        tf_class = meta.get("tf_class", meta.get("class", ""))
        tax_group = meta.get("tax_group", "")
        species_meta = species

        used_in_file = 0
        file_had_records = False

        for rec_i, (header, seq) in enumerate(load_fasta_records(fp)):
            file_had_records = True
            n_records_total += 1

            if max_per_file and used_in_file >= max_per_file:
                break

            if not seq:
                continue

            used_in_file += 1
            n_records_used += 1

            for k in k_list:
                crop = center_crop(seq, int(k))

                if crop is None:
                    continue

                crop = canonical(crop)

                positives.append(
                    Example(
                        example_id=f"UB_POS_{len(positives):09d}",
                        dataset="external_unibind",
                        source_db="UniBind",
                        fixed_len=int(k),
                        seq=crop,
                        canonical_seq=canonical(crop),
                        motif_id=motif_id,
                        tf_symbol=tf_symbol,
                        label=1,
                        tf_family=tf_family,
                        tf_class=tf_class,
                        species=species_meta,
                        tax_group=tax_group,
                        source_file=str(fp),
                        source_id=header,
                        source_extra=(
                            f"motif_id_source={motif_source};"
                            f"original_file={fp.name}"
                        ),
                    )
                )

        if not file_had_records:
            skipped_empty += 1

        if file_i % 250 == 0:
            print(
                f"[positive streaming] processed_files={file_i}/{len(fasta_files)} "
                f"records_used={n_records_used} positives_raw={len(positives)}"
            )

    seen = set()
    deduped: List[Example] = []

    for ex in positives:
        key = (ex.fixed_len, ex.motif_id, ex.canonical_seq)
        if key in seen:
            continue

        seen.add(key)
        ex.example_id = f"UB_POS_{len(deduped):09d}"
        deduped.append(ex)

    print("[positive construction]")
    print(f"  FASTA files scanned: {len(fasta_files)}")
    print(f"  FASTA records touched before per-file limits: {n_records_total}")
    print(f"  FASTA records used after per-file limits: {n_records_used}")
    print(f"  Positive examples before deduplication: {len(positives)}")
    print(f"  Positive examples after deduplication: {len(deduped)}")
    print(f"  Files skipped without motif_id: {skipped_no_motif}")
    print(f"  Empty FASTA files skipped: {skipped_empty}")

    return deduped


# ---------------------------------------------------------------------
# Negative construction
# ---------------------------------------------------------------------

def build_other_tf_hard_negatives(
    positives: Sequence[Example],
    neg_per_pos: int,
    gc_tol_bases: int,
    seed: int,
    pwm_scores: np.ndarray | None = None,
    pwm_cap_quantile: float = 0.10,
) -> tuple[List[Example], List[dict]]:
    """
    Build other-TF hard negatives efficiently.

    Optimized version:
    - Index positives by fixed_len and GC count.
    - For each target, only searches nearby GC buckets.
    - Avoids O(N^2) scans.
    """
    rng = random.Random(seed)

    pos_canons = set(
        (ex.fixed_len, ex.motif_id, ex.canonical_seq)
        for ex in positives
    )

    # Index candidates by K and GC count.
    by_k_gc: dict[tuple[int, int], list[Example]] = {}

    for ex in positives:
        key = (int(ex.fixed_len), int(gc_count(ex.seq)))
        by_k_gc.setdefault(key, []).append(ex)

    negatives: List[Example] = []
    audit = []
    used = set()

    n_targets = len(positives)

    for pos_i, target in enumerate(positives):
        target_gc = gc_count(target.seq)

        candidate_pool = []

        for gc_val in range(target_gc - gc_tol_bases, target_gc + gc_tol_bases + 1):
            candidate_pool.extend(by_k_gc.get((target.fixed_len, gc_val), []))

        if candidate_pool:
            rng.shuffle(candidate_pool)

        n_pool = 0
        n_added = 0

        for cand in candidate_pool:
            if n_added >= neg_per_pos:
                break

            if cand.motif_id == target.motif_id:
                continue

            if cand.canonical_seq == target.canonical_seq:
                continue

            n_pool += 1

            key = (target.fixed_len, target.motif_id, cand.canonical_seq)

            if key in used or key in pos_canons:
                continue

            used.add(key)

            negatives.append(
                Example(
                    example_id=f"UB_NEG_{len(negatives):09d}",
                    dataset="external_unibind",
                    source_db="UniBind",
                    fixed_len=target.fixed_len,
                    seq=cand.seq,
                    canonical_seq=cand.canonical_seq,
                    motif_id=target.motif_id,
                    tf_symbol=target.tf_symbol,
                    label=0,
                    tf_family=target.tf_family,
                    tf_class=target.tf_class,
                    species=target.species,
                    tax_group=target.tax_group,
                    source_file=cand.source_file,
                    source_id=cand.source_id,
                    source_extra=(
                        f"other_tf_source={cand.tf_symbol};"
                        f"other_tf_motif={cand.motif_id};"
                        f"target_tf={target.tf_symbol};"
                        f"target_motif={target.motif_id}"
                    ),
                )
            )

            n_added += 1

        audit.append(
            {
                "target_example_id": target.example_id,
                "fixed_len": target.fixed_len,
                "motif_id": target.motif_id,
                "tf_symbol": target.tf_symbol,
                "n_pool_gc_matched_other_tf": n_pool,
                "n_negatives_added": n_added,
                "neg_per_pos_requested": neg_per_pos,
            }
        )

        if (pos_i + 1) % 10000 == 0:
            print(
                f"[negative construction] processed={pos_i + 1}/{n_targets} "
                f"negatives={len(negatives)}"
            )

    print("[negative construction]")
    print(f"  Positives processed: {len(positives)}")
    print(f"  Negatives generated: {len(negatives)}")
    print(f"  Requested neg_per_pos: {neg_per_pos}")
    print(f"  GC tolerance in bases: {gc_tol_bases}")

    return negatives, audit


# ---------------------------------------------------------------------
# Group-model inference
# ---------------------------------------------------------------------

def discover_group_model_dirs(
    exp_root: Path,
    model_group_by: str,
) -> dict[str, Path]:
    """
    Find model directories under exp_root.

    A valid model directory must contain:
      - best_model.pt
      - split_manifest.csv

    The group value is inferred from split_manifest.csv.
    """
    model_dirs = []

    for best_model in exp_root.rglob("best_model.pt"):
        d = best_model.parent
        if (d / "split_manifest.csv").exists():
            model_dirs.append(d)

    if not model_dirs:
        raise RuntimeError(
            f"No class/family model directories found under {exp_root}. "
            "Expected subdirectories containing best_model.pt and split_manifest.csv."
        )

    group_to_dir = {}

    for d in sorted(model_dirs):
        group_value = infer_group_value_from_split_manifest(
            split_manifest=d / "split_manifest.csv",
            model_group_by=model_group_by,
        )

        if group_value in group_to_dir:
            print(
                f"[warning] Duplicate model for group={group_value}. "
                f"Keeping first: {group_to_dir[group_value]} ; ignoring: {d}"
            )
            continue

        group_to_dir[group_value] = d

    print("[model discovery]")
    print(f"  exp_root: {exp_root}")
    print(f"  model_group_by: {model_group_by}")
    print(f"  discovered models: {len(group_to_dir)}")

    return group_to_dir


def make_temp_exp_dir_for_group_model(
    base_config: Path,
    model_dir: Path,
    temp_root: Path,
) -> Path:
    """
    Create temporary experiment directory compatible with infer_motifgate_from_exp().

    Required expected layout:
      temp_exp_dir/
      ├── config.json
      ├── split_manifest.csv
      └── best_model.pt

    This function bridges both layouts.
    """
    tmp = temp_root / sanitize_group_name(model_dir.name)
    tmp.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_config, tmp / "config.json")
    shutil.copy2(model_dir / "split_manifest.csv", tmp / "split_manifest.csv")
    shutil.copy2(model_dir / "best_model.pt", tmp / "best_model.pt")

    return tmp


def predict_external_examples_with_group_models(
    motifgate_py: str,
    wdir: str,
    exp_root: str,
    examples: Sequence[Example],
    outdir: Path,
    model_group_by: str = "class",
    device: str = "cpu",
    batch_size: int = 512,
) -> Path:
    """
    Run MotifGate inference using one trained model per class/family.

    Routing:
      model_group_by == "class"  -> use ex.tf_class
      model_group_by == "family" -> use ex.tf_family

    Output:
      motifgate_external_predictions_by_group_models.csv
    """
    from motifgate_validation_utils import infer_motifgate_from_exp

    exp_root = Path(exp_root)
    base_config = exp_root / "config.json"

    if not base_config.exists():
        raise FileNotFoundError(
            f"Root config.json not found: {base_config}. "
            "For your structure, config.json must exist in the experiment root."
        )

    # Validate fixed_len consistency.
    try:
        with open(base_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg_fixed_len = int(cfg.get("fixed_len"))
        example_ks = sorted(set(int(ex.fixed_len) for ex in examples))
        if len(example_ks) != 1 or example_ks[0] != cfg_fixed_len:
            print(
                "[warning] K mismatch between external examples and experiment config:"
            )
            print(f"  config fixed_len: {cfg_fixed_len}")
            print(f"  example fixed_len values: {example_ks}")
            print("  This is usually incorrect. Use one K-specific experiment root per K.")
    except Exception as e:
        print(f"[warning] Could not validate fixed_len from config.json: {e}")

    group_to_dir = discover_group_model_dirs(
        exp_root=exp_root,
        model_group_by=model_group_by,
    )

    indices_by_group: dict[str, list[int]] = {}

    for i, ex in enumerate(examples):
        g = get_example_group_value(ex, model_group_by)
        indices_by_group.setdefault(g, []).append(i)

    logits_all = np.full(len(examples), np.nan, dtype=np.float64)
    probs_all = np.full(len(examples), np.nan, dtype=np.float64)
    model_dir_all = [""] * len(examples)

    unmatched_groups = sorted(
        g for g in indices_by_group.keys()
        if g not in group_to_dir
    )

    if unmatched_groups:
        print("[warning] Some external groups do not have a trained model:")
        for g in unmatched_groups[:50]:
            print(f"  missing model_group={g} ; n_examples={len(indices_by_group[g])}")
        if len(unmatched_groups) > 50:
            print(f"  ... {len(unmatched_groups) - 50} more")

    with tempfile.TemporaryDirectory(prefix="motifgate_group_infer_") as td:
        temp_root = Path(td)

        for group_value, idxs in sorted(indices_by_group.items()):
            if group_value not in group_to_dir:
                continue

            model_dir = group_to_dir[group_value]

            tmp_exp_dir = make_temp_exp_dir_for_group_model(
                base_config=base_config,
                model_dir=model_dir,
                temp_root=temp_root,
            )

            subset = [examples[i] for i in idxs]

            print(
                f"[inference] group={group_value} "
                f"n_examples={len(subset)} model_dir={model_dir}"
            )

            logits, probs = infer_motifgate_from_exp(
                motifgate_py=motifgate_py,
                wdir=wdir,
                exp_dir=tmp_exp_dir,
                examples=subset,
                device=device,
                batch_size=batch_size,
            )

            for local_i, global_i in enumerate(idxs):
                logits_all[global_i] = float(logits[local_i])
                probs_all[global_i] = float(probs[local_i])
                model_dir_all[global_i] = str(model_dir)

    pred_rows = []

    for ex, logit, prob, model_dir in zip(examples, logits_all, probs_all, model_dir_all):
        pred_rows.append({
            "example_id": ex.example_id,
            "seq": ex.seq,
            "motif_id": ex.motif_id,
            "tf_symbol": ex.tf_symbol,
            "tf_family": ex.tf_family,
            "tf_class": ex.tf_class,
            "label": ex.label,
            "logit": logit,
            "prob": prob,
            "model_group_by": model_group_by,
            "model_group": get_example_group_value(ex, model_group_by),
            "model_dir": model_dir,
        })

    pred_csv = outdir / "motifgate_external_predictions_by_group_models.csv"

    write_csv(
        pred_csv,
        [
            "example_id",
            "seq",
            "motif_id",
            "tf_symbol",
            "tf_family",
            "tf_class",
            "label",
            "logit",
            "prob",
            "model_group_by",
            "model_group",
            "model_dir",
        ],
        pred_rows,
    )

    matched = int(np.sum(np.isfinite(probs_all)))

    print("[group-model inference summary]")
    print(f"  total examples: {len(examples)}")
    print(f"  examples with predictions: {matched}")
    print(f"  examples without predictions: {len(examples) - matched}")
    print(f"  predictions_csv: {pred_csv}")

    return pred_csv


# ---------------------------------------------------------------------
# Audit outputs
# ---------------------------------------------------------------------

def write_fasta_scan_audit(
    outdir: Path,
    fasta_files: Sequence[Path],
    unibind_root_dir: Path,
) -> None:
    rows = []

    for fp in fasta_files:
        tf_symbol = infer_tf_from_unibind_path(fp, unibind_root_dir)
        motif_id = extract_motif_id_from_unibind_filename(fp)

        rows.append(
            {
                "tf_symbol": tf_symbol,
                "motif_id_from_filename": motif_id,
                "source_file": str(fp),
                "filename": fp.name,
            }
        )

    write_csv(
        outdir / "unibind_fasta_scan_audit.csv",
        ["tf_symbol", "motif_id_from_filename", "source_file", "filename"],
        rows,
    )


def write_group_coverage_audit(
    outdir: Path,
    examples: Sequence[Example],
    model_group_by: str,
    exp_root: str | None,
) -> None:
    rows = []

    group_counts = {}

    for ex in examples:
        g = get_example_group_value(ex, model_group_by)
        key = (
            model_group_by,
            g,
            ex.fixed_len,
        )
        group_counts.setdefault(key, {"n": 0, "n_pos": 0, "n_neg": 0})
        group_counts[key]["n"] += 1
        group_counts[key]["n_pos"] += int(ex.label == 1)
        group_counts[key]["n_neg"] += int(ex.label == 0)

    available_groups = set()

    if exp_root:
        try:
            group_to_dir = discover_group_model_dirs(Path(exp_root), model_group_by)
            available_groups = set(group_to_dir.keys())
        except Exception as e:
            print(f"[warning] Could not write full group coverage audit: {e}")

    for (gb, g, k), vals in sorted(group_counts.items()):
        rows.append({
            "model_group_by": gb,
            "group": g,
            "fixed_len": k,
            "n": vals["n"],
            "n_pos": vals["n_pos"],
            "n_neg": vals["n_neg"],
            "has_trained_model": int(g in available_groups) if available_groups else "",
        })

    write_csv(
        outdir / "external_group_coverage_audit.csv",
        [
            "model_group_by",
            "group",
            "fixed_len",
            "n",
            "n_pos",
            "n_neg",
            "has_trained_model",
        ],
        rows,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--unibind_fasta_dir",
        required=True,
        help="Directory containing UniBind TFBS FASTA files. Recursive TF-folder structure is supported.",
    )

    ap.add_argument(
        "--mapping_csv",
        required=True,
        help="CSV with tf_symbol/motif_id columns. Used mainly for metadata; motif_id is preferably extracted from filename.",
    )

    ap.add_argument(
        "--k_list",
        default="7,8,9,10,11,12,13",
        help="Comma-separated fixed sequence lengths. For model inference, use one K per run.",
    )

    ap.add_argument(
        "--species",
        default="Homo sapiens",
    )

    ap.add_argument(
        "--outdir",
        required=True,
    )

    ap.add_argument(
        "--max_per_tf",
        type=int,
        default=5000,
        help=(
            "Maximum records per FASTA file. Original name kept for compatibility; "
            "in this recursive version the limit is applied per FASTA file."
        ),
    )

    ap.add_argument(
        "--neg_per_pos",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--gc_tol_bases",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=251031,
    )

    ap.add_argument(
        "--motifgate_py",
        default="motifgate",
        help="MotifGate code source for PWM scoring/inference. Default 'motifgate' uses "
             "the installed package; pass a path to the original single-file script for "
             "backward compatibility.",
    )

    ap.add_argument(
        "--wdir",
        default=".",
        help="MotifGate working directory containing Shared/.",
    )

    ap.add_argument(
        "--predictions_csv",
        default="",
        help="Optional external predictions CSV. Used instead of direct model inference if provided alone.",
    )

    ap.add_argument(
        "--motifgate_exp_dir",
        default="",
        help=(
            "Optional single trained MotifGate experiment directory containing "
            "config.json, split_manifest.csv and best_model.pt."
        ),
    )

    ap.add_argument(
        "--motifgate_exp_root",
        default="",
        help=(
            "Optional root directory containing config.json and recursive class/family "
            "model subdirectories with best_model.pt and split_manifest.csv."
        ),
    )

    ap.add_argument(
        "--model_group_by",
        choices=["class", "family"],
        default="class",
        help="How to route external examples to trained group-specific models.",
    )

    ap.add_argument(
        "--device",
        default="cpu",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    unibind_root_dir = Path(args.unibind_fasta_dir)

    if not unibind_root_dir.exists():
        raise FileNotFoundError(f"UniBind directory does not exist: {unibind_root_dir}")

    k_list = [
        int(x)
        for x in args.k_list.replace(" ", "").split(",")
        if x
    ]

    if args.motifgate_exp_root and len(k_list) != 1:
        print(
            "[warning] --motifgate_exp_root should usually be used with exactly one K. "
            f"Current k_list={k_list}. Use one K-specific run per trained experiment root."
        )

    mapping = load_mapping_csv(args.mapping_csv)

    fasta_files = find_unibind_fasta_files(unibind_root_dir)

    if not fasta_files:
        raise FileNotFoundError(
            f"No FASTA files found recursively in {args.unibind_fasta_dir}. "
            "Expected .fa, .fasta, .fas, .fa.gz, .fasta.gz or .fas.gz."
        )

    write_fasta_scan_audit(
        outdir=outdir,
        fasta_files=fasta_files,
        unibind_root_dir=unibind_root_dir,
    )

    positives = build_positive_examples_from_fasta(
        fasta_files=fasta_files,
        unibind_root_dir=unibind_root_dir,
        mapping=mapping,
        k_list=k_list,
        species=args.species,
        max_per_file=args.max_per_tf,
        seed=args.seed,
    )

    if not positives:
        raise RuntimeError(
            "No positive examples were built. "
            "Check UniBind FASTA files, motif IDs in filenames and mapping_csv."
        )

    pwm_scores = None

    if args.motifgate_py:
        try:
            pwm_scores = pwm_raw_scores_with_base(
                args.motifgate_py,
                args.wdir,
                positives,
            )

            for ex, sc in zip(positives, pwm_scores):
                ex.score_pwm_raw = float(sc)

        except Exception as e:
            print(
                "[warning] PWM scoring of positives failed. "
                f"Negatives will not use PWM caps: {e}"
            )

    negatives, audit = build_other_tf_hard_negatives(
        positives=positives,
        neg_per_pos=args.neg_per_pos,
        gc_tol_bases=args.gc_tol_bases,
        seed=args.seed + 11,
        pwm_scores=pwm_scores,
    )

    examples = list(positives) + list(negatives)

    for i, ex in enumerate(examples):
        ex.example_id = f"UB_EXT_{i:09d}"

    write_csv(
        outdir / "external_examples.csv",
        EXAMPLE_FIELDS,
        examples_to_rows(examples),
    )

    write_csv(
        outdir / "negative_sampling_audit.csv",
        [
            "target_example_id",
            "fixed_len",
            "motif_id",
            "tf_symbol",
            "n_pool_gc_matched_other_tf",
            "n_negatives_added",
            "neg_per_pos_requested",
        ],
        audit,
    )

    write_group_coverage_audit(
        outdir=outdir,
        examples=examples,
        model_group_by=args.model_group_by,
        exp_root=args.motifgate_exp_root or None,
    )

    n_pos = sum(ex.label == 1 for ex in examples)
    n_neg = sum(ex.label == 0 for ex in examples)

    write_json(
        outdir / "dataset_summary.json",
        {
            "source": "UniBind",
            "directory_structure": "recursive TF folders",
            "motif_id_source": "UniBind FASTA filename when available",
            "n_fasta_files": len(fasta_files),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": len(examples),
            "k_list": k_list,
            "species": args.species,
            "neg_per_pos": args.neg_per_pos,
            "gc_tol_bases": args.gc_tol_bases,
            "motifgate_exp_dir": args.motifgate_exp_dir,
            "motifgate_exp_root": args.motifgate_exp_root,
            "model_group_by": args.model_group_by,
        },
    )

    predictions_csv_for_metrics = args.predictions_csv or ""

    if args.motifgate_exp_root:
        if not args.motifgate_py:
            raise RuntimeError("--motifgate_exp_root requires --motifgate_py")

        predictions_csv_for_metrics = str(
            predict_external_examples_with_group_models(
                motifgate_py=args.motifgate_py,
                wdir=args.wdir,
                exp_root=args.motifgate_exp_root,
                examples=examples,
                outdir=outdir,
                model_group_by=args.model_group_by,
                device=args.device,
                batch_size=args.batch_size,
            )
        )

    attach_and_export_metrics(
        outdir=outdir,
        examples=examples,
        motifgate_py=args.motifgate_py or None,
        wdir=args.wdir,
        predictions_csv=predictions_csv_for_metrics or None,
        motifgate_exp_dir=args.motifgate_exp_dir or None,
        device=args.device,
        batch_size=args.batch_size,
    )

    print()
    print("[done] UniBind external validation files written to:")
    print(f"  {outdir}")
    print()
    print("Main outputs:")
    print(f"  {outdir / 'external_examples.csv'}")
    print(f"  {outdir / 'negative_sampling_audit.csv'}")
    print(f"  {outdir / 'dataset_summary.json'}")
    print(f"  {outdir / 'unibind_fasta_scan_audit.csv'}")
    print(f"  {outdir / 'external_group_coverage_audit.csv'}")

    if args.motifgate_exp_root:
        print(f"  {outdir / 'motifgate_external_predictions_by_group_models.csv'}")
        print(f"  {outdir / 'external_metrics.csv'}")
        print(f"  {outdir / 'external_metrics_global.csv'}")


if __name__ == "__main__":
    main()
