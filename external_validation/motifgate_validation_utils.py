#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motifgate_validation_utils.py

Shared utilities for MotifGate external validation scripts.

This helper intentionally separates three tasks:
1) benchmark construction, producing CSVs with columns seq, motif_id, label;
2) PWM-prior scoring, using the local MotifGate/Tfbs_v33.py implementation if available;
3) optional MotifGate inference, using a trained experiment directory containing config.json,
   split_manifest.csv and best_model.pt.

The scripts can still be used without direct model inference. In that case, provide a
predictions CSV with either:
    example_id, prob
or:
    seq, motif_id, prob
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASES = "ACGT"
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
B2I = {b: i for i, b in enumerate(BASES)}


@dataclass
class Example:
    example_id: str
    dataset: str
    source_db: str
    fixed_len: int
    seq: str
    canonical_seq: str
    motif_id: str
    tf_symbol: str
    label: int
    tf_family: str = ""
    tf_class: str = ""
    cell_type: str = ""
    species: str = ""
    tax_group: str = ""
    source_file: str = ""
    source_id: str = ""
    score_pwm_raw: float = float("nan")
    source_extra: str = ""


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def read_csv_dict(path: str | Path) -> List[dict]:
    with open_text(path) as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_csv_rows(path: str | Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def revcomp(seq: str) -> str:
    return "".join(COMP.get(ch, "N") for ch in seq.upper()[::-1])


def canonical(seq: str) -> str:
    s = re.sub(r"[^ACGT]", "", str(seq).upper())
    rc = revcomp(s)
    return s if s <= rc else rc


def center_crop(seq: str, k: int) -> Optional[str]:
    s = re.sub(r"[^ACGT]", "", str(seq).upper())
    if len(s) < k:
        return None
    start = (len(s) - k) // 2
    return s[start:start + k]


def gc_count(seq: str) -> int:
    s = str(seq).upper()
    return s.count("G") + s.count("C")


def gc_fraction(seq: str) -> float:
    s = str(seq).upper()
    return gc_count(s) / max(1, len(s))


def sequence_entropy(seq: str) -> float:
    s = str(seq).upper()
    n = max(1, len(s))
    vals = []
    for b in BASES:
        p = s.count(b) / n
        if p > 0:
            vals.append(-p * math.log2(p))
    return float(sum(vals))


def seq_to_idx(seq: str) -> np.ndarray:
    return np.asarray([B2I.get(ch, -1) for ch in str(seq).upper()], dtype=np.int64)


def load_fasta_records(path: str | Path) -> Iterable[Tuple[str, str]]:
    name = None
    chunks: List[str] = []
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, re.sub(r"[^ACGT]", "", "".join(chunks).upper())
                name = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            yield name, re.sub(r"[^ACGT]", "", "".join(chunks).upper())


def parse_bed(path: str | Path, max_rows: int = 0) -> Iterable[dict]:
    n = 0
    with open_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            name = parts[3] if len(parts) >= 4 else f"{Path(path).stem}_{n}"
            summit = None
            # narrowPeak column 10 stores summit offset from start.
            if len(parts) >= 10:
                try:
                    off = int(float(parts[9]))
                    if off >= 0:
                        summit = start + off
                except Exception:
                    summit = None
            if summit is None:
                summit = (start + end) // 2
            yield {
                "chrom": parts[0],
                "start": start,
                "end": end,
                "name": name,
                "score": parts[4] if len(parts) >= 5 else "",
                "strand": parts[5] if len(parts) >= 6 else ".",
                "summit": int(summit),
                "raw": parts,
            }
            n += 1
            if max_rows and n >= max_rows:
                break


def overlap_intervals(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return int(a_start) < int(b_end) and int(b_start) < int(a_end)


def build_interval_index(bed_rows: Iterable[dict]) -> Dict[str, List[Tuple[int, int]]]:
    idx: Dict[str, List[Tuple[int, int]]] = {}
    for r in bed_rows:
        idx.setdefault(r["chrom"], []).append((int(r["start"]), int(r["end"])))
    for chrom in idx:
        idx[chrom].sort()
    return idx


def overlaps_index(chrom: str, start: int, end: int, idx: Dict[str, List[Tuple[int, int]]]) -> bool:
    for a, b in idx.get(chrom, []):
        if b <= start:
            continue
        if a >= end:
            break
        if overlap_intervals(start, end, a, b):
            return True
    return False


def load_mapping_csv(path: str | Path) -> Dict[str, dict]:
    """
    Expected flexible columns:
      tf_symbol or unibind_tf or target
      motif_id
      tf_family
      tf_class
      species
      tax_group
    """
    rows = read_csv_dict(path)
    out = {}
    for r in rows:
        key = (
            r.get("unibind_tf")
            or r.get("tf_symbol")
            or r.get("target")
            or r.get("tf")
            or r.get("name")
            or ""
        ).strip()
        mid = (r.get("motif_id") or r.get("matrix_id") or r.get("jaspar_id") or "").strip()
        if key:
            out[key.upper()] = r
        if mid:
            out[mid.upper()] = r
    return out


def infer_tf_from_filename(path: str | Path) -> str:
    stem = Path(path).name
    stem = re.sub(r"\.(fa|fasta|fas|bed|bed.gz|fa.gz|fasta.gz)$", "", stem, flags=re.I)
    # UniBind per-TF files often contain extra tokens. Keep first token before common delimiters.
    token = re.split(r"[._| -]", stem)[0]
    return token.upper()


def metric_ap_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as e:
        raise RuntimeError("scikit-learn is required for AP/AUROC metrics. Install with: pip install scikit-learn") from e
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y_true, y_score)), float(roc_auc_score(y_true, y_score))


def load_prediction_scores(predictions_csv: str | Path, examples: Sequence[Example]) -> np.ndarray:
    rows = read_csv_dict(predictions_csv)
    by_id = {}
    by_pair = {}
    for r in rows:
        prob = r.get("prob") or r.get("score") or r.get("prediction") or r.get("prob_calibrated")
        if prob is None or prob == "":
            continue
        try:
            p = float(prob)
        except Exception:
            continue
        eid = (r.get("example_id") or "").strip()
        if eid:
            by_id[eid] = p
        seq = (r.get("seq") or r.get("sequence") or "").strip().upper()
        mid = (r.get("motif_id") or r.get("matrix_id") or "").strip()
        if seq and mid:
            by_pair[(seq, mid)] = p
    out = []
    missing = 0
    for ex in examples:
        if ex.example_id in by_id:
            out.append(by_id[ex.example_id])
        elif (ex.seq, ex.motif_id) in by_pair:
            out.append(by_pair[(ex.seq, ex.motif_id)])
        else:
            out.append(float("nan"))
            missing += 1
    if missing:
        print(f"[warning] Missing predictions for {missing}/{len(examples)} examples.")
    return np.asarray(out, dtype=np.float64)


def examples_to_rows(examples: Sequence[Example]) -> List[dict]:
    rows = []
    for ex in examples:
        rows.append({
            "example_id": ex.example_id,
            "dataset": ex.dataset,
            "source_db": ex.source_db,
            "fixed_len": ex.fixed_len,
            "seq": ex.seq,
            "canonical_seq": ex.canonical_seq,
            "motif_id": ex.motif_id,
            "tf_symbol": ex.tf_symbol,
            "label": ex.label,
            "tf_family": ex.tf_family,
            "tf_class": ex.tf_class,
            "cell_type": ex.cell_type,
            "species": ex.species,
            "tax_group": ex.tax_group,
            "source_file": ex.source_file,
            "source_id": ex.source_id,
            "gc_fraction": gc_fraction(ex.seq),
            "gc_count": gc_count(ex.seq),
            "sequence_entropy": sequence_entropy(ex.seq),
            "score_pwm_raw": ex.score_pwm_raw,
            "source_extra": ex.source_extra,
        })
    return rows


EXAMPLE_FIELDS = [
    "example_id", "dataset", "source_db", "fixed_len", "seq", "canonical_seq",
    "motif_id", "tf_symbol", "label", "tf_family", "tf_class", "cell_type",
    "species", "tax_group", "source_file", "source_id", "gc_fraction",
    "gc_count", "sequence_entropy", "score_pwm_raw", "source_extra"
]


def summarize_metrics(examples: Sequence[Example], score: np.ndarray, score_name: str) -> List[dict]:
    y = np.asarray([ex.label for ex in examples], dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    rows = []
    if np.sum(mask) > 0 and len(np.unique(y[mask])) >= 2:
        ap, auc = metric_ap_auc(y[mask], score[mask])
    else:
        ap, auc = float("nan"), float("nan")
    rows.append({
        "level": "global",
        "group": "__ALL__",
        "score_name": score_name,
        "n": int(np.sum(mask)),
        "n_pos": int(np.sum(y[mask] == 1)),
        "n_neg": int(np.sum(y[mask] == 0)),
        "ap": ap,
        "auroc": auc,
    })
    for group_field in ["fixed_len", "motif_id", "tf_symbol", "tf_family", "tf_class", "species", "tax_group", "cell_type"]:
        values = sorted(set(str(getattr(ex, group_field, "")) for ex in examples))
        for val in values:
            idx = np.asarray([str(getattr(ex, group_field, "")) == val for ex in examples], dtype=bool) & mask
            if np.sum(idx) == 0:
                continue
            if len(np.unique(y[idx])) < 2:
                ap, auc = float("nan"), float("nan")
            else:
                ap, auc = metric_ap_auc(y[idx], score[idx])
            rows.append({
                "level": group_field,
                "group": val,
                "score_name": score_name,
                "n": int(np.sum(idx)),
                "n_pos": int(np.sum(y[idx] == 1)),
                "n_neg": int(np.sum(y[idx] == 0)),
                "ap": ap,
                "auroc": auc,
            })
    return rows


def import_motifgate_module(motifgate_py: str | Path | None = None):
    """Return the MotifGate code module.

    Preferred: the installed ``motifgate`` package (the refactored, modular version
    of the original single-file script). Pass an empty value, ``"package"`` or
    ``"motifgate"`` to use it. For backward compatibility, a path to the original
    standalone script (e.g. ``Tfbs_transformer_v33_c.py``) is still accepted and
    loaded as a module. Either way the returned object exposes the same top-level
    names (TransfacIndex, PWMIndex, build_model, make_datasets_and_stats,
    predict_sequences, compute_raw_prior_scores, LabeledExample, canonical_seq, ...).
    """
    import sys

    key = str(motifgate_py).strip().lower() if motifgate_py is not None else ""
    if key in ("", "none", "package", "motifgate"):
        import motifgate as mod
        return mod

    module_name = "motifgate_base"
    spec = importlib.util.spec_from_file_location(module_name, str(motifgate_py))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import MotifGate base script from {motifgate_py}")

    mod = importlib.util.module_from_spec(spec)

    # Critical for scripts using @dataclass.
    # Without this, dataclasses can fail with:
    # 'NoneType' object has no attribute '__dict__'
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)

    return mod


def make_pwm_index_from_base(mod, wdir: str | Path):
    wdir = Path(wdir)
    transfac_dir = wdir / "Shared" / "transfac"
    jaspar_dir = wdir / "Shared" / "Jaspar_pwms"
    # Some runs use Shared/Jaspar_pwms and others Shared/JASPAR; keep explicit.
    ti = mod.TransfacIndex(str(transfac_dir))
    ji = mod.JasparPWMIndex(str(jaspar_dir))
    return mod.PWMIndex(ti, ji)


def pwm_raw_scores_with_base(motifgate_py: str | Path, wdir: str | Path, examples: Sequence[Example]) -> np.ndarray:
    mod = import_motifgate_module(motifgate_py)
    pwm_index = make_pwm_index_from_base(mod, wdir)
    seqs = [ex.seq for ex in examples]
    mids = [ex.motif_id for ex in examples]
    return mod.compute_raw_prior_scores(seqs, mids, pwm_index).astype(np.float64)


def infer_motifgate_from_exp(
    motifgate_py: str | Path,
    wdir: str | Path,
    exp_dir: str | Path,
    examples: Sequence[Example],
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional direct inference from a trained MotifGate experiment directory.

    Required files inside exp_dir:
      config.json
      split_manifest.csv
      best_model.pt

    This reconstructs the feature normalizer and prior scaling from the training
    manifest, then loads the trained checkpoint.
    """
    import argparse
    import torch

    exp_dir = Path(exp_dir)
    mod = import_motifgate_module(motifgate_py)
    pwm_index = make_pwm_index_from_base(mod, wdir)

    with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    args.device = device
    if not hasattr(args, "use_prior_z"):
        args.use_prior_z = 1
    if not hasattr(args, "use_kmer_fcgr"):
        args.use_kmer_fcgr = 0
    if not hasattr(args, "model_arch"):
        args.model_arch = "pwmconv_residual"

    manifest = read_csv_dict(exp_dir / "split_manifest.csv")
    def to_labeled(r):
        return mod.LabeledExample(
            seq=(r.get("seq") or "").upper(),
            canon=r.get("canonical_seq") or mod.canonical_seq((r.get("seq") or "").upper()),
            motif_id=r.get("motif_id") or "",
            tf_family=r.get("tf_family") or "",
            tf_class=r.get("tf_class") or "",
            label=int(float(r.get("label", 0))),
            source=r.get("source") or "manifest",
        )

    train = [to_labeled(r) for r in manifest if r.get("split") == "train"]
    val = [to_labeled(r) for r in manifest if r.get("split") == "val"]
    test = [to_labeled(r) for r in manifest if r.get("split") == "test"]

    ds_train, ds_val, ds_test, stats = mod.make_datasets_and_stats(
        train, val, test, pwm_index=pwm_index,
        use_prior_z=bool(int(getattr(args, "use_prior_z", 1))),
        use_kmer_fcgr=bool(int(getattr(args, "use_kmer_fcgr", 0))),
    )

    model = mod.build_model(args, feat_dim=stats["feature_dim"],
                            prior_mu=stats["prior_mu_train"],
                            prior_sd=stats["prior_sd_train"])
    state = torch.load(exp_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()

    seqs = [ex.seq for ex in examples]
    mids = [ex.motif_id for ex in examples]
    fams = [ex.tf_family for ex in examples]
    clss = [ex.tf_class for ex in examples]
    logits, probs = mod.predict_sequences(
        model=model,
        seqs=seqs,
        motif_ids=mids,
        tf_families=fams,
        tf_classes=clss,
        pwm_index=pwm_index,
        feat_norm=ds_train.feat_norm,
        prior_mu=stats["prior_mu_train"],
        prior_sd=stats["prior_sd_train"],
        use_prior_z=bool(int(getattr(args, "use_prior_z", 1))),
        use_kmer_fcgr=bool(int(getattr(args, "use_kmer_fcgr", 0))),
        device=torch.device(device),
        batch_size=batch_size,
    )
    return logits.astype(np.float64), probs.astype(np.float64)


def attach_and_export_metrics(
    outdir: str | Path,
    examples: Sequence[Example],
    motifgate_py: Optional[str] = None,
    wdir: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    motifgate_exp_dir: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 512,
) -> None:
    outdir = ensure_dir(outdir)
    metric_rows = []

    if motifgate_py and wdir:
        try:
            pwm_scores = pwm_raw_scores_with_base(motifgate_py, wdir, examples)
            for ex, sc in zip(examples, pwm_scores):
                ex.score_pwm_raw = float(sc)
            metric_rows.extend(summarize_metrics(examples, pwm_scores, "pwm_raw"))
            # Sigmoid of raw PWM is not calibrated but can be useful as ranking-equivalent.
            metric_rows.extend(summarize_metrics(examples, 1.0 / (1.0 + np.exp(-np.clip(pwm_scores, -50, 50))), "pwm_sigmoid_raw"))
            write_csv(outdir / "external_examples_with_pwm.csv", EXAMPLE_FIELDS, examples_to_rows(examples))
        except Exception as e:
            print(f"[warning] PWM baseline scoring failed: {e}")

    if predictions_csv:
        pred = load_prediction_scores(predictions_csv, examples)
        metric_rows.extend(summarize_metrics(examples, pred, "motifgate_predictions_csv"))
        pred_rows = []
        for ex, p in zip(examples, pred):
            pred_rows.append({"example_id": ex.example_id, "seq": ex.seq, "motif_id": ex.motif_id, "label": ex.label, "prob": p})
        write_csv(outdir / "external_predictions_merged.csv",
                  ["example_id", "seq", "motif_id", "label", "prob"], pred_rows)

    if motifgate_exp_dir and motifgate_py and wdir:
        try:
            logits, probs = infer_motifgate_from_exp(
                motifgate_py=motifgate_py,
                wdir=wdir,
                exp_dir=motifgate_exp_dir,
                examples=examples,
                device=device,
                batch_size=batch_size,
            )
            metric_rows.extend(summarize_metrics(examples, probs, "motifgate_exp_prob"))
            rows = []
            for ex, logit, prob in zip(examples, logits, probs):
                rows.append({
                    "example_id": ex.example_id,
                    "seq": ex.seq,
                    "motif_id": ex.motif_id,
                    "label": ex.label,
                    "logit": float(logit),
                    "prob": float(prob),
                })
            write_csv(outdir / "motifgate_external_predictions.csv",
                      ["example_id", "seq", "motif_id", "label", "logit", "prob"], rows)
        except Exception as e:
            print(f"[warning] Direct MotifGate inference failed: {e}")

    if metric_rows:
        write_csv(outdir / "external_metrics.csv",
                  ["level", "group", "score_name", "n", "n_pos", "n_neg", "ap", "auroc"],
                  metric_rows)
        # Also write a compact global-only table.
        glob = [r for r in metric_rows if r["level"] == "global"]
        write_csv(outdir / "external_metrics_global.csv",
                  ["level", "group", "score_name", "n", "n_pos", "n_neg", "ap", "auroc"],
                  glob)
