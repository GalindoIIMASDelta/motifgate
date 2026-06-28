from __future__ import annotations

"""PWM/PFM parsing (TRANSFAC, JASPAR), site loading and exact alignment scoring."""

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

from .utils import (base_id, canonical_seq, seq_to_idx)

@dataclass
class MotifRecord:
    motif_id: str
    pwm: np.ndarray
    meta: Dict[str, str]


class TransfacIndex:
    def __init__(self, directory: str):
        self.directory = directory
        self.pwm_exact: Dict[str, np.ndarray] = {}
        self.meta_exact: Dict[str, Dict[str, str]] = {}
        self.pwm_base: Dict[str, np.ndarray] = {}
        self.meta_base: Dict[str, Dict[str, str]] = {}
        if os.path.isdir(directory):
            self._load()

    def _load(self) -> None:
        files = sorted(glob.glob(os.path.join(self.directory, "*")))
        for fp in files:
            try:
                rec = self._parse_transfac_file(fp)
                if rec is None:
                    continue
                self.pwm_exact[rec.motif_id] = rec.pwm
                self.meta_exact[rec.motif_id] = rec.meta
                bid = base_id(rec.motif_id)
                keep = True
                if bid in self.pwm_base:
                    prev_mid = self.meta_base[bid].get("_motif_id", "")
                    try:
                        prev_v = float(prev_mid.split(".")[1])
                        cur_v = float(rec.motif_id.split(".")[1])
                        keep = cur_v >= prev_v
                    except Exception:
                        keep = False
                if keep:
                    meta_b = dict(rec.meta)
                    meta_b["_motif_id"] = rec.motif_id
                    self.pwm_base[bid] = rec.pwm
                    self.meta_base[bid] = meta_b
            except Exception:
                continue

    def _parse_transfac_file(self, fp: str) -> Optional[MotifRecord]:
        motif_id = None
        meta: Dict[str, str] = {}
        rows = []
        in_matrix = False
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("AC"):
                    parts = line.split()
                    if len(parts) >= 2:
                        motif_id = parts[1]
                elif line.startswith("CC"):
                    cc = line[2:].strip()
                    if ":" in cc:
                        k, v = cc.split(":", 1)
                        meta[k.strip()] = v.strip()
                elif line.startswith("PO"):
                    in_matrix = True
                elif line.startswith("XX"):
                    in_matrix = False
                elif in_matrix and re.match(r"^\d+", line):
                    parts = line.split()
                    if len(parts) >= 5:
                        vals = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                        rows.append(vals)
                elif line.startswith("//"):
                    break
        if motif_id is None or not rows:
            return None
        mat = np.asarray(rows, dtype=np.float32)
        mat = mat + 1.0
        mat = mat / mat.sum(axis=1, keepdims=True)
        return MotifRecord(motif_id=motif_id, pwm=mat, meta=meta)

    def get_pwm(self, motif_id: str) -> Optional[np.ndarray]:
        if motif_id in self.pwm_exact:
            return self.pwm_exact[motif_id]
        return self.pwm_base.get(base_id(motif_id))

    def get_meta(self, motif_id: str) -> Optional[Dict[str, str]]:
        if motif_id in self.meta_exact:
            return self.meta_exact[motif_id]
        return self.meta_base.get(base_id(motif_id))

class JasparPWMIndex:
    def __init__(self, directory: str):
        self.directory = directory
        self.pwm_exact: Dict[str, np.ndarray] = {}
        self.pwm_base: Dict[str, np.ndarray] = {}
        if os.path.isdir(directory):
            self._load()

    def _load(self) -> None:
        for fp in sorted(glob.glob(os.path.join(self.directory, "*.jaspar"))):
            try:
                mid, pwm = self._parse_jaspar(fp)
                if mid is None or pwm is None:
                    continue
                self.pwm_exact[mid] = pwm
                self.pwm_base[base_id(mid)] = pwm
            except Exception:
                continue

    def _parse_jaspar(self, fp: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None, None
        mid = None
        if lines[0].startswith(">"):
            parts = lines[0][1:].split()
            if parts:
                mid = parts[0]
        arrs = {}
        for ln in lines[1:]:
            if re.match(r"^[ACGT]\s*\[", ln):
                b = ln[0]
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)
                arrs[b] = np.asarray([float(x) for x in nums], dtype=np.float32)
        if mid is None or len(arrs) != 4:
            return None, None
        mat = np.stack([arrs["A"], arrs["C"], arrs["G"], arrs["T"]], axis=1)
        mat = mat + 1.0
        mat = mat / mat.sum(axis=1, keepdims=True)
        return mid, mat

class PWMIndex:
    def __init__(self, transfac: TransfacIndex, jaspar: JasparPWMIndex):
        self.transfac = transfac
        self.jaspar = jaspar

    def get_full(self, motif_id: str) -> Optional[np.ndarray]:
        pwm = self.transfac.get_pwm(motif_id)
        if pwm is not None:
            return pwm
        pwm = self.jaspar.pwm_exact.get(motif_id)
        if pwm is None:
            pwm = self.jaspar.pwm_base.get(base_id(motif_id))
        return pwm

    def get_meta(self, motif_id: str) -> Optional[Dict[str, str]]:
        return self.transfac.get_meta(motif_id)

def parse_sites_file(fp: str) -> List[str]:
    seqs = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().upper()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            m = re.search(r"[ACGT]+", line)
            if m:
                seqs.append(m.group(0))
    return seqs

def best_alignment_score(seq_idx: np.ndarray, log_pwm: np.ndarray) -> Tuple[float, int, bool, bool]:
    Ls = len(seq_idx)
    Lp = log_pwm.shape[0]

    def score_with_orientation(seq_arr: np.ndarray, pwm_arr: np.ndarray) -> Tuple[float, int, bool]:
        Ls2 = len(seq_arr)
        Lp2 = pwm_arr.shape[0]
        best_s = -1e18
        best_off = 0
        mode_pwm_on_seq = Ls2 >= Lp2
        if mode_pwm_on_seq:
            for off in range(0, Ls2 - Lp2 + 1):
                s = 0.0
                for j in range(Lp2):
                    bi = seq_arr[off + j]
                    s += float(pwm_arr[j, bi]) if bi >= 0 else math.log(0.25)
                if s > best_s:
                    best_s = s
                    best_off = off
        else:
            for off in range(0, Lp2 - Ls2 + 1):
                s = 0.0
                for j in range(Ls2):
                    bi = seq_arr[j]
                    s += float(pwm_arr[off + j, bi]) if bi >= 0 else math.log(0.25)
                if s > best_s:
                    best_s = s
                    best_off = off
        return best_s, best_off, mode_pwm_on_seq

    s_f, off_f, mode_f = score_with_orientation(seq_idx, log_pwm)
    comp_idx = np.array([3, 2, 1, 0], dtype=np.int64)
    rc_idx = comp_idx[seq_idx[::-1]]
    s_r, off_r, mode_r = score_with_orientation(rc_idx, log_pwm)

    if s_r > s_f:
        return s_r, off_r, True, mode_r
    return s_f, off_f, False, mode_f

def compute_raw_prior_scores(seqs: List[str], mids: List[str], pwm_index: PWMIndex) -> np.ndarray:
    scores = np.zeros((len(seqs),), dtype=np.float32)
    pwm_cache: Dict[str, Optional[np.ndarray]] = {}
    log_cache: Dict[str, Optional[np.ndarray]] = {}
    for i, (seq, mid) in enumerate(zip(seqs, mids)):
        if mid not in pwm_cache:
            pwm_cache[mid] = pwm_index.get_full(mid)
            if pwm_cache[mid] is None:
                log_cache[mid] = None
            else:
                log_cache[mid] = np.log(np.clip(pwm_cache[mid], 1e-8, 1.0))
        log_pwm = log_cache[mid]
        if log_pwm is None:
            scores[i] = 0.0
            continue
        score, _, _, _ = best_alignment_score(seq_to_idx(seq), log_pwm)
        scores[i] = float(score)
    return scores

def load_fasta_sequences(path: str) -> List[str]:
    seqs = []
    cur = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().upper()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                continue
            cur.append(re.sub(r"[^ACGT]", "", line))
    if cur:
        seqs.append("".join(cur))
    return seqs

def extract_fixed_len_windows(seqs: List[str], fixed_len: int, stride: int, limit: int) -> List[str]:
    out = []
    for seq in seqs:
        if len(seq) < fixed_len:
            continue
        if len(seq) == fixed_len:
            out.append(canonical_seq(seq))
        else:
            for i in range(0, len(seq) - fixed_len + 1, max(1, stride)):
                out.append(canonical_seq(seq[i:i + fixed_len]))
                if limit > 0 and len(out) >= limit:
                    return out
        if limit > 0 and len(out) >= limit:
            break
    return out

def load_background_sequences_from_dir(directory: str, fixed_len: int, stride: int, limit: int) -> List[str]:
    seqs = []
    files = []
    for pat in ("*.fa", "*.fasta", "*.fas", "*.sites", "*.txt"):
        files.extend(sorted(glob.glob(os.path.join(directory, pat))))
    for fp in files:
        if fp.endswith(".sites"):
            local = parse_sites_file(fp)
            local = [canonical_seq(s) for s in local if len(s) == fixed_len]
        else:
            local = extract_fixed_len_windows(load_fasta_sequences(fp), fixed_len, stride, limit - len(seqs) if limit > 0 else limit)
        seqs.extend(local)
        if limit > 0 and len(seqs) >= limit:
            break
    return seqs[:limit] if limit > 0 else seqs
