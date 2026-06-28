from __future__ import annotations

"""Generic utilities: seeding, IO, sequence encoding and scalar features."""

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

from .constants import (B2I, BASES, CGR_BITS, COMP)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_csv(path: str, header: List[str], rows: Iterable[Iterable]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))

def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
        return auc, ap
    except Exception:
        return float("nan"), float("nan")

def base_id(motif_id: str) -> str:
    return motif_id.split(".")[0]

def revcomp_seq(seq: str) -> str:
    return "".join(COMP.get(b, "N") for b in seq[::-1])

def canonical_seq(seq: str) -> str:
    rc = revcomp_seq(seq)
    return seq if seq <= rc else rc

def stable_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    s = s.strip("._")
    return s[:120] or "unnamed"

def split_counts(n: int, train_frac: float, val_frac: float, test_frac: float, min_each: int = 1) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    n_train = int(math.floor(train_frac * n))
    n_val = int(math.floor(val_frac * n))
    n_test = n - n_train - n_val
    if n >= 3 * min_each:
        while n_train < min_each:
            n_train += 1
            if n_test > min_each:
                n_test -= 1
            elif n_val > min_each:
                n_val -= 1
        while n_val < min_each:
            n_val += 1
            if n_train > min_each:
                n_train -= 1
            elif n_test > min_each:
                n_test -= 1
        while n_test < min_each:
            n_test += 1
            if n_train > min_each:
                n_train -= 1
            elif n_val > min_each:
                n_val -= 1
    n_train = max(0, n_train)
    n_val = max(0, n_val)
    n_test = max(0, n - n_train - n_val)
    return n_train, n_val, n_test

def seq_to_idx(seq: str) -> np.ndarray:
    return np.asarray([B2I.get(ch, -1) for ch in seq], dtype=np.int64)

def encode_onehot(seq: str) -> np.ndarray:
    L = len(seq)
    x = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = B2I.get(ch, -1)
        if j >= 0:
            x[j, i] = 1.0
    return x

def estimate_base_probs(seqs: List[str]) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    for s in seqs:
        for ch in s:
            j = B2I.get(ch, -1)
            if j >= 0:
                counts[j] += 1.0
    if counts.sum() <= 0:
        return np.full(4, 0.25, dtype=np.float32)
    p = counts / counts.sum()
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    return p.astype(np.float32)

def sample_random_seq(L: int, base_probs: np.ndarray, rng: np.random.RandomState) -> str:
    p = np.asarray(base_probs, dtype=np.float64).reshape(-1)
    if p.size != 4 or not np.isfinite(p).all() or p.sum() <= 0:
        p = np.full(4, 0.25, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    arr = rng.choice(list(BASES), size=L, p=p.tolist())
    return "".join(arr.tolist())

def gc_count(seq: str) -> int:
    return seq.count("G") + seq.count("C")

def basic_scalar_features(seq: str) -> np.ndarray:
    L = len(seq)
    gc = (seq.count("G") + seq.count("C")) / max(1, L)
    p = np.array([seq.count(b) / max(1, L) for b in BASES], dtype=np.float32)
    ent = float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum())
    return np.asarray([L, gc, ent], dtype=np.float32)

def kmer_hash_features(seq: str, ks: Tuple[int, ...] = (4, 5, 6), bins: int = 64) -> np.ndarray:
    feats = np.zeros((len(ks) * bins,), dtype=np.float32)
    L = len(seq)
    for ki, k in enumerate(ks):
        if L < k:
            continue
        off = ki * bins
        denom = float(L - k + 1)
        for i in range(L - k + 1):
            idx = 0
            ok = True
            for ch in seq[i:i + k]:
                j = B2I.get(ch, -1)
                if j < 0:
                    ok = False
                    break
                idx = idx * 4 + j
            if ok:
                feats[off + (idx % bins)] += 1.0
        feats[off:off + bins] /= max(1.0, denom)
    return feats.astype(np.float32)

def fcgr_features(seq: str, k: int = 4, pool: int = 2) -> np.ndarray:
    size = 2 ** k
    mat = np.zeros((size, size), dtype=np.float32)
    if len(seq) < k:
        out_size = size // pool
        return np.zeros((out_size * out_size,), dtype=np.float32)
    denom = float(len(seq) - k + 1)
    for i in range(len(seq) - k + 1):
        x = 0
        y = 0
        ok = True
        for ch in seq[i:i + k]:
            bits = CGR_BITS.get(ch)
            if bits is None:
                ok = False
                break
            xb, yb = bits
            x = (x << 1) | xb
            y = (y << 1) | yb
        if ok:
            mat[y, x] += 1.0
    mat /= max(1.0, denom)
    if pool > 1:
        assert size % pool == 0
        mat = mat.reshape(size // pool, pool, size // pool, pool).mean(axis=(1, 3))
    return mat.reshape(-1).astype(np.float32)

def parse_k_list_arg(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    vals = []
    for tok in re.split(r"[,\s]+", s):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise ValueError(f"All K values must be positive. Got: {v}")
        vals.append(v)
    vals = sorted(set(vals))
    return vals

def center_crop(seq: str, k: int) -> str:
    if len(seq) < k:
        raise ValueError(f"Cannot crop length-{len(seq)} sequence to K={k}")
    if len(seq) == k:
        return seq
    start = (len(seq) - k) // 2  # deterministic left-centered crop when parity is ambiguous
    return seq[start:start + k]

def safe_sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)

def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    sx = np.sqrt(np.sum(x ** 2))
    sy = np.sqrt(np.sum(y ** 2))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / (sx * sy))

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    return pearson_corr(rx, ry)

def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    comp_idx = np.array([3, 2, 1, 0], dtype=np.int64)
    return pwm[::-1][:, comp_idx]

def pwm_information_content(pwm: np.ndarray) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=np.float64)
    return (2.0 + np.sum(pwm * np.log2(np.clip(pwm, 1e-8, 1.0)), axis=1)).astype(np.float32)
