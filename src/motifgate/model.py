from __future__ import annotations

"""MotifGate architecture: exact differentiable PWM prior + learned residual branch."""

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

from .constants import (LR_MIN_FACTOR, RESIDUAL_GATE_INIT, USE_SE_BLOCK, WARMUP_EPOCHS_FRAC)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (Hu et al. 2018, adapted for 1-D).
    After multi-scale CNN concat, SE learns per-channel recalibration weights
    via global-average-pool → FC → ReLU → FC → Sigmoid."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        w = x.mean(dim=2)            # (B, C) — global average pool
        w = self.fc(w).unsqueeze(2)   # (B, C, 1)
        return x * w

def make_cosine_warmup_scheduler(optimizer, epochs: int, steps_per_epoch: int):
    """Cosine annealing with linear warmup (restored from v30).
    Prevents early instability (warmup) then settles into flatter minimum (cosine)."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(WARMUP_EPOCHS_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(LR_MIN_FACTOR, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def reverse_complement_onehot_batch(X: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a batch of one-hot encoded sequences.
    X: (B, 4, L) where channels are A,C,G,T → returns (B, 4, L) with T,G,C,A reversed."""
    # Swap A↔T (0↔3), C↔G (1↔2), then reverse along L dimension
    return X[:, [3, 2, 1, 0], :].flip(dims=[2])

class AttentivePool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        att = self.score(x).squeeze(-1)
        att = att.masked_fill(pad_mask, -1e9)
        w = torch.softmax(att, dim=1)
        return torch.sum(w.unsqueeze(-1) * x, dim=1)

class LightMoE(nn.Module):
    def __init__(self, d_model: int, num_experts: int, dropout: float):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        weights = torch.softmax(router_logits, dim=-1)
        expert_out = torch.stack([ex(x) for ex in self.experts], dim=1)
        out = torch.sum(weights.unsqueeze(-1) * expert_out, dim=1)
        return out, router_logits, weights

def reverse_complement_onehot_torch(x: torch.Tensor) -> torch.Tensor:
    comp_idx = torch.tensor([3, 2, 1, 0], dtype=torch.long, device=x.device)
    return x.index_select(0, comp_idx).flip(-1)

class ExactPWMConv(nn.Module):
    """
    Differentiable exact scorer that mirrors best_alignment_score():
    - uses log PWM
    - checks forward and reverse-complement sequence
    - takes the best valid offset
    - supports both PWM-on-seq and seq-on-PWM when lengths differ
    """

    def __init__(self, prior_mu: float = 0.0, prior_sd: float = 1.0):
        super().__init__()
        self.register_buffer("prior_mu", torch.tensor(float(prior_mu), dtype=torch.float32))
        self.register_buffer("prior_sd", torch.tensor(float(max(prior_sd, 1e-6)), dtype=torch.float32))

    @staticmethod
    def _score_windows_seq_on_pwm(x_pos: torch.Tensor, log_pwm: torch.Tensor) -> torch.Tensor:
        # x_pos: [Ls,4], log_pwm: [Lp,4], with Ls <= Lp
        Ls = int(x_pos.shape[0])
        Lp = int(log_pwm.shape[0])
        amb = torch.clamp(1.0 - x_pos.sum(dim=1), min=0.0, max=1.0).sum() * math.log(0.25)
        wins = torch.stack([log_pwm[off:off + Ls] for off in range(Lp - Ls + 1)], dim=0)  # [nwin,Ls,4]
        return (wins * x_pos.unsqueeze(0)).sum(dim=(1, 2)) + amb

    @staticmethod
    def _score_windows_pwm_on_seq(x: torch.Tensor, log_pwm: torch.Tensor) -> torch.Tensor:
        # x: [4,Ls], log_pwm: [Lp,4], with Ls >= Lp
        Lp = int(log_pwm.shape[0])
        patches = x.unfold(-1, Lp, 1).permute(1, 2, 0)  # [nwin,Lp,4]
        amb = torch.clamp(1.0 - patches.sum(dim=2), min=0.0, max=1.0).sum(dim=1) * math.log(0.25)
        return (patches * log_pwm.unsqueeze(0)).sum(dim=(1, 2)) + amb

    def _score_one_orientation(self, x: torch.Tensor, log_pwm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ls = int(x.shape[1])
        Lp = int(log_pwm.shape[0])
        if Ls >= Lp:
            scores = self._score_windows_pwm_on_seq(x, log_pwm)
            best_score, best_off = torch.max(scores, dim=0)
            mode_pwm_on_seq = torch.tensor(1, dtype=torch.int64, device=x.device)
        else:
            scores = self._score_windows_seq_on_pwm(x.transpose(0, 1), log_pwm)
            best_score, best_off = torch.max(scores, dim=0)
            mode_pwm_on_seq = torch.tensor(0, dtype=torch.int64, device=x.device)
        return best_score, best_off.to(torch.int64), mode_pwm_on_seq

    def forward(self, X: torch.Tensor, seq_pad: torch.Tensor, pwm: torch.Tensor, pwm_pad: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = int(X.shape[0])
        raw_scores = []
        best_offsets = []
        best_rev = []
        best_mode = []

        for i in range(B):
            Ls = int((~seq_pad[i]).sum().item())
            Lp = int((~pwm_pad[i]).sum().item())
            if Ls <= 0 or Lp <= 0:
                raw_scores.append(X.new_tensor(0.0))
                best_offsets.append(torch.tensor(0, dtype=torch.int64, device=X.device))
                best_rev.append(torch.tensor(0, dtype=torch.int64, device=X.device))
                best_mode.append(torch.tensor(1, dtype=torch.int64, device=X.device))
                continue

            x = X[i, :, :Ls]
            p = torch.clamp(pwm[i, :Lp], 1e-8, 1.0)
            log_pwm = torch.log(p)

            s_f, off_f, mode_f = self._score_one_orientation(x, log_pwm)
            x_rc = reverse_complement_onehot_torch(x)
            s_r, off_r, mode_r = self._score_one_orientation(x_rc, log_pwm)

            choose_rev = (s_r > s_f)
            raw = torch.where(choose_rev, s_r, s_f)
            off = torch.where(choose_rev, off_r, off_f)
            mode = torch.where(choose_rev, mode_r, mode_f)
            rev = choose_rev.to(torch.int64)

            raw_scores.append(raw)
            best_offsets.append(off)
            best_rev.append(rev)
            best_mode.append(mode)

        raw = torch.stack(raw_scores, dim=0).view(B, 1)
        z = (raw - self.prior_mu) / (self.prior_sd + 1e-6)
        return {
            "baseline_raw": raw,
            "baseline_z": z,
            "best_offset": torch.stack(best_offsets, dim=0).view(B, 1),
            "best_rev": torch.stack(best_rev, dim=0).view(B, 1),
            "best_mode_pwm_on_seq": torch.stack(best_mode, dim=0).view(B, 1),
        }

class CleanTFBSHybrid(nn.Module):
    arch_name = "clean_hybrid"

    def __init__(
        self,
        seq_channels: int,
        feat_dim: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        num_experts: int,
        use_pwm: bool,
        use_cross_attention: bool,
        use_moe: bool,
        use_se: bool = True,       # Squeeze-and-Excitation
    ):
        super().__init__()
        self.use_pwm = bool(use_pwm)
        self.use_cross_attention = bool(use_cross_attention and use_pwm)
        self.use_moe = bool(use_moe)

        cnn_ch = 48
        self.seq_conv3 = nn.Conv1d(seq_channels, cnn_ch, kernel_size=3, padding=1)
        self.seq_conv5 = nn.Conv1d(seq_channels, cnn_ch, kernel_size=5, padding=2)
        self.seq_conv7 = nn.Conv1d(seq_channels, cnn_ch, kernel_size=7, padding=3)
        cat_ch = cnn_ch * 3  # 144

        # optional SE recalibration after CNN concat
        self.se = SqueezeExcitation(cat_ch, reduction=4) if use_se else None

        self.seq_proj = nn.Conv1d(cat_ch, d_model, kernel_size=1)
        self.seq_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.seq_pool = AttentivePool(d_model)

        if self.use_pwm:
            self.pwm_proj = nn.Linear(4, d_model)
            self.pwm_ln = nn.LayerNorm(d_model)
            self.pwm_pool = AttentivePool(d_model)
            if self.use_cross_attention:
                self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                self.cross_ln = nn.LayerNorm(d_model)

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        fuse_in = d_model * (3 if self.use_pwm else 2)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        if self.use_moe:
            self.moe = LightMoE(d_model, num_experts=num_experts, dropout=dropout)

        cls_hidden = max(64, d_model * 2)  # proportional to d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1),
        )
        self.prior_head = nn.Linear(d_model, 1)

    def forward(
        self,
        X: torch.Tensor,
        seq_pad: torch.Tensor,
        feat: torch.Tensor,
        pwm: torch.Tensor,
        pwm_pad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z = torch.cat([self.seq_conv3(X), self.seq_conv5(X), self.seq_conv7(X)], dim=1)
        z = self.act(z)
        z = self.dropout(z)
        # SE recalibration
        if self.se is not None:
            z = self.se(z)
        z = self.seq_proj(z).transpose(1, 2)
        z = self.seq_ln(z)

        if self.use_pwm:
            p = self.pwm_proj(pwm)
            p = self.pwm_ln(p)
            if self.use_cross_attention:
                cross, _ = self.cross_attn(z, p, p, key_padding_mask=pwm_pad, need_weights=False)
                z = self.cross_ln(z + cross)
            pwm_emb = self.pwm_pool(p, pwm_pad)
        else:
            pwm_emb = None

        seq_emb = self.seq_pool(z, seq_pad)
        feat_emb = self.feat_mlp(feat)

        if self.use_pwm:
            fused = self.fuse(torch.cat([seq_emb, pwm_emb, feat_emb], dim=1))
        else:
            fused = self.fuse(torch.cat([seq_emb, feat_emb], dim=1))

        router_logits = None
        router_weights = None
        if self.use_moe:
            moe_out, router_logits, router_weights = self.moe(fused)
            fused = fused + moe_out

        logit = self.classifier(fused)
        prior_hat = self.prior_head(seq_emb)

        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "seq_emb": seq_emb,
            "pwm_emb": pwm_emb,
            "prior_hat": prior_hat,
            "router_logits": router_logits,
            "router_weights": router_weights,
            "baseline_component": None,
            "baseline_raw": None,
            "residual_component": None,
            "residual_raw": None,
        }

class PWMConvExactModel(nn.Module):
    arch_name = "pwmconv_exact"

    def __init__(self, prior_mu: float, prior_sd: float):
        super().__init__()
        self.exact_pwmconv = ExactPWMConv(prior_mu=prior_mu, prior_sd=prior_sd)

    def forward(
        self,
        X: torch.Tensor,
        seq_pad: torch.Tensor,
        feat: torch.Tensor,
        pwm: torch.Tensor,
        pwm_pad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = self.exact_pwmconv(X, seq_pad, pwm, pwm_pad)
        logit = base["baseline_z"]
        zeros = torch.zeros((X.shape[0], 1), dtype=X.dtype, device=X.device)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "seq_emb": zeros,
            "pwm_emb": None,
            "prior_hat": logit,
            "router_logits": None,
            "router_weights": None,
            "baseline_component": base["baseline_z"],
            "baseline_raw": base["baseline_raw"],
            "residual_component": zeros,
            "residual_raw": zeros,
        }

class PWMConvResidualHybrid(CleanTFBSHybrid):
    arch_name = "pwmconv_residual"

    def __init__(
        self,
        seq_channels: int,
        feat_dim: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        num_experts: int,
        use_pwm: bool,
        use_cross_attention: bool,
        use_moe: bool,
        prior_mu: float,
        prior_sd: float,
        residual_gate_init: float = 0.1,  # default 0.1 (was 0.0)
        use_se: bool = True,
    ):
        super().__init__(
            seq_channels=seq_channels,
            feat_dim=feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            num_experts=num_experts,
            use_pwm=use_pwm,
            use_cross_attention=use_cross_attention,
            use_moe=use_moe,
            use_se=use_se,
        )
        self.exact_pwmconv = ExactPWMConv(prior_mu=prior_mu, prior_sd=prior_sd)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init), dtype=torch.float32))

    def forward(
        self,
        X: torch.Tensor,
        seq_pad: torch.Tensor,
        feat: torch.Tensor,
        pwm: torch.Tensor,
        pwm_pad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = self.exact_pwmconv(X, seq_pad, pwm, pwm_pad)
        out = super().forward(X, seq_pad, feat, pwm, pwm_pad)
        residual_raw = out["logit"]
        residual_component = self.residual_gate * residual_raw
        logit = base["baseline_z"] + residual_component
        out["logit"] = logit
        out["prob"] = torch.sigmoid(logit)
        out["baseline_component"] = base["baseline_z"]
        out["baseline_raw"] = base["baseline_raw"]
        out["residual_component"] = residual_component
        out["residual_raw"] = residual_raw
        return out

def build_model(
    args,
    feat_dim: int,
    prior_mu: float,
    prior_sd: float,
) -> nn.Module:
    arch = str(getattr(args, "model_arch", "pwmconv_residual")).strip().lower()
    use_se = bool(getattr(args, "use_se", USE_SE_BLOCK))
    if arch == "clean_hybrid":
        return CleanTFBSHybrid(
            seq_channels=4,
            feat_dim=feat_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
            num_experts=args.num_experts,
            use_pwm=bool(args.use_pwm),
            use_cross_attention=bool(args.use_cross_attention),
            use_moe=bool(args.use_moe),
            use_se=use_se,
        )
    if arch == "pwmconv_exact":
        return PWMConvExactModel(prior_mu=prior_mu, prior_sd=prior_sd)
    if arch == "pwmconv_residual":
        model = PWMConvResidualHybrid(
            seq_channels=4,
            feat_dim=feat_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
            num_experts=args.num_experts,
            use_pwm=bool(args.use_pwm),
            use_cross_attention=bool(args.use_cross_attention),
            use_moe=bool(args.use_moe),
            prior_mu=prior_mu,
            prior_sd=prior_sd,
            residual_gate_init=float(getattr(args, "residual_gate_init", RESIDUAL_GATE_INIT)),
            use_se=use_se,
        )
        if bool(getattr(args, "freeze_residual_gate", 0)):
            # Ablation: keep the residual gate fixed at its init value (not trained).
            model.residual_gate.requires_grad_(False)
        return model
    raise ValueError(f"Unsupported model_arch: {arch}")

def seq_pwm_match_loss(seq_emb: torch.Tensor, pwm_emb: Optional[torch.Tensor], y: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    if pwm_emb is None:
        return torch.tensor(0.0, device=seq_emb.device)
    sim = F.cosine_similarity(seq_emb, pwm_emb, dim=1)
    t = (2.0 * y.view(-1) - 1.0)
    return F.softplus(-alpha * t * sim).mean()
