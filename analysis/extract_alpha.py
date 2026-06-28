#!/usr/bin/env python3
"""
Extract the learned residual gate (alpha) from every trained per-group MotifGate model and
summarise its distribution. The gate value is a built-in effect size: alpha -> 0 means the
exact PWM prior already suffices for that group; larger |alpha| means the learned residual
contributes more. This turns an architectural detail into an interpretable, per-family read-out.

It scans an experiment root for per-group checkpoints (best_model.pt / *.pt), reads the
'residual_gate' parameter directly from the checkpoint (no data needed), writes a tidy CSV,
and (optionally) a histogram and an alpha-vs-deltaAP scatter if a significance table is given.

Usage
-----
python analysis/extract_alpha.py \
    --exp-root Shared/Results/K13_site_level_family_last \
    --out-csv analysis_out/alpha_by_group_K13.csv \
    --hist analysis_out/alpha_hist_K13.png \
    [--significance ExternalValidation/UniBind/K13_family/significance_family.csv \
     --scatter analysis_out/alpha_vs_dAP_K13.png]
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

try:
    import torch
except Exception as e:
    sys.exit(f"PyTorch is required: {e}")


def _find_gate(obj):
    """Recursively find a tensor/scalar whose key ends with 'residual_gate'."""
    # unwrap common checkpoint containers
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.split(".")[-1] == "residual_gate":
                try:
                    return float(np.asarray(v).reshape(-1)[0])
                except Exception:
                    try:
                        return float(v.item())
                    except Exception:
                        pass
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in obj:
                g = _find_gate(obj[key])
                if g is not None:
                    return g
        # last resort: scan nested dicts
        for v in obj.values():
            if isinstance(v, dict):
                g = _find_gate(v)
                if g is not None:
                    return g
    return None


def _group_name(path, exp_root):
    rel = os.path.relpath(os.path.dirname(path), exp_root)
    parts = [p for p in rel.split(os.sep) if p not in (".", "")]
    return parts[-1] if parts else os.path.basename(os.path.dirname(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--ckpt-glob", default="**/best_model.pt",
                    help="glob (relative to exp-root) for per-group checkpoints")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--hist", default="")
    ap.add_argument("--significance", default="", help="optional significance_*.csv to merge (group,dAP)")
    ap.add_argument("--scatter", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.exp_root, args.ckpt_glob), recursive=True))
    if not paths:
        # fall back to any *.pt
        paths = sorted(glob.glob(os.path.join(args.exp_root, "**", "*.pt"), recursive=True))
    if not paths:
        sys.exit(f"No checkpoints found under {args.exp_root}")

    rows = []
    for p in paths:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(p, map_location="cpu")
        a = _find_gate(ckpt)
        if a is None:
            print(f"[warn] no residual_gate in {p}")
            continue
        rows.append((_group_name(p, args.exp_root), a, abs(a), p))

    if not rows:
        sys.exit("Found checkpoints but none contained a 'residual_gate' parameter.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "alpha", "abs_alpha", "checkpoint"])
        for r in rows:
            w.writerow(r)

    alphas = np.array([r[1] for r in rows], float)
    print(f"[alpha] groups={len(rows)}  mean={alphas.mean():+.4f}  median={np.median(alphas):+.4f}  "
          f"min={alphas.min():+.4f}  max={alphas.max():+.4f}  |alpha|<0.02: {(np.abs(alphas)<0.02).sum()}")
    print(f"[alpha] wrote {args.out_csv}")

    if args.hist:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axx = plt.subplots(figsize=(6, 4))
        axx.hist(alphas, bins=24, color="#4C3BCF", alpha=0.85, edgecolor="white")
        axx.axvline(0, color="#56536A", lw=1, ls="--")
        axx.set_xlabel(r"learned residual gate $\alpha$"); axx.set_ylabel("number of groups")
        axx.set_title(r"Distribution of the learned gate $\alpha$ across groups")
        fig.tight_layout(); os.makedirs(os.path.dirname(args.hist) or ".", exist_ok=True)
        fig.savefig(args.hist, dpi=200); print(f"[alpha] wrote {args.hist}")

    if args.significance and args.scatter:
        import pandas as pd
        sig = pd.read_csv(args.significance)
        a_df = pd.DataFrame(rows, columns=["group", "alpha", "abs_alpha", "checkpoint"])
        merged = a_df.merge(sig[["group", "dAP"]], on="group", how="inner")
        if len(merged):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axx = plt.subplots(figsize=(6, 4.4))
            axx.axhline(0, color="#8B889E", lw=1, ls="--"); axx.axvline(0, color="#8B889E", lw=1, ls="--")
            axx.scatter(merged["alpha"], merged["dAP"], s=42, color="#138A7A", edgecolor="white", zorder=3)
            axx.set_xlabel(r"learned residual gate $\alpha$")
            axx.set_ylabel(r"$\Delta$AP vs PWM prior")
            axx.set_title(r"Does a larger gate $\alpha$ track a larger residual gain?")
            fig.tight_layout(); fig.savefig(args.scatter, dpi=200)
            print(f"[alpha] wrote {args.scatter}  (merged {len(merged)} groups)")


if __name__ == "__main__":
    main()
