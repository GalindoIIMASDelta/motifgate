#!/usr/bin/env python3
"""
Paired, motif-clustered bootstrap with Benjamini-Hochberg FDR for MotifGate vs a baseline.

Answers the reviewer request: report significance for MotifGate-vs-PWM (and vs any baseline)
with paired bootstrap confidence intervals over motifs/sites and multiple-comparison control
across families (and, optionally, across K).

Statistical design
------------------
* Effect = Delta AP (and Delta AUROC) = metric(model) - metric(baseline), evaluated on the
  SAME examples (paired).
* Resampling unit = motif (cluster bootstrap), because sites within a motif are correlated;
  resampling individual sites would understate the variance. For each bootstrap replicate we
  resample motifs with replacement and pool all their sites.
* Per-family and global 95% CIs come from the 2.5/97.5 percentiles of the replicate deltas.
* Two-sided bootstrap p-value per family: p = 2 * min(P(delta<=0), P(delta>=0)).
* Multiple comparisons across families are controlled with Benjamini-Hochberg FDR (q-values).

Inputs (the files the external-validation pipeline already emits)
---------------------------------------------------------------
--examples     external_examples.csv     (needs: example_id, label, score_pwm_raw,
                                           motif_id, tf_family, tf_class)
--predictions  motifgate_..._predictions.csv  (needs: example_id, prob)   [MotifGate scores]
  (or pass --merged ONE csv that already contains label, group, motif, baseline & model scores)

Usage
-----
python analysis/stats_significance.py \
    --examples ExternalValidation/UniBind/K13_family/external_examples.csv \
    --predictions ExternalValidation/UniBind/K13_family/motifgate_external_predictions_by_group_models.csv \
    --group-by family --n-boot 2000 --k-tag K13 \
    --out ExternalValidation/UniBind/K13_family/significance_family.csv
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _metrics(y, s):
    """AP and AUROC; return (nan, nan) if a class is missing."""
    y = np.asarray(y); s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def _cluster_bootstrap(df, base_col, model_col, n_boot, rng):
    """Paired motif-clustered bootstrap. Returns arrays of dAP, dAUROC replicates."""
    motifs = df["motif_id"].to_numpy()
    uniq = np.unique(motifs)
    # pre-index rows per motif for speed
    idx_by_motif = {m: np.where(motifs == m)[0] for m in uniq}
    y = df["label"].to_numpy()
    sb = df[base_col].to_numpy(float)
    sm = df[model_col].to_numpy(float)
    dap, dauc = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_motif[m] for m in pick])
        ap_b, au_b = _metrics(y[rows], sb[rows])
        ap_m, au_m = _metrics(y[rows], sm[rows])
        if np.isfinite(ap_b) and np.isfinite(ap_m):
            dap.append(ap_m - ap_b); dauc.append(au_m - au_b)
    return np.array(dap), np.array(dauc)


def _two_sided_p(deltas):
    if len(deltas) == 0:
        return np.nan
    p_le = np.mean(deltas <= 0); p_ge = np.mean(deltas >= 0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _bh_fdr(pvals):
    """Benjamini-Hochberg q-values. NaNs are ignored (returned as NaN)."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    m = len(ok)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = p[i] * m / k
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples")
    ap.add_argument("--predictions")
    ap.add_argument("--merged", help="single CSV alternative to --examples/--predictions")
    ap.add_argument("--group-by", default="family", choices=["family", "class"])
    ap.add_argument("--baseline-col", default="score_pwm_raw")
    ap.add_argument("--model-col", default="prob")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--motif-col", default="motif_id")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=251031)
    ap.add_argument("--k-tag", default="", help="optional label (e.g. K13) carried into the output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- load & merge ----
    if args.merged:
        df = pd.read_csv(args.merged)
    else:
        if not (args.examples and args.predictions):
            sys.exit("Provide either --merged or both --examples and --predictions.")
        ex = pd.read_csv(args.examples)
        pr = pd.read_csv(args.predictions)
        key = "example_id" if "example_id" in ex.columns and "example_id" in pr.columns else None
        if key is None:
            sys.exit("Could not find a common 'example_id' column to join examples and predictions.")
        prob_col = next((c for c in ["prob", "score", "prediction", "prob_calibrated"] if c in pr.columns), None)
        if prob_col is None:
            sys.exit("Predictions file has no prob/score/prediction column.")
        pr = pr.rename(columns={prob_col: args.model_col})
        df = ex.merge(pr[[key, args.model_col]], on=key, how="inner")

    gcol = "tf_family" if args.group_by == "family" else "tf_class"
    for need in (args.label_col, args.baseline_col, args.model_col, args.motif_col, gcol):
        if need not in df.columns:
            sys.exit(f"Missing required column: {need}. Found: {list(df.columns)}")
    df = df.rename(columns={args.label_col: "label", args.motif_col: "motif_id"})
    df["label"] = df["label"].astype(int)
    df = df[np.isfinite(df[args.baseline_col]) & np.isfinite(df[args.model_col])].copy()

    # ---- per-group ----
    rows = []
    for g, sub in df.groupby(gcol):
        if sub["label"].sum() == 0 or sub["label"].sum() == len(sub):
            continue
        ap_b, au_b = _metrics(sub["label"], sub[args.baseline_col])
        ap_m, au_m = _metrics(sub["label"], sub[args.model_col])
        dap, dauc = _cluster_bootstrap(sub, args.baseline_col, args.model_col, args.n_boot, rng)
        rows.append({
            "k": args.k_tag, "group_by": args.group_by, "group": g,
            "n_motifs": sub["motif_id"].nunique(), "n_pos": int(sub["label"].sum()),
            "n_neg": int((sub["label"] == 0).sum()),
            "AP_baseline": ap_b, "AP_model": ap_m, "dAP": ap_m - ap_b,
            "dAP_lo": np.nanpercentile(dap, 2.5) if len(dap) else np.nan,
            "dAP_hi": np.nanpercentile(dap, 97.5) if len(dap) else np.nan,
            "dAP_p": _two_sided_p(dap),
            "AUROC_baseline": au_b, "AUROC_model": au_m, "dAUROC": au_m - au_b,
            "dAUROC_lo": np.nanpercentile(dauc, 2.5) if len(dauc) else np.nan,
            "dAUROC_hi": np.nanpercentile(dauc, 97.5) if len(dauc) else np.nan,
            "dAUROC_p": _two_sided_p(dauc),
        })
    res = pd.DataFrame(rows).sort_values("dAP", ascending=False).reset_index(drop=True)
    res["dAP_q_BH"] = _bh_fdr(res["dAP_p"].to_numpy())
    res["dAUROC_q_BH"] = _bh_fdr(res["dAUROC_p"].to_numpy())

    # ---- global (resample motifs across all groups) ----
    ap_b, au_b = _metrics(df["label"], df[args.baseline_col])
    ap_m, au_m = _metrics(df["label"], df[args.model_col])
    gdap, gdauc = _cluster_bootstrap(df, args.baseline_col, args.model_col, args.n_boot, rng)
    glob = {
        "k": args.k_tag, "group_by": args.group_by, "group": "__GLOBAL__",
        "n_motifs": df["motif_id"].nunique(), "n_pos": int(df["label"].sum()),
        "n_neg": int((df["label"] == 0).sum()),
        "AP_baseline": ap_b, "AP_model": ap_m, "dAP": ap_m - ap_b,
        "dAP_lo": np.nanpercentile(gdap, 2.5), "dAP_hi": np.nanpercentile(gdap, 97.5),
        "dAP_p": _two_sided_p(gdap), "dAP_q_BH": np.nan,
        "AUROC_baseline": au_b, "AUROC_model": au_m, "dAUROC": au_m - au_b,
        "dAUROC_lo": np.nanpercentile(gdauc, 2.5), "dAUROC_hi": np.nanpercentile(gdauc, 97.5),
        "dAUROC_p": _two_sided_p(gdauc), "dAUROC_q_BH": np.nan,
    }
    out = pd.concat([res, pd.DataFrame([glob])], ignore_index=True)
    out.to_csv(args.out, index=False)

    # ---- console summary ----
    n_sig = int((out["dAP_q_BH"] < 0.05).sum())
    n_pos = int(((out["dAP_q_BH"] < 0.05) & (out["dAP"] > 0)).sum())
    med = np.nanmedian(res["dAP"]) if len(res) else float("nan")
    print(f"[stats] groups={len(res)}  median dAP={med:+.4f}  "
          f"families with q<0.05: {n_sig} ({n_pos} positive)")
    print(f"[stats] GLOBAL dAP={glob['dAP']:+.4f} "
          f"[{glob['dAP_lo']:+.4f}, {glob['dAP_hi']:+.4f}] p={glob['dAP_p']:.3g} | "
          f"dAUROC={glob['dAUROC']:+.4f} [{glob['dAUROC_lo']:+.4f}, {glob['dAUROC_hi']:+.4f}] "
          f"p={glob['dAUROC_p']:.3g}")
    print(f"[stats] wrote {args.out}")


if __name__ == "__main__":
    main()
