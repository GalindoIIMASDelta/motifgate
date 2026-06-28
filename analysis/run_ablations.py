#!/usr/bin/env python3
"""
Ablation runner for MotifGate. Re-runs the training/evaluation experiment with one component
turned off at a time, using the package's own CLI flags, and collects test AP/AUROC into a
single table. Answers the reviewer request for a clean ablation table showing which components
earn their place in a ~37k-parameter model.

Each ablation is a set of CLI overrides applied on top of your base command. The base command
is the same one you use for a normal run (point --wdir at the dir containing Shared/).

Ablations
---------
  full                 reference run (all components on, gate learned)
  prior_only           model_arch=pwmconv_exact         (exact PWM baseline-as-model)
  residual_only        use_pwm=0                         (no PWM prior term; learned branch only)
  gate_fixed_0         residual_gate_init=0  + freeze    (alpha pinned to 0)
  gate_fixed_0p1       residual_gate_init=0.1 + freeze   (alpha pinned to 0.1)
  no_cross_attention   use_cross_attention=0
  no_moe               use_moe=0
  no_se                use_se=0
  no_aux_losses        use_aux_losses=0 lambda_prior=0 lambda_match=0
  no_rc_aug            rc_augmentation=0
  no_label_smoothing   label_smoothing=0

Usage
-----
python analysis/run_ablations.py \
    --wdir . --protocol site_level --group_by family --fixed_len 13 --device cpu --seed 251031 \
    --out-root Ablations/K13 --ablations all
  # add --dry-run first to print the commands without executing.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys

ABLATIONS = {
    "full":               [],
    "prior_only":         ["--model_arch", "pwmconv_exact"],
    "residual_only":      ["--use_pwm", "0"],
    "gate_fixed_0":       ["--residual_gate_init", "0", "--freeze_residual_gate", "1"],
    "gate_fixed_0p1":     ["--residual_gate_init", "0.1", "--freeze_residual_gate", "1"],
    "no_cross_attention": ["--use_cross_attention", "0"],
    "no_moe":             ["--use_moe", "0"],
    "no_se":              ["--use_se", "0"],
    "no_aux_losses":      ["--use_aux_losses", "0", "--lambda_prior", "0", "--lambda_match", "0"],
    "no_rc_aug":          ["--rc_augmentation", "0"],
    "no_label_smoothing": ["--label_smoothing", "0"],
}


def collect_metrics(results_root):
    """Best-effort: pull test AP/AUROC from summary.csv or all_metrics.json."""
    out = {}
    sm = os.path.join(results_root, "summary.csv")
    if os.path.exists(sm):
        with open(sm) as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[-1]  # global/last row
            for k, v in r.items():
                kl = k.lower()
                if ("ap" in kl or "auc" in kl or "auroc" in kl) and ("test" in kl or "global" in kl or kl in ("ap", "auc", "auroc")):
                    out[k] = v
            if not out:  # fallback: keep all AP/AUC-ish columns
                out = {k: v for k, v in r.items() if "ap" in k.lower() or "auc" in k.lower()}
    if not out:
        jm = os.path.join(results_root, "all_metrics.json")
        if os.path.exists(jm):
            try:
                data = json.load(open(jm))
                rec = data[-1] if isinstance(data, list) and data else data
                for k, v in (rec.items() if isinstance(rec, dict) else []):
                    if isinstance(v, (int, float)) and ("ap" in k.lower() or "auc" in k.lower()):
                        out[k] = v
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdir", default=".")
    ap.add_argument("--protocol", default="site_level")
    ap.add_argument("--group_by", default="family")
    ap.add_argument("--fixed_len", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=251031)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--module", default="motifgate", help="package module invoked as python -m <module>")
    ap.add_argument("--extra", default="", help="extra args appended to every run, quoted")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ablations", default="all", help="'all' or comma-separated names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = list(ABLATIONS) if args.ablations == "all" else [a.strip() for a in args.ablations.split(",")]
    bad = [n for n in names if n not in ABLATIONS]
    if bad:
        sys.exit(f"Unknown ablations: {bad}. Choices: {list(ABLATIONS)}")

    base = [args.python, "-m", args.module,
            "--wdir", args.wdir, "--protocol", args.protocol, "--group_by", args.group_by,
            "--fixed_len", str(args.fixed_len), "--device", args.device, "--seed", str(args.seed)]
    extra = args.extra.split() if args.extra else []

    os.makedirs(args.out_root, exist_ok=True)
    summary_path = os.path.join(args.out_root, "ablation_summary.csv")
    collected = []
    for name in names:
        rroot = os.path.join(args.out_root, name)
        cmd = base + extra + ABLATIONS[name] + ["--results_root", rroot]
        print(f"\n=== ablation: {name} ===\n{' '.join(cmd)}")
        if args.dry_run:
            continue
        os.makedirs(rroot, exist_ok=True)
        rc = subprocess.run(cmd).returncode
        metrics = collect_metrics(rroot) if rc == 0 else {}
        row = {"ablation": name, "returncode": rc, "overrides": " ".join(ABLATIONS[name]) or "(none)",
               "results_root": rroot, **metrics}
        collected.append(row)
        # write incrementally
        keys = sorted({k for r in collected for k in r})
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(collected)
        print(f"[ablation:{name}] rc={rc} metrics={metrics}")

    if args.dry_run:
        print(f"\n[dry-run] {len(names)} ablations planned. Re-run without --dry-run to execute.")
    else:
        print(f"\n[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
