# analysis/ — post-hoc analyses for the rebuttal

Three scripts that run on the outputs you already produce (no retraining needed except the ablation
runner). Each maps to a specific reviewer concern.

| Script | Answers | What it does |
|---|---|---|
| `stats_significance.py` | "report significance with paired bootstrap CIs and multiple-comparison control" (R1·3) | Paired, **motif-clustered** bootstrap of ΔAP/ΔAUROC with 95% CIs, two-sided p-values, and **Benjamini–Hochberg** q-values, per family and global. Consumes `external_examples.csv` + the predictions CSV. |
| `extract_alpha.py` | "report α's learned distribution" (R1·minor) and "what does interpretability teach?" (R2·3) | Reads the residual gate from every per-group checkpoint, writes a CSV + histogram, and (optionally) an α-vs-ΔAP scatter. Supports the **gate-as-diagnostic** framing. |
| `run_ablations.py` | "provide a clean ablation table" (R1·4) | Re-runs training with one component removed/fixed at a time (prior-only, residual-only, α fixed 0/0.1, no cross-attn, no MoE, no SE, no aux losses, no RC-aug, no label smoothing) and collects test AP/AUROC. |

## Quick start (K = 13, family protocol)

```bash
# 1) significance table (runs in seconds on existing CSVs)
python analysis/stats_significance.py \
  --examples    ExternalValidation/UniBind/K13_family/external_examples.csv \
  --predictions ExternalValidation/UniBind/K13_family/motifgate_external_predictions_by_group_models.csv \
  --group-by family --n-boot 2000 --k-tag K13 \
  --out analysis_out/significance_family_K13.csv

# 2) learned gate distribution (reads trained group checkpoints)
python analysis/extract_alpha.py \
  --exp-root Shared/Results/<K13_family_experiment> \
  --out-csv analysis_out/alpha_by_group_K13.csv \
  --hist    analysis_out/alpha_hist_K13.png \
  --significance analysis_out/significance_family_K13.csv \
  --scatter analysis_out/alpha_vs_dAP_K13.png

# 3) ablations (re-trains; preview the commands first with --dry-run)
python analysis/run_ablations.py \
  --wdir . --protocol site_level --group_by family --fixed_len 13 --device cpu --seed 251031 \
  --out-root Ablations/K13 --ablations all --dry-run
```

Notes:
- `run_ablations.py` uses the package CLI flag `--freeze_residual_gate` (added for the α-fixed
  ablations) to pin the gate at its init value without training it.
- For multiple K, run the significance script per K and pool the p-values before Benjamini–Hochberg
  if you want one FDR across families × K.
- `residual_only` (`--use_pwm 0`) removes the PWM prior term; `prior_only`
  (`--model_arch pwmconv_exact`) is the exact baseline-as-model.
