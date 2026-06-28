#!/usr/bin/env bash
# Reproduce the manuscript results. Run from the repo root.
# Outputs go to Shared/Results/<suffix>/.
set -euo pipefail
PY="python -m motifgate"
KLIST="7,8,9,10,11,12,13,14"

# --- Fig 1 / Tables S1-S2: per-K AP & AUROC, site-level, family and class ---
for GB in family class; do
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" \
      --run_name "fig1_${GB}"
done

# --- Table 2: ablations (one component removed at a time) at site-level ---
# Full model is the fig1 run above. Each line removes ONE component.
for GB in family class; do
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --use_moe 0            --run_name "abl_no_moe_${GB}"
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --use_se 0             --run_name "abl_no_se_${GB}"
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --use_cross_attention 0 --run_name "abl_no_xattn_${GB}"
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --use_aux_losses 0     --run_name "abl_no_aux_${GB}"
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --residual_gate_init 0.0 --run_name "abl_gate_${GB}"
  # d_model = 64 / 128 sensitivity
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --d_model 64           --run_name "abl_d64_${GB}"
  $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" --d_model 128          --run_name "abl_d128_${GB}"
done

# --- Table 3: motif-level and group-level transfer ---
for GB in family class; do
  $PY --protocol motif_heldout --group_by "$GB" --compare_k_list "$KLIST" --run_name "motif_${GB}"
  $PY --protocol group_heldout --group_by "$GB" --compare_k_list "$KLIST" --run_name "group_${GB}"
done

echo "[run_all] done. Aggregate with your figure/table scripts over Shared/Results/."
