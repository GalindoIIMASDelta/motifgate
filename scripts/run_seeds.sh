#!/usr/bin/env bash
# 25-seed stability analysis (seeds = 251031 + 100*i, i = 0..24).
set -euo pipefail
PY="python -m motifgate"
KLIST="7,8,9,10,11,12,13,14"

SEEDS=(251031 251131 251231 251331 251431 251531 251631 251731 251831 251931 \
       252031 252131 252231 252331 252431 252531 252631 252731 252831 252931 \
       253031 253131 253231 253331 253431)

if [ "${#SEEDS[@]}" -ne 25 ]; then echo "Expected 25 seeds, got ${#SEEDS[@]}" >&2; exit 1; fi

for GB in family class; do
  for S in "${SEEDS[@]}"; do
    $PY --protocol site_level --group_by "$GB" --compare_k_list "$KLIST" \
        --seed "$S" --run_name "seed_${GB}_${S}"
  done
done
echo "[run_seeds] done. Aggregate summary.csv across runs to get mean +/- SD."
