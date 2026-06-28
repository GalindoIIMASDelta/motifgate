# Reproducibility notes

For reviewers and for anyone reproducing the results. Records command↔result
mapping, environment, the exact 25-seed protocol, and a few implementation details.

## 1. Environment

All results were produced on **CPU only** (no GPU).

| Package      | What was used                                   | Version |
|--------------|-------------------------------------------------|--------------------|
| Python       | 3.12.7                                          | 3.12.7             |
| numpy        | 1.26.4                                          | 1.26.4.            |
| scikit-learn | 1.7.1                                           | 1.7.1.             |
| torch        | 2.9.1                                           | 2.9.1              |
| torchaudio   | 2.9.1                                           | 2.9.1              |
| torchvision  | 0.24.1                                          | 0.24.1             |
| pip          | 24.2                                            | 24.2.              |


```bash
python -m pip freeze > environment.lock.txt
# also record the interpreter:
python -V >> environment.lock.txt
```

`environment.lock.txt` is the authoritative record; `requirements.txt` only gives
floors. On Apple Silicon the PyPI `torch` wheel is the universal CPU/MPS build (its
version string has no `+cuXXX` suffix), which matches a CPU-only run.

## 2. Hardware used for ALL reported numbers

- **Machine:** Apple MacBook Pro, **M2** SoC, 8 cores, **8 GB RAM**, 128 GB SSD.
- **Accelerator:** none — **CPU only** (no CUDA, no GPU). MPS was not used.
- This is why MotifGate is presented as lightweight: the full pipeline (K = 7–14,
  50 epochs, early stopping) runs end-to-end on a consumer laptop CPU. The CPU
  runtimes in Table S5 were measured on this same machine.

Determinism: `set_seed` sets `cudnn.deterministic=True`, `benchmark=False`. On CPU,
runs are reproducible for a fixed seed.

## 3. Seed protocol (25-run stability)

The script runs **one seed per invocation**; `scripts/run_seeds.sh` loops the 25 base
seeds below to reproduce the mean ± SD values (AP SD ≈ 0.012, AUROC SD ≈ 0.009–0.011,
gate SD ≈ 0.006–0.007). The 25 base seeds (251031 + 100·i, i = 0..24) are:

```
251031, 251131, 251231, 251331, 251431, 251531, 251631, 251731, 251831, 251931,
252031, 252131, 252231, 252331, 252431, 252531, 252631, 252731, 252831, 252931,
253031, 253131, 253231, 253331, 253431
```

## 3b. Data preparation (JASPAR positives + UniBind external)

- **Positives (training/eval):** JASPAR ships per-motif `<MA_id>.sites` files; these
  are used directly as positives. Family/class come from the matching `transfac/`
  file. A `.sites` and its PWM may differ in version (e.g. `MA0001.1.sites` ↔
  `MA0001.3.transfac`); MotifGate matches them via the version-stripped base id
  (`MA0001`), so this is expected and correct.
- **External validation (UniBind):** files are named by TF only (e.g. `RUNX1.fasta`).
  `download_data.py unibind` maps each TF to its JASPAR motif id (dimer-aware, so
  `AHR` matches `AHR::ARNT`), keeps sequences with length in [7, 14], and writes
  `<MA_id>.sites` into a separate `Shared_unibind/`. UniBind does not cover every
  JASPAR family/class; unmapped TFs are skipped and listed in
  `Shared_unibind/unmatched_unibind.txt`. Report the UniBind coverage (how many
  families/classes were available) alongside the external-validation results.

## 4. Command → result mapping

| Result | Command (see `scripts/run_all.sh`) |
|--------|-------------------------------------|
| Fig. 1, Tables S1/S2 | `--compare_k_list 7,8,9,10,11,12,13,14` for `--group_by family` and `class` |
| Table 1 (baselines) | MotifGate + each script in `baselines/` on the **exported split manifests** |
| Table 2 (ablations) | ablation block in `run_all.sh` (toggles `--use_moe/--use_se/--use_cross_attention/...`) |
| Table 3 (motif/group/UniBind) | `--protocol motif_heldout`, `--protocol group_heldout`, UniBind external eval |
| Fig. 2 / Table S3 | `--export_ig 1` outputs (IG, IG-IC, PWM reconstruction) |

## 5. Negative sampling — REAL TFBS ONLY (no synthetic)

MotifGate uses **only real TFBS sequences** (binding sites of other motifs) as
negatives. It **never** generates synthetic, shuffled, or mutated sequences.

- Pool: globally deduplicated positives across all motifs/families/classes.
- Anti-leakage: candidates positive anywhere in the current group are rejected;
  negatives are not reused within a split.
- **Realized ratio:** the target is 1:2, but when the real pool cannot supply enough
  admissible, previously-unused sequences, the split is built with the maximum real
  negatives available and a `[neg-quota]` message is logged, e.g.:

  ```
  [neg-quota] split=train: the 1:2 positive-negative ratio could not be completed
  using only real TFBS negatives. Built N real negatives for P positives
  (expected 2P; missing M); realized ratio 1:r. MotifGate uses ONLY real TFBS
  negatives (no synthetic shuffling).
  ```

  The realized ratio and shortfall are recorded per split in the run summaries
  (`realized_pos_neg_ratio`, `n_missing_negatives`, `negative_source="real_tfbs_only"`).
  Report these so AP is interpreted against the realized prevalence (chance AP depends
  on the realized ratio, not a fixed 1:2).

## 6. Learning-rate schedule

Constant LR = 2e-3 with AdamW and gradient clipping (norm 1.0); no scheduler — exactly
as described in Methods 3.7 and Algorithm S2. The optional cosine+warmup scheduler is
**off by default** (`--use_cosine_warmup 0`) and was not used for the reported numbers.

## 7. PWM cap fallback

The per-motif PWM safety cap (q = 0.10 quantile of training-positive raw scores) falls
back to a group-level then global quantile when a motif has too few training positives
to estimate a stable quantile.

## 8. External validation on UniBind

The UniBind external validation lives in `external_validation/` (see its README). It
runs on the `motifgate` package (`--motifgate_py motifgate`), builds external
positives/negatives from UniBind, routes each example to the trained per-group model,
and reports PWM vs MotifGate AP/AUROC on the covered subset. Example (K=13, family):

```bash
python external_validation/motifgate_external_unibind_validation.py \
    --unibind_fasta_dir Data/UniBind_TFBS_per_TF_FASTA \
    --mapping_csv external_validation/metadata/jaspar_unibind_mapping.csv \
    --k_list 13 --outdir ExternalValidation/UniBind/K13_family \
    --motifgate_py motifgate --wdir . \
    --motifgate_exp_root Shared/Results/<K13_family_experiment> \
    --model_group_by family --device cpu --batch_size 512 --max_per_tf 200
```

When reporting, state: the covered-subset coverage fraction, the single K used, and
the `--max_per_tf` cap; and give per-group results (external behaviour is
family-dependent — the residual branch helps some families and regresses others,
notably heterogeneous zinc-finger families). `--model_group_by` must match the
experiment root (family↔family, class↔class).

