# External validation on UniBind

This folder reproduces the **external validation** of MotifGate on UniBind TFBS
(the held-out evaluation that complements the in-distribution JASPAR results). It
builds external positives/negatives from UniBind, routes each example to the trained
per-group MotifGate model, and reports PWM-prior vs MotifGate metrics on the subset
of groups that have a trained model ("covered subset").

## Files

- `motifgate_external_unibind_validation.py` — the driver.
- `motifgate_validation_utils.py` — helpers (FASTA/mapping IO, PWM scoring, model
  inference, metrics). It runs on the installed **`motifgate` package** by default
  (`--motifgate_py motifgate`); a path to the original single-file script is still
  accepted for backward compatibility.
- `metadata/jaspar_unibind_mapping.csv` — TF→motif metadata
  (`tf_symbol,motif_id,tf_family,tf_class,species,tax_group`). Used as a fallback for
  family/class; the motif id is taken from the UniBind file name when present.

## Inputs

**UniBind** — "FASTA files for the TFBSs per TF"
(`https://unibind.uio.no/downloads/`). Recursive TF-folder layout:

```
UniBind_TFBS_per_TF_FASTA/
├── AR/   ERP001226.LNCaP_prostate_carcinoma.AR.MA0007.3.damo.fa
├── CTCF/ ...
└── ...
```

The TF symbol is inferred from the parent folder and the JASPAR motif id from the
file name (`MA\d+\.\d+`). The trained MotifGate experiment root (one model per
class/family) provides `config.json` plus per-group subdirectories with
`best_model.pt` and `split_manifest.csv`, together with the JASPAR PWMs under
`<wdir>/Shared/transfac` and `<wdir>/Shared/Jaspar_pwms` (see `data/download_data.py`).

## Run (one K per trained experiment root)

```bash
python external_validation/motifgate_external_unibind_validation.py \
    --unibind_fasta_dir Data/UniBind_TFBS_per_TF_FASTA \
    --mapping_csv       external_validation/metadata/jaspar_unibind_mapping.csv \
    --k_list 13 \
    --outdir ExternalValidation/UniBind/K13_family \
    --motifgate_py motifgate \
    --wdir . \
    --motifgate_exp_root Shared/Results/K13_site_level_family_last \
    --model_group_by family \
    --device cpu --batch_size 512 --max_per_tf 200
```

Notes:
- `--model_group_by` must match the experiment root: a **family** root pairs with
  `--model_group_by family`, a **class** root with `--model_group_by class`. Do not
  cross them.
- Use exactly one K per run, matching the trained experiment's `fixed_len`.
- Negatives are **real UniBind sites of other TFs** (GC-matched, no synthetic),
  consistent with the training-time policy.

## Outputs

- `external_examples.csv`, `negative_sampling_audit.csv`, `unibind_fasta_scan_audit.csv`
- `external_group_coverage_audit.csv`, `dataset_summary.json`
- `motifgate_external_predictions_by_group_models.csv`
- `external_metrics.csv` and `external_metrics_global.csv` (PWM vs MotifGate, AP/AUROC)

## Reporting caveats (important for the paper)

The metrics are computed on the **covered subset only** (groups with a trained
model) and for a **single K**, with `--max_per_tf` capping sequences per TF. Report
the coverage fraction and these caps alongside the numbers, and present per-group
results (the residual branch helps some families and hurts others on external data).
