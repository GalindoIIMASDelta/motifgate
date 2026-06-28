# MotifGate

**A PWM-aware residual learning model for transcription factor binding site (TFBS) prediction in short windows (K = 7–14 bp).**

MotifGate decomposes the prediction logit into an explicit, differentiable PWM
prior and a learned residual branch conditioned on the queried motif:

```
ℓ(x, m) = z_PWM(x, m) + α · r_θ(x, m)
P(binding | x, m) = σ(ℓ(x, m))
```

where `z_PWM` is the standardized exact PWM alignment score (forward +
reverse-complement, best offset), `r_θ` is the residual logit from the learned
encoder, and `α` is a trainable scalar gate (initialized at 0.1). This keeps the
motif contribution visible and makes the learned correction directly measurable.

This repository accompanies the manuscript *"MotifGate: a PWM-aware residual
learning model for TFBS prediction"* and contains the exact model code,
leakage-controlled evaluation protocols, hard-negative construction, and
paper-ready exporters.

---

## Repository layout

```
.
├── src/motifgate/         # the model package (split by responsibility)
│   ├── constants.py       # global constants / defaults / runtime config
│   ├── utils.py           # seeding, IO, sequence encoding, scalar features
│   ├── pwm_io.py          # TRANSFAC/JASPAR parsing + exact alignment scoring
│   ├── data.py            # examples, dedup, splits, datasets, hard negatives
│   ├── model.py           # ExactPWMConv + residual branch + build_model
│   ├── train.py           # training loop, evaluation, calibration, FDR points
│   ├── interpret.py       # integrated gradients, IG-IC, PWM reconstruction
│   ├── exports.py         # CSV/JSON exporters and summary builders
│   ├── experiment.py      # run_single_experiment orchestration
│   └── cli.py             # argument parser + main()
├── scripts/
│   ├── run_all.sh         # reproduce every table/figure
│   └── run_seeds.sh       # 25-seed stability analysis driver
├── configs/               # example configurations per experiment
├── data/
│   └── download_data.py   # fetch + prepare JASPAR 2026 / UniBind inputs
├── baselines/             # FIMO, BaMMmotif2, LS-GKM, DNABERT-2, XGBoost
├── external_validation/   # UniBind external validation (driver + utils + mapping)
├── analysis/              # rebuttal analyses: significance (bootstrap+FDR), alpha, ablations
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── REPRODUCIBILITY.md     # reproducibility notes (READ THIS for review)
└── LICENSE
```

The code is a **faithful module split of the original single-file script**: every
function/class keeps its exact implementation, so results are identical to the
original. `import motifgate` re-exports all symbols.

---

## Installation

Python ≥ 3.12.7 recommended.
numpy ≥ 1.26.4
scikit-learn ≥ 1.7.1
torch ≥ 2.9.1 # CPU build on macOS arm64 (no CUDA required)
torchaudio ≥ 2.9.1
torchvision ≥ 0.24.1

```bash
# clone, then from the repo root:
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

> **Runs on CPU.** All results were produced on CPU only (Apple M2 laptop, 8 GB
> RAM) — the model is lightweight enough that no GPU is required. A GPU is optional
> and only speeds things up; install a CUDA build of PyTorch if you have one.

---

## Data

MotifGate uses **JASPAR 2026** PFMs (priors) and binding-site sequences, with
external validation on **UniBind**. These resources are **not redistributed here**
(see their licenses). Download them manually, then prepare the expected layout with
`data/download_data.py`:

```bash
# 1) JASPAR: assemble Shared/ from the per-motif TRANSFAC files and the per-motif
#    <MA_id>.sites files that JASPAR ships (both are already in the right format):
python data/download_data.py jaspar \
    --transfac-dir jaspar_transfac/ \
    --sites-dir   jaspar_sites/ \
    --out ./Shared
#    (use --transfac FILE instead of --transfac-dir if you have one combined file)

# 2) UniBind external-validation set: TF-named FASTA -> <MA_id>.sites, K in [7,14]
python data/download_data.py unibind \
    --tar damo_fasta_per_TF.tar --jaspar ./Shared --out ./Shared_unibind
```

Sources: JASPAR (`https://jaspar.elixir.no/downloads/`, "JASPAR collections PFMs
(non-redundant)", TRANSFAC format **plus** the per-motif `.sites` files) and UniBind
(`https://unibind.uio.no/downloads/`, "FASTA files for the TFBSs per TF"). The script
produces:

```
Shared/
├── transfac/        # one TRANSFAC file per motif (PWM + tf_family/tf_class)
├── Jaspar_pwms/     # one .jaspar file per motif (optional/redundant)
└── sites/           # one <MOTIF_ID>.sites per motif (positive TFBS sequences)
```

The file name is the join key: a sites file **must** be named `<MOTIF_ID>.sites`
(e.g. `MA0002.2.sites`) matching the JASPAR matrix AC. See `data/download_data.py`
for details and the TF-name→motif mapping (it logs any unmatched files).

---

## Quick start

```bash
# Site-level, family analysis, K = 9 (single seed)
python -m motifgate \
    --wdir . \
    --protocol site_level \
    --group_by family \
    --fixed_len 9 \
    --run_name demo

# Fair matched-source comparison across K = 7..14
python -m motifgate --protocol site_level --group_by class \
    --compare_k_list 7,8,9,10,11,12,13,14 --run_name kcompare
```

Outputs are written to `Shared/Results/<suffix>/`, including `config.json`,
`summary.csv`, `all_metrics.json`, per-split manifests, calibration, IG, PWM
reconstruction, and leak-audit CSVs.

---

## Reproducing the manuscript

```bash
bash scripts/run_all.sh       # all protocols × group_by × K  (Tables 1–3, Fig 1)
bash scripts/run_seeds.sh     # 25-seed stability analysis    (seed variation)
```

See `REPRODUCIBILITY.md` for exact mappings between commands and each
table/figure, the seed list, hardware used for the reported numbers, and
important notes on negative sampling and the LR schedule.

---

## Evaluation protocols

- **site_level** — motif appears in train/val/test, but site instances are
  disjoint (generalization to unseen instances).
- **motif_heldout** — whole motifs held out (generalization to unseen motifs).
- **group_heldout** — whole families/classes held out (hardest transfer).

All protocols apply canonical reverse-complement deduplication **before**
splitting and run integrity/leak audits.

## Negatives: real TFBS only

MotifGate uses **only real TFBS** (binding sites of other transcription factors) as
negatives and **never** generates synthetic, shuffled, or mutated sequences. The target
ratio is 1:2; when the real pool cannot supply enough admissible, previously-unused
sequences, the split is built with the maximum number of real negatives available and
the realized ratio is logged (`[neg-quota] ...`) and recorded in each run summary
(`realized_pos_neg_ratio`, `n_missing_negatives`, `negative_source="real_tfbs_only"`).

---

## License & citation

Code released under the terms in [`LICENSE`](LICENSE). If you use MotifGate,
please cite the manuscript (see [`CITATION.cff`](CITATION.cff)).
