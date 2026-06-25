# PLV-EEGNet: Phase-Locking-Value Ablation with a Same-Marginal Control Benchmark

Reproducible code and result deposit for the manuscript submitted to the
*Journal of Neuroscience Methods*. This bundle is anonymized for single-blind
review.

## Overview

This study asks whether a phase-locking-value (PLV) connectivity branch added to
EEGNet improves motor-imagery decoding **because it carries genuine inter-channel
phase-coupling information**, or because the branch merely adds capacity / acts as
a regularizer. We answer this with an **ablation + same-marginal control
benchmark**:

| Condition | `--model` value | What it tests |
|-----------|-----------------|---------------|
| Baseline EEGNet | `eegnet` | No connectivity branch. |
| PLV-EEGNet (real) | `plv_eegnet` | Real per-trial PLV features. |
| Zero | `plv_eegnet_zero` | PLV branch present but features zeroed — pure capacity control. |
| Shuffle-pairs | `plv_eegnet_shuffle_pairs` | PLV values reassigned across channel pairs each epoch — destroys pairing while preserving the **marginal** PLV distribution. |
| Shuffle-pairs (fixed) | `plv_eegnet_shuffle_pairs_fixed` | Same as above but the permutation is fixed per trial across epochs — isolates the same-marginal effect from per-epoch reshuffling regularization. |

Auxiliary connectivity metrics — **PLI** (`plv_eegnet_pli`) and **wPLI**
(`plv_eegnet_wpli`) — probe volume-conduction sensitivity. Auxiliary
robustness families (CSD spatial filtering, P->16 bottleneck `bn16`, broadband
1–120 Hz band-matched controls, line-notch) are included for the high-channel
datasets.

## Datasets

Four public motor-imagery EEG datasets, all downloaded **via MOABB** (not
redistributed here):

| Short name | Dataset | Channels | Classes | Epochs trained |
|-----------|---------|----------|---------|----------------|
| BCI 2a | BCI Competition IV-2a | 22 | 4 | 300 |
| BCI 2b | BCI Competition IV-2b | 3 | 2 | 300 |
| Lee2019 | Lee et al. 2019 (OpenBMI) | 62 | 2 | 200 |
| Schirrmeister | Schirrmeister et al. 2017 (High-Gamma) | 128 | 4 | 200 |

> **MOABB 1.5.0 is required.** Other MOABB versions filter the Lee2019 sessions
> differently and will change the reported numbers. Pin `moabb==1.5.0`.

All runs use a proper nested train / inner-validation / test split (filenames
carry the `propersplit` suffix) and three random seeds: **42, 43, 44**.

## Environment

Reference environment: Linux + NVIDIA GPU, CUDA 13.0, Python 3.x, in a `.venv`.
Full pinned freeze in `requirements.txt`.

### Key versions

```
torch        2.11.0+cu130   (CUDA 13.0 build; freeze shows torch==2.11.0 + nvidia-*-cu13 wheels)
mne          1.12.1
moabb        1.5.0          <-- REQUIRED EXACT VERSION (Lee2019 session filtering)
numpy        2.4.4
scipy        1.17.1
scikit-learn 1.8.0
```

Setup:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Repository layout

```
src/
  datasets/        # MOABB loaders + preprocessing (bci2a, bci2b, lee2019, schirrmeister)
  models/          # eegnet, plv_eegnet (PLV branch + plv_modes), phase_* baselines
  train_all_subjects.py   # main per-subject training driver
  train.py, eval.py
scripts/
  build_tables.py        # paper Tables I-III (Markdown to stdout)
  build_figures.py       # Figures 1-4 (PNG + PDF into figures/)
  bootstrap_and_mixed.py # Figure 5 (bootstrap rho forest) + mixed-effects fit
  bootstrap_contrasts.py # per-seed contrasts + subject-bootstrap CIs
  analyze_new_exps.py    # fixed-shuffle + bottleneck validation experiments
  precompute_csd.py      # CSD npz precompute for Lee2019/Schirrmeister aux runs
tests/                   # smoke + loader + converter + shuffle-pairs unit tests
results/                 # authoritative per-dataset result JSONs (see manifest)
figures/                 # rendered figures (PNG + PDF)
requirements.txt
LICENSE
```

## Reproducibility

### 1. Regenerate results from scratch (training)

Each result JSON in `results/` is produced by one invocation of the training
driver. General form:

```bash
.venv/bin/python -m src.train_all_subjects \
    --dataset {bci2a|bci2b|lee2019|schirrmeister} \
    --model   {eegnet|plv_eegnet|plv_eegnet_zero|plv_eegnet_shuffle_pairs|plv_eegnet_shuffle_pairs_fixed|plv_eegnet_pli|plv_eegnet_wpli} \
    --seed    {42|43|44} \
    --epochs  {300 for BCI2a/2b, 200 for Lee2019/Schirrmeister} \
    --out     results/<name>_propersplit.json
```

Auxiliary flags used by the robustness families:

- `--plv-bottleneck-dim 16` — the `bn16` (P->16 bottleneck) variants.
- `--bandpass-low 1 --bandpass-high 120` (+ `--phase-band-low/-high`) — the
  `bb1-120` band-matched broadband controls.
- `--notch 50` — the line-notch variants.
- CSD variants: first run `python -m scripts.precompute_csd --dataset {lee2019|schirrmeister}`,
  then train with `--data-dir data/<dataset>_csd`.

### 2. Regenerate tables and figures from the provided JSONs

The `results/` JSONs are already included, so the paper artifacts can be rebuilt
without any GPU training:

| Paper artifact | Command |
|----------------|---------|
| Tables I–III (main ablation, per-dataset) | `.venv/bin/python -m scripts.build_tables > tables.md` |
| Figures 1–4 (headline scatter, ablation bars, band-match, bimodal) | `.venv/bin/python -m scripts.build_figures` |
| Figure 5 (bootstrap rho forest) + mixed-effects fit | `.venv/bin/python -m scripts.bootstrap_and_mixed` |
| Per-seed contrasts + subject-bootstrap 95% CIs | `.venv/bin/python -m scripts.bootstrap_contrasts > contrast_cis.md` |
| Reviewer validation experiments (fixed-shuffle, Lee2019 bottleneck) | `.venv/bin/python -m scripts.analyze_new_exps` |

Run all commands from the bundle root. `build_tables` imports the dataset spec
from `build_figures`, so keep `scripts/` intact.

### 3. Tests

```bash
.venv/bin/python -m pytest tests/
```

## Results provenance

The result JSONs were computed across several GPU machines. Single source of
truth per dataset / family:

- **Lee2019** (main + bn16/wpli aux): primary compute node. **PLI** aux family
  was computed on a secondary node.
- **Schirrmeister main 5** (baseline, real, zero, shuffle_pairs,
  shuffle_pairs_fixed) and **PLI**: secondary compute node.
- **Schirrmeister aux** (CSD, bn16/bottleneck, bb1-120 band-match, notch, wpli):
  primary compute node.
- **BCI 2a / BCI 2b** (all 5 conditions x 3 seeds): secondary compute node.

See `results/MANIFEST.md` for the per-file machine/condition/seed table.

## License

MIT. See `LICENSE`. Copyright held by "The Authors" pending de-anonymization.
