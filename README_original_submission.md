# Permutation scheduling in connectivity-augmented EEG decoding

An anonymized code-and-results deposit for a manuscript submitted to *Digital
Signal Processing*. The repository contains the final analysis inputs, the
controlled-experiment manifest, compressed checkpoint predictions, verification
code, and rendered figures. Public EEG datasets are downloaded through MOABB and
are not redistributed.

## Study design

The controlled experiment crosses two training schedules with two inference
schedules. A fixed schedule assigns one deterministic channel-pair permutation
to every global trial ID. An independently resampled schedule draws a new
permutation at every presentation. Both preserve the same one-presentation value
multiset, but they define different cross-presentation stochastic processes.

Four public datasets are used: BCI Competition IV 2a, BCI Competition IV 2b,
Lee2019/OpenBMI, and Schirrmeister2017/HGD. All subject-level runs use three
training seeds (42--44), a train/inner-validation/evaluation split, and
validation-only checkpoint selection. BCI 2a/2b use 300 epochs; Lee2019 and HGD
use 200 epochs.

The nonpermuted benchmark includes plain EEGNet, PLV-EEGNet with a zero-valued
connectivity branch, and PLV-EEGNet with real phase-locking values. PLI and the
implemented phase-only weighted PLI are included for Lee2019 and HGD as
sensitivity analyses. The deposit deliberately excludes superseded CSD,
high-gamma/band-matching, and bottleneck outputs that are not used by the final
manuscript.

Two targeted Lee2019/HGD controls distinguish a dataset-global fixed coordinate
relabeling from trial-specific fixed permutations and repeat the fixed/resampled
training comparison without auxiliary-branch BatchNorm. These 18 runs were
analyzed using a plan recorded before any targeted run completed.

## Layout

```text
src/        dataset loaders, models, training, and evaluation
scripts/    controlled-run, checkpoint-evaluation, analysis, and verifier code
tests/      unit and smoke tests
results/    per-subject JSONs, manifest, analysis outputs, and compressed predictions
figures/    final PDF and PNG figures
```

Key provenance files:

- `results/controlled_permutation_experiment_manifest.json`: the 24 controlled
  training runs and their source hashes.
- `results/*_checkpoint_eval_mc20_reproducible.json`: checkpoint-selection
  outputs, Monte Carlo summaries, prediction hashes, fixed-assignment key hashes,
  and software metadata.
- `results/controlled_permutation_predictions/`: compressed prediction arrays
  referenced by the checkpoint-evaluation records.
- `results/controlled_permutation_schedule_analysis.json`: all 2x2 schedule
  cells and paired contrasts.
- `results/seed_epoch_diagnostics.json`: seed-specific schedule gaps,
  leave-one-seed-out summaries, and selected-epoch ceiling diagnostics.
- `results/subject_level_association_analysis.json`: prespecified per-dataset
  subject-level association analysis.
- `results/targeted_dsp_controls_analysis_plan.md`: targeted contrasts and
  multiplicity families recorded before targeted run completion.
- `results/targeted_dsp_controls_analysis.json`: global-fixed and
  no-normalization control results.

## Environment

MOABB 1.5.0 is required because Lee2019 session handling is version-sensitive.
All analyses used MNE 1.12.1, MOABB 1.5.0, NumPy 2.4.4, SciPy 1.17.1, and
scikit-learn 1.8.0. BCI 2a/2b and Lee2019 controlled runs used Python 3.12.7,
PyTorch 2.6.0 (CUDA 12.4), and an RTX 4090. HGD controlled runs used Python
3.12.3, PyTorch 2.11.0 (CUDA 13.0), and an RTX 5090. `requirements.txt` captures
the latter analysis environment; exact training metadata are also embedded in
the checkpoint-evaluation JSONs.

## Verify and reproduce analyses

Create an environment and run the tests:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Verify every controlled grid, the manifest, source hashes, prediction hashes,
and fixed-assignment identity across schedules:

```bash
.venv/bin/python scripts/verify_controlled_permutation_artifacts.py
.venv/bin/python scripts/verify_targeted_dsp_controls.py
```

Rebuild the controlled-schedule statistics, controlled figure, and subject-level
association figure from archived outputs:

```bash
.venv/bin/python -m scripts.analyze_controlled_permutation_schedule
.venv/bin/python scripts/analyze_seed_epoch_diagnostics.py
.venv/bin/python -m scripts.bootstrap_and_mixed
.venv/bin/python scripts/analyze_targeted_dsp_controls.py
```

The shell launchers in `scripts/` document the training and checkpoint-evaluation
commands for both epoch budgets. Re-running stochastic GPU training need not be
bitwise identical across hardware/software stacks; the archived prediction
arrays and hashes define the analyzed checkpoint-evaluation snapshot.

## License

MIT. See `LICENSE`. Copyright is held by “The Authors” pending
de-anonymization.
