# Revision evidence snapshot

This directory retains the main-grid and graph-model summaries used by the
manuscript asset builder. Build from existing evidence with:

```sh
rtk proxy .venv/bin/python -m scripts.build_revision_manuscript_assets --from-snapshot
```

## Archive key mapping

- `frozen`: fixed fitting-set standardization with learnable affine parameters;
  not BatchNorm frozen after a warmup.
- GCN `baseline`: identity adjacency within the graph model, not plain EEGNet.
- Historical `wpli`: the implemented phase-only pwPLI, not amplitude-weighted wPLI.
- `schirrmeister`: High-Gamma Dataset (HGD).

The anchor table reports matched five-replicate means; core table means use ten.
Dimension means average the three mask families with the same five replicates.
No new inferential family is attached to these descriptive accuracy columns.

The separate `results/strength_classifier_20260921/` directory contains the
140-fit summary, prediction-derived grid, training/E-bank seals, and E-bank QA.
Actual training host/environment is recorded in each distributed train.json.
Initial manifest worker labels are allocation plans, not execution-host records.
See ../../README.md for the conversion provenance and interpretation limits.
