# Fixed and resampled permutations are distinct controls in connectivity-augmented EEG decoding

This repository contains code and evidence for the original submission and the
September 2026 revision. Public source EEG datasets are obtained from their
providers and are not redistributed here.

## Article

Jing-Rung Huang and Wen-Hsiang Lu. Fixed and resampled permutations are distinct
controls in connectivity-augmented EEG decoding. *Digital Signal Processing*,
185, 106598 (2026). https://doi.org/10.1016/j.dsp.2026.106598

The article's data and code availability statement cites the anonymized review
mirror of this repository. This public repository is the de-anonymized source.
Citation metadata is in [CITATION.cff](CITATION.cff).

## Current revision

Start with **[revision_20260921/README.md](revision_20260921/README.md)**.
It contains the locked source/configuration snapshots, summaries, table/figure
builders, and compressed training/evaluation evidence for:

- 10,760 main-grid fits (ten paired replicates for the core comparisons);
- 280 HGD and 1,080 Lee2019 graph-model fits;
- 140 HGD strength-constrained finite-bank classifier fits;
- training/E-bank seals and 11,244 T plus 2,240 E bank QA records.

The revision provides both direct summary-based table reconstruction and a
prediction-based audit. It distinguishes historical evaluation exposure from
the new training/evaluation boundary. It does not claim bitwise reproducibility
of neural-network training or uniform sampling over constrained graphs.

## Original submission

The original files remain at the repository root. Their historical documentation
is preserved in [README_original_submission.md](README_original_submission.md).
The original submission snapshot is commit `d63b7da23473b27dc5e13f552ba7d11bc7cee9b0`.
Use that snapshot when checking its original source hashes. Do not mix its
three-seed/20-draw analysis with the revision's ten-replicate/100-draw core.

## License

MIT; see [LICENSE](LICENSE). Third-party code retains its accompanying license and
provenance. Data access remains governed by the original dataset providers.
