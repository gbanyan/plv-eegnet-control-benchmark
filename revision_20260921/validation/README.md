# Packaging validation

Executed on the Mac from the isolated revision snapshot, 2026-09-21:

- 12,260 evaluation records exactly matched declared grids.
- 36,780 fixed-checkpoint accuracies recomputed from predictions.
- 6,880 primary Monte Carlo summary values recomputed within 1e-7 tolerance.
- All training, validation-curve and prediction bytes matched recorded hashes.
- All four source-manifest sets and evaluation/seal links verified.
- 11,244 T and 2,240 E bank record hashes matched their bank seals.
- Main cells/contrasts/family adjustments reproduced; graph and strength
  checkpoint summaries matched to 1e-10 numeric tolerance. Tiny differences
  in some graph p-values/intervals at about 1e-16 are platform rounding.
- Targeted protocol/operator/graph/bank tests: 34 passed, 3 skipped (CUDA).

No raw EEG, model checkpoint weights, or complete surrogate-bank arrays are
included or re-audited here. Their recorded hashes must not be described as
independent validation of omitted bytes. No new scientific training was run.

Packaging: 62,557 evidence members in 134 independently compressed tar parts,
522,954,974 compressed bytes. All parts and members passed the unpacker checksum
checks; extraction into a fresh directory was also tested on the first part.
Main training seals cover the complete declared batch membership. Table source
files reproduced byte-for-byte; both asset ledgers validated. No Git LFS pointers
are needed for anonymous downloads (largest part: 6,920,019 bytes).
