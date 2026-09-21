# September 2026 revision reproducibility snapshot

Run commands below **from this directory**, not the original repository root.
The revised title is *Fixed and resampled permutations are distinct controls in
connectivity-augmented EEG decoding*.

## Included evidence

| Study | Fits | Replicates | Manifest |
| --- | ---: | --- | --- |
| Main core and sensitivities | 10,760 | 10 core / 5 sensitivities | `configs/dsp_revision_20260919.json` |
| HGD channel-GCN | 280 | 5 | `configs/gcn_revision_v1_20260920.json` |
| Lee2019 channel-GCN | 1,080 | 5 | `configs/gcn_lee_revision_v1_20260920.json` |
| HGD strength-constrained bank | 140 | 5 | `configs/strength_classifier_v1_20260921.json` |

`src/` and the manifest-listed scripts retain their exact frozen bytes.
`results/revision_manuscript_20260921/` and
`results/strength_classifier_20260921/` contain final numerical summaries.
`paper/revision_2026-09/` contains the generated scientific tables and figures,
not author correspondence or the manuscript's identifying front matter.

`evidence/` contains independent gzip-compressed tar parts and a hash inventory.
These expand into `runs/` with training records, validation curves, evaluation
records, prediction arrays, training seals, and per-trial T/E bank QA records.
All 12,260 fits and all three checkpoint rules are included. Raw EEG, model
checkpoint weights and full surrogate-bank arrays are not redistributed in this
snapshot. Their recorded hashes are provenance references, not proof that a
reader has independently checked those omitted files. The original failed-bank
and refinement-pilot summaries remain in the bank evidence directory.

## Environment

For summary reproduction and prediction auditing (CPU is sufficient):

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
```

The supplied analysis dependency versions were tested on macOS with Python 3.14.
Training records separately report the actual Linux host, Python, PyTorch, CUDA
and GPU for each fit. The main and constrained-classifier environments used
PyTorch 2.11.0/CUDA 13.0 on RTX 4090/5090 hosts; these records supersede initial
manifest worker labels for actual execution provenance. Deterministic GPU
training was not enforced. GPU bank generation additionally needs Triton/CUDA;
no GPU is needed for the commands below.

Historical notes document BCI2a's MOABB conversion, and the 18 local/remote caches
were hash-matched. Cache-linked conversion logs and historical MNE/MOABB versions
are not independently established. The original root requirements file is an
historical environment declaration, not a verified cache-conversion lock.

## Rebuild tables and figures from summaries

```sh
.venv/bin/python -m scripts.build_revision_manuscript_assets --from-snapshot
.venv/bin/python -m scripts.build_strength_manuscript_assets
```

Table values are deterministic; PDF/PNG metadata or rendering can depend on the
Matplotlib environment. `asset_hashes.json` records the exact exported files.

## Unpack and audit underlying evidence

```sh
.venv/bin/python unpack_evidence.py
.venv/bin/python verify_archive.py
.venv/bin/python verify_statistics.py
```

The unpacker checks part and member SHA-256 hashes and rejects unsafe paths.
The archive verifier checks all four frozen source manifests, complete job grids,
training/curve/prediction hashes, evaluation-to-seal links, and recomputes fixed
accuracies and primary Monte Carlo metrics from prediction arrays. It also
checks all 11,244 T and 2,240 E bank record hashes. It does not verify omitted
EEG, checkpoint or bank-array bytes. The statistics verifier reconstructs the
main declared contrast families and all graph/strength checkpoint contrasts;
comparison tolerance is 1e-10 for ordinary cross-platform floating-point noise.

To rebuild the main summary explicitly after unpacking:

```sh
.venv/bin/python -m scripts.summarize_revision \
  --manifest configs/dsp_revision_20260919.json \
  --roots runs/dsp-revision-collected/gblinux runs/dsp-revision-collected/gblab146 \
  --out validation/recomputed_main
```

The original `summarize_strength_classifier` validates checkpoint weights before
summarizing and therefore requires those additional local artifacts. Use
`verify_statistics.py` for the distributed prediction/record-only audit.

## Targeted tests

```sh
.venv/bin/python -m pytest -q tests/test_permutation_operator_moments.py \
  tests/test_revision_protocol.py tests/test_gcn_revision.py \
  tests/test_strength_bank_runtime.py tests/test_strength_targeted.py \
  tests/test_strength_classifier_integration.py
```

The packaging check passed 34 tests, with 3 CUDA-dependent cases skipped on the
Mac. This is not a new full training run or an execution of the entire GPU bank
pipeline. Results and scope are recorded in `validation/README.md`.

## Interpretation and archive keys

- `frozen` means fitting-set standardization with learnable affine terms, not
  BatchNorm frozen after a warmup.
- GCN `baseline` means identity adjacency, not plain EEGNet.
- Historical `wpli` means the implemented phase-only pwPLI.
- `schirrmeister` means HGD; F/R label training schedules under common fixed
  inference unless explicitly stated otherwise.
- Anchor and dimensionality tables use matched five-replicate grids; primary
  core means use ten. The dimension analysis averages three mask families.
- Generic statistics contain TOST fields for some descriptive contrasts. Only
  manuscript-declared inference/global-relabeling families claim equivalence;
  the GCN and finite-bank extensions have no declared equivalence endpoint.
- Reused E partitions had been inspected historically. The constrained-bank
  generator was amended using T-only checks, and bank realizations are shared
  across classifier replicates. The Lee GCN extension was chosen after HGD
  results. These scope limits are not removed by checkpoint sealing.

## Packaging and preservation

The root-level original submission is preserved separately. Compact, immutable
archive parts are stored as ordinary Git blobs so anonymous downloads do not
depend on Git LFS pointer resolution. Each part is kept small for direct file
retrieval; raw record paths/hashes remain unchanged after extraction. No external
service credentials or private author/reviewer correspondence are included.
