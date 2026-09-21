"""Generate and verify the frozen DSP revision run matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.revision import atomic_json, sha256_file

DATASETS = {"bci2a": 9, "bci2b": 9, "lee2019": 54, "schirrmeister": 14}
SOURCE_FILES = [p.as_posix() for p in sorted(Path("src").rglob("*.py"))] + [
                "scripts/revision_manifest.py", "scripts/train_revision.py",
                "scripts/evaluate_revision.py", "scripts/run_revision_worker.py",
                "scripts/summarize_revision.py"]


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path, *, verify=True):
    manifest = json.loads(path.read_text())
    if verify:
        for name, expected in manifest["source_sha256"].items():
            if sha256_file(Path(name)) != expected:
                raise RuntimeError(f"frozen source hash differs: {name}")
    return manifest


def jobs_for_batch(manifest, batch):
    return [job for job in manifest["jobs"] if job["batch"] == batch]


def make_manifest():
    conditions = {}
    batches = []
    jobs = []

    def add_condition(name, mode, normalization="batchnorm", **options):
        conditions[name] = {"mode": mode, "normalization": normalization,
                            "band": [8.0, 30.0], **options}

    for mode in ("real", "fixed", "resampled", "baseline", "zero", "global"):
        add_condition(mode, mode)
    for norm in ("none", "layernorm", "frozen", "bn_noaffine"):
        for mode in ("fixed", "resampled"):
            add_condition(f"{mode}_{norm}", mode, norm)
    for label, band in (("mu", [8.0, 13.0]), ("beta", [13.0, 30.0])):
        for mode in ("fixed", "resampled"):
            add_condition(f"{mode}_{label}", mode, band=band)
    for mode in ("fixed", "resampled"):
        add_condition(f"{mode}_edge", mode, edge=True)
    for mode in ("donor_fixed", "donor_resampled"):
        add_condition(mode, mode)
    for dimension in (128, 512, 2048):
        for mask in (1, 2, 3):
            for mode in ("fixed", "resampled"):
                add_condition(f"{mode}_p{dimension}_mask{mask}", mode,
                              dimension=dimension, mask=mask)
    for norm in ("batchnorm", "none"):
        for q in (0.05, 0.2):
            add_condition(f"refresh_q{q}_{norm}", "refresh", norm, refresh_q=q)

    def batch(phase, dataset, names, seeds):
        name = f"{phase}_{dataset}"
        worker = "gblab146" if dataset == "schirrmeister" else "gblinux"
        batches.append({"name": name, "dataset": dataset, "worker": worker, "phase": phase})
        for subject in range(1, DATASETS[dataset] + 1):
            for replicate in seeds:
                for condition in names:
                    jobs.append({"id": f"{dataset}_s{subject:02d}_r{replicate}_{condition}",
                                 "batch": name, "dataset": dataset, "subject": subject,
                                 "replicate": replicate, "condition": condition,
                                 "worker": worker})

    for dataset in DATASETS:
        batch("core", dataset, ["fixed", "resampled", "real"], range(1001, 1011))
    for dataset in ("lee2019", "schirrmeister"):
        batch("normalization", dataset,
              [f"{mode}_{norm}" for norm in ("none", "layernorm", "frozen", "bn_noaffine")
               for mode in ("fixed", "resampled")], range(1001, 1006))
    for dataset in ("lee2019", "schirrmeister"):
        batch("bands", dataset, [f"{mode}_{band}" for band in ("mu", "beta")
                                 for mode in ("fixed", "resampled")], range(1001, 1006))
    for dataset in DATASETS:
        names = ["baseline", "zero"]
        if dataset in ("lee2019", "schirrmeister"):
            names.append("global")
        batch("anchors", dataset, names, range(1001, 1006))
    batch("refresh", "schirrmeister", [n for n in conditions if n.startswith("refresh_")], range(1001, 1006))
    for dataset in ("lee2019", "schirrmeister"):
        batch("donors", dataset, ["donor_fixed", "donor_resampled"], range(1001, 1006))
        batch("edges", dataset, ["fixed_edge", "resampled_edge"], range(1001, 1006))
    batch("dimensions", "schirrmeister", [n for n in conditions if "_mask" in n], range(1001, 1006))
    if len(jobs) != 10760 or len({j["id"] for j in jobs}) != len(jobs):
        raise AssertionError("unexpected run matrix")
    return {
        "protocol": "dsp-revision-v1-20260919", "status": "locked_before_new_E_results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training": {"batch_size": 64, "lr": 0.001, "weight_decay": 0.0001,
                     "minimum_epochs": 100, "patience": 75, "min_delta": 0.0001,
                     "maximum_epochs": 500,
                     "fixed_budget": {"bci2a": 300, "bci2b": 300, "lee2019": 200, "schirrmeister": 200},
                     "inner_validation_fraction": 0.2,
                     "selection": "earliest_minimum_validation_loss_before_virtual_stop",
                     "sensitivity": ["earliest_maximum_validation_accuracy_over_completed_trajectory",
                                     "fixed_budget_epoch"],
                     "trajectory_end": "max(virtual_stop,fixed_budget), capped at 500",
                     "seed_streams": "independent subject-specific SHA256-derived named streams",
                     "dtype": "float32", "deterministic_algorithms": False},
        "evaluation": {"mc_draws": 100, "mc_conditions": ["fixed", "resampled"],
                       "mc_checkpoint": "primary", "evaluate_only_after_batch_seal": True,
                       "probabilities": "float32; mean single-draw and ensemble kept separate"},
        "analysis": {"equivalence_margin_pp": 1.0, "bootstrap_resamples": 10000,
                     "primary_family": "fixed vs resampled, four datasets, Holm",
                     "uncertainty": "balanced random-intercept moment decomposition and subject-mean t interval; subject bootstrap sensitivity",
                     "note": "new subject-specific seeds; existing E sets are historically inspected"},
        "conditional_not_launched": {
            "E6_constrained_network_null": "requires validated preservation/diversity and runtime pilot; 140 fits not silently substituted",
            "GNN": "optional architectural extension outside the locked matrix"},
        "source_sha256": {name: sha256_file(Path(name)) for name in SOURCE_FILES},
        "conditions": conditions, "batches": batches, "jobs": jobs,
        "job_counts_by_worker": dict(Counter(j["worker"] for j in jobs)),
        "job_counts_by_phase": dict(Counter(j["batch"].split("_")[0] for j in jobs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("refusing to overwrite frozen protocol")
    manifest = make_manifest()
    atomic_json(args.out, manifest)
    print(json.dumps({"manifest_sha256": manifest_hash(manifest), "jobs": len(manifest["jobs"]),
                      "workers": manifest["job_counts_by_worker"]}))


if __name__ == "__main__":
    main()
