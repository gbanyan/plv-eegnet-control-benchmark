"""Report complete declared contrasts; never fill missing jobs or pool seeds as people."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.revision_manifest import DATASETS, load_manifest, manifest_hash
from src.revision import atomic_json, stream_seed


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(values) - rank) * values[index])
        result[index] = min(1.0, previous)
    return result.tolist()


def contrast_statistics(differences, *, margin_pp=1.0, bootstrap=10000, seed=917):
    """Balanced random-intercept moments with subject-df t inference.

    Differences are [subjects, independent subject-specific paired replicates]
    in accuracy fractions. Within-run variance and between-subject variance are
    estimated separately; the latter is truncated at zero. The resulting SE
    includes finite-replicate variation. Normal random-effects inference is a
    modeling assumption, not an assertion of Gaussian individual accuracies.
    """
    d = np.asarray(differences, dtype=float) * 100
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 2 or not np.isfinite(d).all():
        raise ValueError("need a complete finite subject-by-replicate matrix")
    n, r = d.shape
    subject = d.mean(1)
    mean = float(subject.mean())
    within = float(d.var(1, ddof=1).mean())
    subject_variance = float(subject.var(ddof=1))
    between = max(0.0, subject_variance - within / r)
    se = float(np.sqrt(between / n + within / (n * r)))
    if se > 0:
        statistic = mean / se
        p = float(2 * stats.t.sf(abs(statistic), n - 1))
        p_equivalence = float(max(stats.t.sf((mean + margin_pp) / se, n - 1),
                                  stats.t.cdf((mean - margin_pp) / se, n - 1)))
    else:
        statistic = None
        p = 1.0 if mean == 0 else 0.0
        p_equivalence = 0.0 if abs(mean) < margin_pp else 1.0
    critical95, critical90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    rng = np.random.default_rng(seed)
    boot = subject[rng.integers(n, size=(bootstrap, n))].mean(1)
    return {
        "mean_difference_pp": mean, "n_subjects": n, "paired_replicates": r,
        "model_standard_error_pp": se,
        "model_95ci_pp": [mean - float(critical95) * se, mean + float(critical95) * se],
        "model_90ci_pp": [mean - float(critical90) * se, mean + float(critical90) * se],
        "between_subject_variance_pp2": between, "within_subject_run_variance_pp2": within,
        "t_statistic": statistic, "p_two_sided": p,
        "equivalence_margin_pp": margin_pp, "p_equivalence": p_equivalence,
        "conditional_subject_bootstrap_95ci_pp": np.percentile(boot, [2.5, 97.5]).tolist(),
        "per_replicate_mean_pp": d.mean(0).tolist(),
        "uncertainty_model": "balanced normal random-intercept moments; t df=n_subjects-1",
    }


def make_summary(manifest, roots):
    expected = {j["id"]: j for j in manifest["jobs"]}
    protocol_hash = manifest_hash(manifest)
    records, trained = {}, set()
    progress = []
    for root in roots:
        for p in (root / "jobs").glob("*/train.json"):
            if p.parent.name in expected:
                trained.add(p.parent.name)
        for p in (root / "jobs").glob("*/evaluation.json"):
            data = json.loads(p.read_text())
            identifier = data["job"]["id"]
            if identifier not in expected or data["manifest_sha256"] != protocol_hash:
                raise RuntimeError(f"foreign evaluation record: {p}")
            if identifier in records and records[identifier] != data:
                raise RuntimeError("conflicting duplicate evaluation")
            records[identifier] = data
        path = root / "progress.json"
        if path.exists():
            progress.append({"root": str(root), **json.loads(path.read_text())})

    def matrix(dataset, terms, rule="primary", metric="fixed_accuracy"):
        # Terms are weighted conditions. The intersection of their declared
        # grids defines the planned paired subset (e.g. first 5 vs core's 10).
        indices = []
        for condition in terms:
            indices.append({(j["subject"], j["replicate"]): j["id"] for j in manifest["jobs"]
                            if j["dataset"] == dataset and j["condition"] == condition})
        if not indices or any(not index for index in indices):
            return None
        grid = set.intersection(*(set(index) for index in indices))
        if not grid or any(index[key] not in records for index in indices for key in grid):
            return None
        subjects, replicates = sorted({k[0] for k in grid}), sorted({k[1] for k in grid})
        if grid != {(s, r) for s in subjects for r in replicates}:
            raise RuntimeError("nonrectangular planned contrast")
        values = np.zeros((len(subjects), len(replicates)))
        for condition, index in zip(terms, indices):
            for a, subject in enumerate(subjects):
                for b, replicate in enumerate(replicates):
                    cell = records[index[(subject, replicate)]]["metrics"][rule]
                    if metric not in cell:
                        return None
                    values[a, b] += terms[condition] * cell[metric]
        return values

    contrasts = {}

    def add(name, dataset, terms, family=None, rule="primary"):
        values = matrix(dataset, terms, rule)
        if values is None:
            return
        contrasts[name] = {"dataset": dataset, "terms": terms, "rule": rule, "family": family,
                           **contrast_statistics(values,
                               margin_pp=manifest["analysis"]["equivalence_margin_pp"],
                               bootstrap=manifest["analysis"]["bootstrap_resamples"], seed=stream_seed(name))}

    for dataset in DATASETS:
        add(f"{dataset}:schedule", dataset, {"resampled": 1, "fixed": -1}, "primary")
        for rule in ("accuracy", "fixed_budget"):
            add(f"{dataset}:schedule:{rule}", dataset, {"resampled": 1, "fixed": -1}, rule=rule)
        for other in ("fixed", "resampled", "baseline", "zero"):
            add(f"{dataset}:real_minus_{other}", dataset, {"real": 1, other: -1},
                "augmentation" if other in ("baseline", "zero") else None)
        for condition in ("fixed", "resampled"):
            mc = matrix(dataset, {condition: 1}, metric="mc_mean_accuracy")
            fixed = matrix(dataset, {condition: 1})
            if mc is not None and fixed is not None:
                name = f"{dataset}:inference:{condition}"
                contrasts[name] = {"dataset": dataset, "family": "inference",
                    **contrast_statistics(mc - fixed, bootstrap=manifest["analysis"]["bootstrap_resamples"],
                                          margin_pp=manifest["analysis"]["equivalence_margin_pp"], seed=stream_seed(name))}
                selected = [v for v in records.values() if v["job"]["dataset"] == dataset
                            and v["job"]["condition"] == condition]
                errors = np.array([v["metrics"]["primary"]["mc_standard_error"] for v in selected])
                contrasts[name]["mc_standard_error_of_grand_mean_pp"] = float(100 * np.sqrt(np.sum(errors**2)) / len(errors))
        if dataset not in ("lee2019", "schirrmeister"):
            continue
        add(f"{dataset}:global_minus_real", dataset, {"global": 1, "real": -1}, "global")
        for norm in ("none", "layernorm", "frozen", "bn_noaffine"):
            add(f"{dataset}:schedule:{norm}", dataset, {f"resampled_{norm}": 1, f"fixed_{norm}": -1})
            add(f"{dataset}:normalization_interaction:{norm}", dataset,
                {f"resampled_{norm}": 1, f"fixed_{norm}": -1, "resampled": -1, "fixed": 1}, "normalization")
        for band in ("mu", "beta", "edge"):
            add(f"{dataset}:schedule:{band}", dataset, {f"resampled_{band}": 1, f"fixed_{band}": -1},
                "bands" if band != "edge" else None)
        add(f"{dataset}:schedule:donor", dataset, {"donor_resampled": 1, "donor_fixed": -1})
    for dimension in (128, 512, 2048):
        terms = {}
        for mask in (1, 2, 3):
            terms[f"resampled_p{dimension}_mask{mask}"] = 1 / 3
            terms[f"fixed_p{dimension}_mask{mask}"] = -1 / 3
        add(f"schirrmeister:dimension:{dimension}", "schirrmeister", terms)
        add(f"schirrmeister:dimension_interaction:{dimension}", "schirrmeister",
            {**terms, "resampled": -1, "fixed": 1}, "dimensions")
    for norm in ("batchnorm", "none"):
        reference = "fixed" if norm == "batchnorm" else "fixed_none"
        for q in (0.05, 0.2):
            add(f"schirrmeister:refresh:{q}:{norm}", "schirrmeister",
                {f"refresh_q{q}_{norm}": 1, reference: -1})
    family_sizes = {"primary": 4, "augmentation": 8, "normalization": 8,
                    "inference": 8, "global": 2, "bands": 4, "dimensions": 3}
    for family, size in family_sizes.items():
        members = [c for c in contrasts.values() if c.get("family") == family]
        for member in members:
            member["family_complete"] = len(members) == size
            member["p_holm"] = None
        if len(members) == size:
            for member, adjusted in zip(members, holm([c["p_two_sided"] for c in members])):
                member["p_holm"] = adjusted
            if family in ("inference", "global"):
                for member, adjusted in zip(members, holm([c["p_equivalence"] for c in members])):
                    member["p_equivalence_holm"] = adjusted
    cells = {}
    for dataset in DATASETS:
        for condition in manifest["conditions"]:
            values = matrix(dataset, {condition: 1})
            if values is not None:
                cells[f"{dataset}:{condition}"] = {"mean_accuracy_percent": float(values.mean() * 100),
                                                    "subjects": values.shape[0], "replicates": values.shape[1]}
    counts = {}
    for batch in manifest["batches"]:
        identifiers = {j["id"] for j in manifest["jobs"] if j["batch"] == batch["name"]}
        counts[batch["name"]] = {"expected": len(identifiers), "trained": len(identifiers & trained),
                                  "evaluated": len(identifiers & set(records))}
    return {"manifest_sha256": protocol_hash, "expected_jobs": len(expected), "trained_jobs": len(trained),
            "evaluated_jobs": len(records), "all_locked_jobs_evaluated": len(records) == len(expected),
            "conditional_not_launched": manifest["conditional_not_launched"], "progress": progress,
            "batches": counts, "cells": cells, "contrasts": contrasts,
            "limitations": ["Previously inspected E partitions; new runs are protocol-locked, not historically untouched.",
                            "Only complete declared grids are analyzed; cross-dataset Holm adjustment awaits every family member.",
                            "Random-intercept t intervals assume approximately normal effects; bootstrap intervals are conditional sensitivities."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    summary = make_summary(manifest, args.roots)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "summary.json", summary)
    lines = ["# DSP revision run status", "", f"Training complete: {summary['trained_jobs']} / {summary['expected_jobs']} jobs.",
             f"Evaluation complete: {summary['evaluated_jobs']} / {summary['expected_jobs']} jobs.", "",
             "Only complete paired grids are reported. E6 constrained-null feasibility and an optional GNN are outside this locked queue.", "",
             "| Batch | Trained | Evaluated | Planned |", "| --- | ---: | ---: | ---: |"]
    for name, count in summary["batches"].items():
        lines.append(f"| {name} | {count['trained']} | {count['evaluated']} | {count['expected']} |")
    lines += ["", "## Completed contrasts", "", "| Contrast | Difference (pp) | Model 95% CI | Holm p |",
              "| --- | ---: | --- | --- |"]
    for name, result in summary["contrasts"].items():
        lo, hi = result["model_95ci_pp"]
        p = result.get("p_holm")
        lines.append(f"| {name} | {result['mean_difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p if p is not None else 'pending / descriptive'} |")
    lines += ["", *summary["limitations"], ""]
    (args.out / "STATUS.md").write_text("\n".join(lines))
    print(json.dumps({"trained": summary["trained_jobs"], "evaluated": summary["evaluated_jobs"],
                      "planned": summary["expected_jobs"]}))


if __name__ == "__main__":
    main()
