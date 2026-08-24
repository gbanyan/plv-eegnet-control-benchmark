"""Analyze the controlled permutation-schedule experiment.

The experiment crosses two training schedules (fixed versus independently
resampled assignments) with two inference schedules (fixed versus resampled).
Both schedules use the same deterministic checkpoint-selection rule; each
schedule nevertheless produces and selects its own checkpoint.

Run from the project root after the two same-node checkpoint evaluations have
been copied into ``results``::

    .venv/bin/python -m scripts.analyze_controlled_permutation_schedule

Outputs a machine-readable summary, a manuscript-ready LaTeX table, and a
two-panel figure.  Inference-resampled accuracy is the expectation over the
pre-specified 20 single-draw evaluations, not the probability ensemble score.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


RESULTS = Path("results")
FIGURES = Path("figures")
TABLES = Path("paper/tables")
DATASETS = ("bci2a", "bci2b", "lee2019", "schirrmeister")
DATASET_LABELS = {
    "bci2a": "BCI 2a",
    "bci2b": "BCI 2b",
    "lee2019": "Lee2019",
    "schirrmeister": "HGD",
}
EPOCHS = {"bci2a": 300, "bci2b": 300, "lee2019": 200, "schirrmeister": 200}
SEEDS = (42, 43, 44)
def _load(dataset: str, training: str) -> dict:
    path = RESULTS / (
        f"{dataset}_{training}_globaltrialid_exactrandperm_dedicatedrng_checkpoint_eval_mc20_reproducible.json"
    )
    payload = json.loads(path.read_text())
    expected_model = {
        "fixed": "plv_eegnet_shuffle_pairs_fixed",
        "resampled": "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed",
    }[training]
    if payload["dataset"] != dataset or payload["training_model"] != expected_model:
        raise ValueError(f"metadata mismatch in {path}")
    if payload["draws"] != 20:
        raise ValueError(f"expected 20 inference draws in {path}")
    if not payload["all_fixed_scores_match_training_records"]:
        raise ValueError(f"fixed inference did not reproduce training records: {path}")
    return payload


def _subject_arrays(payload: dict) -> dict:
    """Return seed-averaged subject arrays for each inference metric."""
    rows = payload["per_subject_seed"]
    subjects = sorted({int(row["subject"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    expected = {(subject, seed) for subject in subjects for seed in seeds}
    indexed = {}
    for row in rows:
        seed = int(row["seed"])
        key = (int(row["subject"]), seed)
        if key in indexed:
            raise ValueError(f"duplicate subject/seed row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("checkpoint evaluation is not a complete subject-by-seed grid")

    def aggregate(field: str) -> np.ndarray:
        return np.asarray([
            np.mean([float(indexed[(subject, seed)][field]) for seed in seeds])
            for subject in subjects
        ])

    seed_means = {
        str(seed): {
            "fixed_accuracy": float(np.mean([
                indexed[(subject, seed)]["fixed_accuracy"] for subject in subjects
            ])),
            "resampled_mean_accuracy": float(np.mean([
                indexed[(subject, seed)]["resampled_mean_accuracy"]
                for subject in subjects
            ])),
            "resampled_probability_ensemble_accuracy": float(np.mean([
                indexed[(subject, seed)]["resampled_probability_ensemble_accuracy"]
                for subject in subjects
            ])),
        }
        for seed in seeds
    }
    draw_matrix = np.asarray([
        indexed[(subject, seed)]["resampled_draw_accuracies"]
        for subject in subjects
        for seed in seeds
    ], dtype=float)
    return {
        "subjects": np.asarray(subjects),
        "fixed": aggregate("fixed_accuracy"),
        "resampled": aggregate("resampled_mean_accuracy"),
        "ensemble": aggregate("resampled_probability_ensemble_accuracy"),
        "draw_sd": aggregate("resampled_sd_accuracy"),
        "draw_range": aggregate("resampled_max_accuracy")
        - aggregate("resampled_min_accuracy"),
        "draw_matrix": draw_matrix,
        "seed_means": seed_means,
    }


def _load_benchmark_condition(dataset: str, model: str) -> dict:
    """Load legacy-independent baseline/zero/real results as paired arrays."""
    per_seed = []
    subject_grid = None
    for seed in SEEDS:
        path = RESULTS / (
            f"{dataset}_{model}_seed{seed}_{EPOCHS[dataset]}ep_propersplit.json"
        )
        payload = json.loads(path.read_text())
        rows = sorted(payload["per_subject"], key=lambda row: int(row["subject"]))
        subjects = np.asarray([int(row["subject"]) for row in rows])
        if subject_grid is None:
            subject_grid = subjects
        elif not np.array_equal(subject_grid, subjects):
            raise ValueError(f"subject grid mismatch in {path}")
        per_seed.append([float(row["eval_at_best_val"]) for row in rows])
    matrix = np.asarray(per_seed)
    return {
        "subjects": subject_grid,
        "subject_seed_mean": matrix.mean(axis=0),
        "per_seed_dataset_mean": {
            str(seed): float(matrix[index].mean()) for index, seed in enumerate(SEEDS)
        },
    }
def _bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def _wilcoxon(values: np.ndarray) -> tuple[float, float]:
    if np.allclose(values, 0):
        return 0.0, 1.0
    result = stats.wilcoxon(
        values, zero_method="wilcox", method="auto", alternative="two-sided"
    )
    return float(result.statistic), float(result.pvalue)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values in original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted.tolist()


def _contrast(a: np.ndarray, b: np.ndarray, *, seed: int) -> dict:
    delta = a - b
    t_result = stats.ttest_rel(a, b, alternative="two-sided")
    w_stat, w_p = _wilcoxon(delta)
    sd = float(delta.std(ddof=1))
    return {
        "mean_delta_pp": float(delta.mean() * 100),
        "sd_delta_pp": sd * 100,
        "bootstrap_95ci_pp": [x * 100 for x in _bootstrap_ci(delta, seed=seed)],
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p": w_p,
        "cohen_dz": float(delta.mean() / sd) if sd > 0 else math.nan,
        "n_subjects": int(len(delta)),
    }


def _cell(values: np.ndarray, *, seed: int) -> dict:
    return {
        "mean_accuracy": float(values.mean()),
        "sd_across_subjects": float(values.std(ddof=1)),
        "bootstrap_95ci": _bootstrap_ci(values, seed=seed),
        "n_subjects": int(len(values)),
    }


def _fmt_p(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    if value < 0.001:
        return f"={value:.4f}"
    return f"={value:.3f}"


def _analyze_dataset(dataset: str, *, seed: int) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    arrays = {
        training: _subject_arrays(_load(dataset, training))
        for training in ("fixed", "resampled")
    }
    if not np.array_equal(arrays["fixed"]["subjects"], arrays["resampled"]["subjects"]):
        raise ValueError(f"subject grids differ between schedules for {dataset}")

    cells = {
        training: {
            inference: _cell(arrays[training][inference], seed=seed + i * 10 + j)
            for j, inference in enumerate(("fixed", "resampled", "ensemble"))
        }
        for i, training in enumerate(("fixed", "resampled"))
    }
    contrasts = {
        "training_resampled_minus_fixed_at_fixed_inference": _contrast(
            arrays["resampled"]["fixed"], arrays["fixed"]["fixed"], seed=seed + 101
        ),
        "training_resampled_minus_fixed_at_resampled_inference": _contrast(
            arrays["resampled"]["resampled"], arrays["fixed"]["resampled"], seed=seed + 102
        ),
        "inference_resampled_minus_fixed_after_fixed_training": _contrast(
            arrays["fixed"]["resampled"], arrays["fixed"]["fixed"], seed=seed + 103
        ),
        "inference_resampled_minus_fixed_after_resampled_training": _contrast(
            arrays["resampled"]["resampled"], arrays["resampled"]["fixed"], seed=seed + 104
        ),
    }
    stochasticity = {
        training: {
            "mean_within_checkpoint_draw_sd_pp": float(arrays[training]["draw_sd"].mean() * 100),
            "mean_within_checkpoint_draw_range_pp": float(arrays[training]["draw_range"].mean() * 100),
            # Preserve draw index while averaging over all subject-seed
            # checkpoints, then estimate the Monte Carlo SE of that aggregate
            # 20-draw mean. This retains any cross-checkpoint dependence
            # induced by using the same draw index/seed convention.
            "aggregate_20_draw_monte_carlo_se_pp": float(
                arrays[training]["draw_matrix"].mean(axis=0).std(ddof=1)
                / math.sqrt(arrays[training]["draw_matrix"].shape[1])
                * 100
            ),
        }
        for training in ("fixed", "resampled")
    }
    benchmark = {
        "baseline": _load_benchmark_condition(dataset, "eegnet"),
        "zero": _load_benchmark_condition(dataset, "plv_eegnet_zero"),
        "real": _load_benchmark_condition(dataset, "plv_eegnet"),
    }
    for condition in benchmark.values():
        if not np.array_equal(condition["subjects"], arrays["fixed"]["subjects"]):
            raise ValueError(f"benchmark subject grid differs for {dataset}")
    benchmark_cells = {
        name: _cell(values["subject_seed_mean"], seed=seed + 200 + index)
        for index, (name, values) in enumerate(benchmark.items())
    }
    benchmark_cells["shuffle_fixed_training_fixed_inference"] = _cell(
        arrays["fixed"]["fixed"], seed=seed + 210
    )
    benchmark_cells["shuffle_resampled_training_fixed_inference"] = _cell(
        arrays["resampled"]["fixed"], seed=seed + 211
    )
    real = benchmark["real"]["subject_seed_mean"]
    benchmark_contrasts = {
        "zero_minus_baseline": _contrast(
            benchmark["zero"]["subject_seed_mean"],
            benchmark["baseline"]["subject_seed_mean"],
            seed=seed + 219,
        ),
        "real_minus_baseline": _contrast(
            real, benchmark["baseline"]["subject_seed_mean"], seed=seed + 220
        ),
        "real_minus_zero": _contrast(
            real, benchmark["zero"]["subject_seed_mean"], seed=seed + 221
        ),
        "real_minus_shuffle_fixed_training": _contrast(
            real, arrays["fixed"]["fixed"], seed=seed + 222
        ),
        "real_minus_shuffle_resampled_training": _contrast(
            real, arrays["resampled"]["fixed"], seed=seed + 223
        ),
    }
    return {
        "n_subjects": int(len(arrays["fixed"]["subjects"])),
        "cells": cells,
        "per_seed_dataset_mean": {
            training: arrays[training]["seed_means"]
            for training in ("fixed", "resampled")
        },
        "paired_contrasts": contrasts,
        "resampled_inference_stochasticity": stochasticity,
        "benchmark_cells": benchmark_cells,
        "benchmark_per_seed_dataset_mean": {
            name: values["per_seed_dataset_mean"]
            for name, values in benchmark.items()
        },
        "benchmark_paired_contrasts": benchmark_contrasts,
    }, arrays


def _write_table(summary: dict) -> None:
    lines = [
        "% Generated by scripts/analyze_controlled_permutation_schedule.py",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Training schedule & Fixed inference & Mean resampled & Probability ensemble \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        label = DATASET_LABELS[dataset]
        for i, training in enumerate(("fixed", "resampled")):
            cells = summary["datasets"][dataset]["cells"][training]
            ds_cell = label if i == 0 else ""
            lines.append(
                f"{ds_cell} & {training.capitalize()} & "
                f"{cells['fixed']['mean_accuracy'] * 100:.2f} & "
                f"{cells['resampled']['mean_accuracy'] * 100:.2f} & "
                f"{cells['ensemble']['mean_accuracy'] * 100:.2f} \\\\"
            )
        contrast = summary["datasets"][dataset]["paired_contrasts"][
            "training_resampled_minus_fixed_at_fixed_inference"
        ]
        lo, hi = contrast["bootstrap_95ci_pp"]
        lines.append(
            f"\\multicolumn{{2}}{{r}}{{Training effect at fixed inference:}} & "
            f"\\multicolumn{{3}}{{l}}{{{contrast['mean_delta_pp']:+.2f} pp "
            f"[{lo:+.2f}, {hi:+.2f}], "
            f"$d_z$={contrast['cohen_dz']:.3f}, "
            f"$p_{{t,\\mathrm{{Holm}}}}${_fmt_p(contrast['paired_t_p_holm'])}}} \\\\"
        )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "controlled_permutation_schedule.tex").write_text("\n".join(lines))


def _write_benchmark_table(summary: dict) -> None:
    lines = [
        "% Generated by scripts/analyze_controlled_permutation_schedule.py",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Baseline & Zero & Shuffle $R,F$ & Real PLV & Shuffle $F,F$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        result = summary["datasets"][dataset]
        cells = result["benchmark_cells"]
        values = [
            cells["baseline"]["mean_accuracy"],
            cells["zero"]["mean_accuracy"],
            cells["shuffle_resampled_training_fixed_inference"]["mean_accuracy"],
            cells["real"]["mean_accuracy"],
            cells["shuffle_fixed_training_fixed_inference"]["mean_accuracy"],
        ]
        lines.append(
            f"{DATASET_LABELS[dataset]} & "
            + " & ".join(f"{value * 100:.2f}" for value in values)
            + " \\\\"
        )
        contrasts = result["benchmark_paired_contrasts"]
        lines.append(
            "\\multicolumn{2}{r}{Real $-$ shuffle:} & "
            f"{contrasts['real_minus_shuffle_resampled_training']['mean_delta_pp']:+.2f} & "
            "-- & "
            f"{contrasts['real_minus_shuffle_fixed_training']['mean_delta_pp']:+.2f} \\\\"
        )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "controlled_permutation_benchmark.tex").write_text("\n".join(lines))


def _write_figure(all_arrays: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes_grid = plt.subplots(
        2, 2, figsize=(7.2, 5.5), sharey=False, constrained_layout=True
    )
    axes = axes_grid.ravel()
    x = np.arange(2)
    colors = {"fixed": "#3B6FB6", "resampled": "#D45A3A"}
    for ax, dataset in zip(axes, DATASETS):
        arrays = all_arrays[dataset]
        for training_index, training in enumerate(("fixed", "resampled")):
            means = [arrays[training][inference].mean() * 100 for inference in ("fixed", "resampled")]
            intervals = [
                _bootstrap_ci(
                    arrays[training][inference],
                    seed=91_000 + DATASETS.index(dataset) * 100
                    + training_index * 10 + inference_index,
                )
                for inference_index, inference in enumerate(("fixed", "resampled"))
            ]
            yerr = np.asarray([
                [mean - interval[0] * 100 for mean, interval in zip(means, intervals)],
                [interval[1] * 100 - mean for mean, interval in zip(means, intervals)],
            ])
            ax.errorbar(
                x, means, yerr=yerr, marker="o", linewidth=1.8, capsize=3,
                color=colors[training], label=f"{training.capitalize()} training",
            )
        ax.set_xticks(x, ["Fixed", "Resampled"])
        ax.set_xlabel("Inference permutation schedule")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_controlled_permutation_schedule.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_controlled_permutation_schedule.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_subject_csv(all_arrays: dict) -> None:
    path = RESULTS / "controlled_permutation_schedule_subject_means.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "subject", "training", "inference", "accuracy"),
            lineterminator="\n",
        )
        writer.writeheader()
        for dataset in DATASETS:
            for training in ("fixed", "resampled"):
                arrays = all_arrays[dataset][training]
                for subject_index, subject in enumerate(arrays["subjects"]):
                    for inference in ("fixed", "resampled", "ensemble"):
                        writer.writerow({
                            "dataset": dataset,
                            "subject": int(subject),
                            "training": training,
                            "inference": inference,
                            "accuracy": float(arrays[inference][subject_index]),
                        })


def main() -> None:
    summary = {
        "analysis": "controlled permutation schedule 2x2",
        "unit_of_inference": "subject after averaging the three training seeds",
        "resampled_inference_metric": "mean single-draw accuracy across 20 draws",
        "bootstrap_resamples": 10_000,
        "datasets": {},
    }
    all_arrays = {}
    for index, dataset in enumerate(DATASETS):
        summary["datasets"][dataset], all_arrays[dataset] = _analyze_dataset(
            dataset, seed=73_001 + index * 1_000
        )

    primary_key = "training_resampled_minus_fixed_at_fixed_inference"
    secondary_keys = (
        "inference_resampled_minus_fixed_after_fixed_training",
        "inference_resampled_minus_fixed_after_resampled_training",
    )
    primary = [
        summary["datasets"][dataset]["paired_contrasts"][primary_key]
        for dataset in DATASETS
    ]
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = _holm_adjust([contrast[field] for contrast in primary])
        for contrast, value in zip(primary, adjusted):
            contrast[f"{field}_holm"] = value

    # Secondary baseline-anchored PLV family: real PLV versus plain EEGNet and
    # versus the zero-input augmented model in every dataset (2 x 4 tests).
    plv_family = [
        summary["datasets"][dataset]["benchmark_paired_contrasts"][key]
        for dataset in DATASETS
        for key in ("real_minus_baseline", "real_minus_zero")
    ]
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = _holm_adjust([contrast[field] for contrast in plv_family])
        for contrast, value in zip(plv_family, adjusted):
            contrast[f"{field}_holm_plv_k8"] = value

    secondary = [
        summary["datasets"][dataset]["paired_contrasts"][key]
        for dataset in DATASETS
        for key in secondary_keys
    ]
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = _holm_adjust([contrast[field] for contrast in secondary])
        for contrast, value in zip(secondary, adjusted):
            contrast[f"{field}_holm"] = value

    output = RESULTS / "controlled_permutation_schedule_analysis.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    _write_table(summary)
    _write_benchmark_table(summary)
    _write_figure(all_arrays)
    _write_subject_csv(all_arrays)
    print(f"written: {output}")
    print(f"written: {TABLES / 'controlled_permutation_schedule.tex'}")
    print(f"written: {FIGURES / 'fig_controlled_permutation_schedule.pdf'}")


if __name__ == "__main__":
    main()
