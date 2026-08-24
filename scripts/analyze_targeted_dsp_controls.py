"""Analyze preregistered targeted DSP controls after all runs are present.

The script is intentionally fail-closed: it writes no output unless every
dataset/condition/seed file exists and has the expected subject set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "targeted_dsp_controls_analysis.json"
SEEDS = (42, 43, 44)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260824

DATASETS = {
    "lee2019": {"epochs": 200, "subjects": list(range(1, 55))},
    "schirrmeister": {"epochs": 200, "subjects": list(range(1, 15))},
}

FILE_TEMPLATES = {
    "real": "{dataset}_plv_eegnet_seed{seed}_{epochs}ep_propersplit.json",
    "global_fixed": "{dataset}_global_fixed_seed{seed}_{epochs}ep_propersplit.json",
    "bn_fixed": (
        "{dataset}_plv_eegnet_shuffle_pairs_fixed_globaltrialid_"
        "exactrandperm_dedicatedrng_seed{seed}_{epochs}ep_propersplit.json"
    ),
    "bn_resampled": (
        "{dataset}_plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_"
        "globaltrialid_exactrandperm_dedicatedrng_seed{seed}_{epochs}ep_"
        "propersplit.json"
    ),
    "no_norm_fixed": (
        "{dataset}_trial_fixed_no_norm_seed{seed}_{epochs}ep_propersplit.json"
    ),
    "no_norm_resampled": (
        "{dataset}_train_resampled_eval_fixed_no_norm_seed{seed}_{epochs}ep_"
        "propersplit.json"
    ),
}


def _load_condition(dataset: str, condition: str, spec: dict) -> np.ndarray:
    per_seed = []
    expected = spec["subjects"]
    for seed in SEEDS:
        path = RESULTS / FILE_TEMPLATES[condition].format(
            dataset=dataset, seed=seed, epochs=spec["epochs"]
        )
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        rows = sorted(payload["per_subject"], key=lambda row: row["subject"])
        subjects = [int(row["subject"]) for row in rows]
        if subjects != expected:
            raise ValueError(
                f"subject mismatch in {path}: expected {expected}, got {subjects}"
            )
        per_seed.append(
            np.asarray([row["eval_at_best_val"] for row in rows], dtype=float)
        )
    return np.stack(per_seed, axis=0)


def _holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def _contrast(diff: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    bootstrap = diff[rng.integers(0, n, size=(N_BOOTSTRAP, n))].mean(axis=1)
    t_result = stats.ttest_1samp(diff, popmean=0.0)
    if np.allclose(diff, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(
            stats.wilcoxon(
                diff, alternative="two-sided", zero_method="wilcox", method="auto"
            ).pvalue
        )
    return {
        "n_subjects": n,
        "mean_pp": 100.0 * mean,
        "sd_pp": 100.0 * sd,
        "cohen_dz": None if sd == 0.0 else mean / sd,
        "bootstrap_percentile_95ci_pp": (
            100.0 * np.percentile(bootstrap, [2.5, 97.5])
        ).tolist(),
        "paired_t_p_unadjusted": float(t_result.pvalue),
        "wilcoxon_p_unadjusted": wilcoxon_p,
    }


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        "analysis_status": "specified before targeted run completion",
        "seeds": list(SEEDS),
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "unit_of_inference": "subject after averaging three seeds",
        "datasets": {},
    }
    family_rows = {"global_vs_real": [], "normalization_interaction": []}

    for dataset, spec in DATASETS.items():
        arrays = {
            condition: _load_condition(dataset, condition, spec)
            for condition in FILE_TEMPLATES
        }
        means = {condition: values.mean(axis=0) for condition, values in arrays.items()}
        bn_effect = means["bn_resampled"] - means["bn_fixed"]
        no_norm_effect = means["no_norm_resampled"] - means["no_norm_fixed"]
        contrasts = {
            "global_fixed_minus_real": _contrast(
                means["global_fixed"] - means["real"], rng
            ),
            "bn_resampled_minus_fixed": _contrast(bn_effect, rng),
            "no_norm_resampled_minus_fixed": _contrast(no_norm_effect, rng),
            "no_norm_minus_bn_schedule_interaction": _contrast(
                no_norm_effect - bn_effect, rng
            ),
        }
        output["datasets"][dataset] = {
            "condition_mean_accuracy": {
                condition: float(values.mean()) for condition, values in means.items()
            },
            "contrasts": contrasts,
            "seed_level_schedule_effect_pp": {
                "bn": (
                    100.0
                    * (arrays["bn_resampled"] - arrays["bn_fixed"]).mean(axis=1)
                ).tolist(),
                "no_norm": (
                    100.0
                    * (arrays["no_norm_resampled"] - arrays["no_norm_fixed"]).mean(axis=1)
                ).tolist(),
            },
        }
        family_rows["global_vs_real"].append(
            contrasts["global_fixed_minus_real"]
        )
        family_rows["normalization_interaction"].append(
            contrasts["no_norm_minus_bn_schedule_interaction"]
        )

    for family, rows in family_rows.items():
        adjusted_t = _holm([row["paired_t_p_unadjusted"] for row in rows])
        adjusted_w = _holm([row["wilcoxon_p_unadjusted"] for row in rows])
        for row, p_t, p_w in zip(rows, adjusted_t, adjusted_w):
            row[f"paired_t_p_holm_{family}"] = p_t
            row[f"wilcoxon_p_holm_{family}"] = p_w

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written: {OUTPUT}")


if __name__ == "__main__":
    main()
