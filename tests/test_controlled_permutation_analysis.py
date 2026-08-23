from __future__ import annotations

import numpy as np

from scripts.analyze_controlled_permutation_schedule import (
    _contrast,
    _holm_adjust,
    _subject_arrays,
)


def _row(subject: int, seed: int, fixed: float, resampled: float) -> dict:
    return {
        "subject": subject,
        "seed": seed,
        "checkpoint": (
            "results/checkpoints/lee2019_plv_eegnet_shuffle_pairs_fixed_"
            f"globaltrialid_exactrandperm_dedicatedrng_seed{seed}_"
            f"200ep_propersplit/L{subject:02d}_best.pt"
        ),
        "fixed_accuracy": fixed,
        "resampled_mean_accuracy": resampled,
        "resampled_probability_ensemble_accuracy": resampled + 0.01,
        "resampled_draw_accuracies": [resampled - 0.01, resampled + 0.01],
        "resampled_sd_accuracy": 0.02,
        "resampled_min_accuracy": resampled - 0.03,
        "resampled_max_accuracy": resampled + 0.03,
    }


def test_subject_arrays_average_seeds_before_subject_inference() -> None:
    payload = {
        "per_subject_seed": [
            _row(2, 43, 0.70, 0.72),
            _row(1, 42, 0.50, 0.54),
            _row(2, 42, 0.60, 0.64),
            _row(1, 43, 0.60, 0.62),
        ]
    }
    arrays = _subject_arrays(payload)
    np.testing.assert_array_equal(arrays["subjects"], [1, 2])
    np.testing.assert_allclose(arrays["fixed"], [0.55, 0.65])
    np.testing.assert_allclose(arrays["resampled"], [0.58, 0.68])
    np.testing.assert_allclose(arrays["draw_range"], [0.06, 0.06])
    assert np.isclose(arrays["seed_means"]["42"]["fixed_accuracy"], 0.55)
    assert np.isclose(arrays["seed_means"]["43"]["fixed_accuracy"], 0.65)


def test_contrast_direction_is_first_minus_second_in_percentage_points() -> None:
    result = _contrast(
        np.asarray([0.68, 0.8, 0.92]),
        np.asarray([0.6, 0.7, 0.8]),
        seed=123,
    )
    assert np.isclose(result["mean_delta_pp"], 10.0)
    assert result["n_subjects"] == 3


def test_holm_adjustment_is_step_down_and_restores_original_order() -> None:
    adjusted = _holm_adjust([0.04, 0.001, 0.02, 0.5])
    np.testing.assert_allclose(adjusted, [0.08, 0.004, 0.06, 0.5])
