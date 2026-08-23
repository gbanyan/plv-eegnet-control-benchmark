import numpy as np

from scripts.bootstrap_and_mixed import (
    bootstrap_spearman,
    permutation_spearman_p,
)


def test_exact_spearman_permutation_detects_perfect_order_at_n4() -> None:
    x = np.arange(4, dtype=float)
    p, method, enumerations = permutation_spearman_p(x, x)
    assert method == "exact"
    assert enumerations == 24
    assert np.isclose(p, 2 / 24)


def test_bootstrap_spearman_is_seed_reproducible() -> None:
    x = np.arange(8, dtype=float)
    y = np.asarray([0, 2, 1, 4, 3, 6, 7, 5], dtype=float)
    first = bootstrap_spearman(x, y, n_boot=100, seed=7)
    second = bootstrap_spearman(x, y, n_boot=100, seed=7)
    np.testing.assert_allclose(first[:3], second[:3])
    np.testing.assert_allclose(first[3], second[3])
    assert first[4] == second[4]
