"""Exact checks for the permutation-operator moments reported in the paper.

The test enumerates every permutation of a short vector.  It therefore checks
the finite-population identities without Monte Carlo error:

* every coordinate has mean ``mu`` and variance ``s2``;
* distinct coordinates have covariance ``-s2 / (P - 1)``;
* independent redraws have expected squared displacement ``2 * P * s2``;
* repeating one fixed permutation has zero displacement.
* every relative permutation obeys the general centered quadratic identity.
"""
from itertools import permutations
from fractions import Fraction


def test_exact_uniform_permutation_moments():
    v = tuple(Fraction(x, 10) for x in (1, 4, 8, 9))
    p = len(v)
    draws = tuple(tuple(v[i] for i in index) for index in permutations(range(p)))

    mu = sum(v) / p
    s2 = sum((x - mu) ** 2 for x in v) / p

    for k in range(p):
        assert sum(draw[k] for draw in draws) / len(draws) == mu
        assert sum((draw[k] - mu) ** 2 for draw in draws) / len(draws) == s2
        for ell in range(p):
            if k == ell:
                continue
            covariance = sum(
                (draw[k] - mu) * (draw[ell] - mu) for draw in draws
            ) / len(draws)
            assert covariance == -s2 / (p - 1)


def test_exact_cross_presentation_displacement():
    v = tuple(Fraction(x, 10) for x in (1, 4, 8, 9))
    p = len(v)
    draws = tuple(tuple(v[i] for i in index) for index in permutations(range(p)))
    mu = sum(v) / p
    s2 = sum((x - mu) ** 2 for x in v) / p

    distances = tuple(
        sum((x - y) ** 2 for x, y in zip(first, second))
        for first in draws
        for second in draws
    )
    assert sum(distances) / len(distances) == 2 * p * s2
    assert all(sum((x - x) ** 2 for x in draw) == 0 for draw in draws)

    fixed_cross_covariance = sum((draw[0] - mu) ** 2 for draw in draws) / len(draws)
    resampled_cross_covariance = sum(
        (first[0] - mu) * (second[0] - mu)
        for first in draws
        for second in draws
    ) / (len(draws) ** 2)
    assert fixed_cross_covariance == s2
    assert resampled_cross_covariance == 0


def test_relative_permutation_kernel_displacement_identity():
    """Check the general quadratic identity before averaging any schedule."""
    v = tuple(Fraction(x, 10) for x in (1, 4, 8, 9))
    p = len(v)
    mu = sum(v) / p
    s2 = sum((x - mu) ** 2 for x in v) / p
    draws = tuple(tuple(v[i] for i in index) for index in permutations(range(p)))

    for first in draws:
        for second in draws:
            distance = sum((x - y) ** 2 for x, y in zip(first, second))
            relative_inner_product = sum(
                (x - mu) * (y - mu) for x, y in zip(first, second)
            )
            assert distance == 2 * p * s2 - 2 * relative_inner_product


def test_constant_vector_is_schedule_degenerate():
    """Fixed and resampled schedules coincide when coordinate variance is zero."""
    for v in ((Fraction(3, 5),), (Fraction(3, 5),) * 4):
        p = len(v)
        draws = tuple(tuple(v[i] for i in index) for index in permutations(range(p)))
        mu = sum(v) / p
        s2 = sum((x - mu) ** 2 for x in v) / p
        assert s2 == 0
        assert all(draw == v for draw in draws)
        assert all(
            sum((x - y) ** 2 for x, y in zip(first, second)) == 0
            for first in draws
            for second in draws
        )
