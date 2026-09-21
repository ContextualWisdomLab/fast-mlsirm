"""High-node-count Gauss-Hermite rule used by the polytomous monotonicity checks.

Negative evidence kept on purpose: ``numpy.polynomial.hermite_e.hermegauss``
returns all-zero weights from ``q = 371`` and NaN weights from ``q = 400``
(numpy 2.5.2), which made both ``check_*_expected_total_score_monotonicity``
functions return an all-NaN curve reported as ``monotone=True``. If numpy fixes
``hermegauss`` the first test fails and the local rule may be retired.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_mlsirm.polytomous import (
    _probabilists_gauss_hermite,
    check_bifactor_expected_total_score_monotonicity,
    check_focal_expected_total_score_monotonicity,
)

# 371: onset of the hermegauss failure; 481: onset previously cited; 1024: round.
HIGH_Q = (371, 481, 1024)
THRESHOLDS = np.array([1.2, 0.0, -1.2])


def test_numpy_hermegauss_still_fails_at_high_q() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, weights = np.polynomial.hermite_e.hermegauss(481)
        normalized = weights / weights.sum()
    assert not np.all(np.isfinite(normalized))


@pytest.mark.parametrize("q", HIGH_Q)
def test_rule_is_finite_and_exact_at_high_q(q: int) -> None:
    nodes, weights = _probabilists_gauss_hermite(q)
    assert np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert weights @ nodes**2 == pytest.approx(1.0, abs=1e-12)
    assert weights @ nodes**4 == pytest.approx(3.0, abs=1e-11)


@pytest.mark.parametrize("q", (1, 2, 21, 121, 300))
def test_rule_matches_hermegauss_where_hermegauss_is_finite(q: int) -> None:
    ref_nodes, ref_weights = np.polynomial.hermite_e.hermegauss(q)
    nodes, weights = _probabilists_gauss_hermite(q)
    np.testing.assert_allclose(nodes, ref_nodes, rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights, ref_weights / ref_weights.sum(), rtol=0, atol=1e-13)


class _Fit:
    slope = np.array([[1.3, 0.7], [0.9, 0.4]])
    a_general = np.array([1.3, 0.9])
    a_specific = np.array([0.7, 0.0])
    threshold = np.tile(THRESHOLDS, (2, 1))


@pytest.mark.parametrize(
    "check",
    (
        lambda grid, q: check_focal_expected_total_score_monotonicity(_Fit(), 0, grid, q),
        lambda grid, q: check_bifactor_expected_total_score_monotonicity(_Fit(), grid, q),
    ),
    ids=("focal", "bifactor"),
)
def test_public_checks_return_finite_curves_at_high_q(check) -> None:
    """The observed bug was the verdict: an all-NaN curve reported monotone."""
    grid = np.linspace(-3.0, 3.0, 7)
    high = check(grid, 481)
    assert np.all(np.isfinite(high.expected_total))
    assert high.monotone
    np.testing.assert_allclose(high.expected_total, check(grid, 121).expected_total, atol=1e-9)
