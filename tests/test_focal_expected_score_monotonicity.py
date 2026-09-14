"""Contract for :func:`focal_expected_total_score_monotonicity`.

The load-bearing claim is that marginalizing over any number of nuisance
dimensions collapses to one Gaussian integral per item, because the model is
compensatory and the fitted prior is independent standard normal. That is an
algebraic identity rather than an approximation, so it is checked against a
brute-force Monte Carlo integration over the nuisance dimensions directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import focal_expected_total_score_monotonicity

N_CAT = 4
THRESHOLDS = np.array([1.2, 0.0, -1.2])


class _Fit:
    """The two fields the diagnostic reads from a multidimensional graded fit."""

    def __init__(self, slope: np.ndarray) -> None:
        self.slope = np.asarray(slope, dtype=np.float64)
        self.threshold = np.tile(THRESHOLDS, (self.slope.shape[0], 1))


def _expected_item(base: np.ndarray) -> np.ndarray:
    ge = np.ones(base.shape + (N_CAT + 1,))
    for k in range(1, N_CAT):
        ge[..., k] = 1.0 / (1.0 + np.exp(-(base + THRESHOLDS[k - 1])))
    ge[..., N_CAT] = 0.0
    cells = ge[..., :N_CAT] - ge[..., 1:]
    return (cells * np.arange(N_CAT)).sum(-1)


def _monte_carlo_curve(slope: np.ndarray, focal: int, grid: np.ndarray, draws: int):
    """Integrate the nuisance dimensions by sampling the prior directly."""
    rng = np.random.default_rng(20_260_915)
    nuisance = np.delete(slope, focal, axis=1)
    z = rng.standard_normal((draws, nuisance.shape[1]))
    curve = np.zeros(grid.size)
    for item in range(slope.shape[0]):
        offset = z @ nuisance[item]
        curve += np.array(
            [_expected_item(slope[item, focal] * t + offset).mean() for t in grid]
        )
    return curve


def test_the_nuisance_collapse_matches_direct_integration() -> None:
    """Three nuisance dimensions, integrated two ways."""
    slope = np.array(
        [
            [1.30, -0.70, 0.50, 0.20],
            [0.90, 0.40, -0.60, 0.30],
            [1.10, 0.25, 0.35, -0.45],
        ]
    )
    grid = np.linspace(-3.0, 3.0, 13)

    report = focal_expected_total_score_monotonicity(_Fit(slope), 0, grid)
    reference = _monte_carlo_curve(slope, 0, grid, draws=200_000)

    # 200k draws over three dimensions; the residual is sampling noise, and the
    # quadrature is the exact side of the comparison.
    np.testing.assert_allclose(report.expected_total, reference, atol=5e-3)


def test_a_negative_focal_slope_is_what_makes_the_curve_decrease() -> None:
    """The derivative constrains the FOCAL column only."""
    slope = np.array([[0.30, -1.80], [0.25, 1.60], [0.20, -1.40], [-2.40, 0.90]])
    grid = np.linspace(-4.0, 4.0, 401)

    focal_zero = focal_expected_total_score_monotonicity(_Fit(slope), 0, grid)
    assert not focal_zero.monotone
    assert focal_zero.total_decrease > 0.0


def test_large_negative_nuisance_loadings_do_not_make_the_focal_curve_decrease() -> None:
    """The distinction the docstring draws, asserted rather than described."""
    slope = np.array([[1.30, -1.90], [1.10, -1.70], [0.95, -2.10], [1.25, -1.50]])
    grid = np.linspace(-4.0, 4.0, 401)

    report = focal_expected_total_score_monotonicity(_Fit(slope), 0, grid)

    assert report.monotone
    assert report.total_decrease == 0.0


def test_the_report_is_stable_under_grid_and_quadrature_refinement() -> None:
    slope = np.array([[1.30, -0.70, 0.50], [0.90, 0.40, -0.60], [1.10, 0.25, 0.35]])

    totals = [
        focal_expected_total_score_monotonicity(
            _Fit(slope), 0, np.linspace(-4.0, 4.0, points), q_nuisance=nodes
        ).total_decrease
        for points, nodes in ((201, 21), (401, 41), (801, 81))
    ]

    assert max(totals) - min(totals) < 1e-9


@pytest.mark.parametrize("dimension", [-1, 3])
def test_the_focal_dimension_must_exist(dimension: int) -> None:
    slope = np.array([[1.0, 0.5, 0.2]])
    with pytest.raises(ValueError, match="dimension"):
        focal_expected_total_score_monotonicity(
            _Fit(slope), dimension, np.linspace(-2.0, 2.0, 9)
        )
