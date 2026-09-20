"""Tests for equal-probability standard-normal quadrature nodes."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators.mmle import (
    equal_probability_normal_nodes,
    gauss_hermite_nodes,
)

# Standard-normal mid-bin quantiles q = Phi^{-1}((i - 0.5)/n), i = 1..n, pinned
# independently (mpmath erfinv, 80 dps) and stored at float64 resolution.
_EQPROB_NORMAL_NODES: dict[int, tuple[np.ndarray, float]] = {
    1: (np.array([0.0], dtype=np.float64), 1.0),
    2: (
        np.array(
            [-0.6744897501960817, 0.6744897501960817],
            dtype=np.float64,
        ),
        0.5,
    ),
    5: (
        np.array(
            [
                -1.2815515655446004,
                -0.5244005127080409,
                0.0,
                0.5244005127080407,
                1.2815515655446004,
            ],
            dtype=np.float64,
        ),
        0.2,
    ),
    11: (
        np.array(
            [
                -1.6906216295848977,
                -1.0968035620935128,
                -0.7478585947633022,
                -0.4727891209922674,
                -0.22988411757923205,
                0.0,
                0.2298841175792322,
                0.4727891209922672,
                0.7478585947633022,
                1.0968035620935128,
                1.6906216295848988,
            ],
            dtype=np.float64,
        ),
        1.0 / 11.0,
    ),
}


@pytest.mark.parametrize("n_nodes", sorted(_EQPROB_NORMAL_NODES))
def test_equal_probability_normal_nodes_match_pinned_mid_bin_quantiles(
    n_nodes: int,
) -> None:
    expected_nodes, expected_weight = _EQPROB_NORMAL_NODES[n_nodes]
    nodes, weights = equal_probability_normal_nodes(n_nodes)

    np.testing.assert_allclose(nodes, expected_nodes, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights, expected_weight)
    assert np.all(np.isfinite(nodes))
    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(1.0)

    for i, node in enumerate(nodes):
        assert node == pytest.approx(-nodes[n_nodes - 1 - i])

    if n_nodes > 1:
        assert np.all(np.diff(nodes) > 0.0)

    for odd_power in (1, 3, 5):
        moment = float(np.sum(weights * nodes**odd_power))
        assert moment == pytest.approx(0.0, abs=1e-14)


def test_equal_probability_normal_nodes_differs_from_gauss_hermite() -> None:
    nodes, _ = equal_probability_normal_nodes(5)
    gh_nodes, _ = gauss_hermite_nodes(5)
    assert not np.allclose(nodes, gh_nodes)


@pytest.mark.parametrize("invalid", [0, -1])
def test_equal_probability_normal_nodes_rejects_invalid_count(invalid: int) -> None:
    with pytest.raises(ValueError, match="n_nodes must be >= 1"):
        equal_probability_normal_nodes(invalid)
