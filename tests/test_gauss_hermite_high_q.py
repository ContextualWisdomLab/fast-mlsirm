"""High node-count Gauss-Hermite contract for the NumPy marginal path (#2110).

``numpy.polynomial.hermite_e.hermegauss`` overflows its weight formula from
q = 371 (weights sum to 0 or NaN), which turned every marginal likelihood
evaluated at a study-setting node count >= 371 into NaN. ``marginal._gh`` must
return a finite unit-sum rule for any caller-selected q (#1929: no node cap)
and must stay bit-identical to the normalized ``hermegauss`` rule wherever
that rule is finite, so existing results for q <= 370 do not move.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators import marginal


def _normalized_hermegauss(q: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    return nodes, weights / weights.sum()


@pytest.mark.parametrize("q", [371, 400, 481, 961, 1000])
def test_high_q_rule_is_finite_unit_mass_and_integrates_normal_moments(q: int) -> None:
    nodes, weights = marginal._gh(q)

    assert nodes.shape == weights.shape == (q,)
    assert np.isfinite(nodes).all()
    assert np.isfinite(weights).all()
    assert (weights >= 0.0).all()
    assert np.all(np.diff(nodes) > 0.0)
    np.testing.assert_allclose(nodes, -nodes[::-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights, weights[::-1], rtol=0.0, atol=0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-13)
    # A q-point Gauss rule is exact for polynomials of degree <= 2q - 1, so the
    # N(0, 1) moments E[x^2] = 1, E[x^4] = 3, E[x^6] = 15 must be reproduced
    # up to floating-point roundoff (observed <= 1e-13 for these q).
    assert float(weights @ nodes**2) == pytest.approx(1.0, abs=1e-12)
    assert float(weights @ nodes**4) == pytest.approx(3.0, abs=1e-12)
    assert float(weights @ nodes**6) == pytest.approx(15.0, abs=1e-11)


@pytest.mark.parametrize("q", [1, 2, 7, 21, 121, 241, 370])
def test_rule_is_bit_identical_to_normalized_hermegauss_where_finite(q: int) -> None:
    expected_nodes, expected_weights = _normalized_hermegauss(q)
    nodes, weights = marginal._gh(q)

    np.testing.assert_array_equal(nodes, expected_nodes)
    np.testing.assert_array_equal(weights, expected_weights)


def test_marginal_log_weights_are_finite_at_high_q() -> None:
    _, weights = marginal._gh(481)
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    # Underflowed far-tail weights may be exactly zero (log = -inf, which the
    # log-sum-exp E-step treats as zero mass); no weight may be NaN.
    assert not np.isnan(log_weights).any()
    assert np.isfinite(log_weights).sum() > 0


@pytest.mark.parametrize("q", [0, -3])
def test_nonpositive_node_count_is_rejected(q: int) -> None:
    with pytest.raises(ValueError, match="q must be >= 1"):
        marginal._gh(q)
