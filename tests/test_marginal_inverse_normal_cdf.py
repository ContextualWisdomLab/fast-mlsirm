"""Regression tests for the NumPy marginal QMC normal-quantile helper."""

from __future__ import annotations

import pytest

from fast_mlsirm.estimators.marginal import _inv_normal_cdf


@pytest.mark.parametrize("probability", (1e-6, 1e-4, 1e-3))
def test_inverse_normal_low_tail_is_antisymmetric(probability: float) -> None:
    """Keep the Acklam lower-tail branch symmetric with the upper-tail branch."""
    assert _inv_normal_cdf(probability) == pytest.approx(
        -_inv_normal_cdf(1.0 - probability),
        rel=1e-12,
        abs=1e-12,
    )
