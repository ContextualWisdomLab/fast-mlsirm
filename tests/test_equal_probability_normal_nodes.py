"""Tests for equal-probability standard-normal quadrature nodes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.stats import norm

from fast_mlsirm.estimators.mmle import (
    equal_probability_normal_nodes,
    gauss_hermite_nodes,
)


def test_equal_probability_normal_nodes_n5() -> None:
    nodes, weights = equal_probability_normal_nodes(5)
    expected_nodes = norm.ppf((np.arange(1, 6) - 0.5) / 5)
    np.testing.assert_allclose(nodes, expected_nodes)
    np.testing.assert_allclose(weights, 0.2)
    assert weights.sum() == pytest.approx(1.0)
    gh_nodes, _ = gauss_hermite_nodes(5)
    assert not np.allclose(nodes, gh_nodes)


def test_equal_probability_normal_nodes_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n_nodes must be >= 1"):
        equal_probability_normal_nodes(0)
