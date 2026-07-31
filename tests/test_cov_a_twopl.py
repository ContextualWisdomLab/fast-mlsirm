"""Coverage for the compensatory 2PL fitter validation paths (twopl.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.models import confirmatory
from fast_mlsirm.twopl import TwoPlFit, fit_2pl


def _binary(seed=0, n_persons=60, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def test_fit_2pl_happy_path_marshals_result():
    y = _binary()
    fit = fit_2pl(y, 1, q=7, max_iter=5)
    assert isinstance(fit, TwoPlFit)
    assert fit.loading.shape == (6, 1)
    assert fit.intercept.shape == (6,)
    assert fit.theta.shape == (60, 1)
    assert fit.corr.shape == (1, 1)
    assert fit.n_dims == 1
    assert fit.termination_reason in ("converged", "max_iter_reached")


def test_fit_2pl_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_2pl(_binary(), 1, q=7)


def test_fit_2pl_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_2pl(np.zeros(6), 1)


def test_fit_2pl_rejects_dims_over_gh_cap():
    pattern = np.eye(4, dtype=np.int64)
    pattern = np.vstack([pattern, pattern[:2]])  # 6 items x 4 dims
    with pytest.raises(ValueError):
        fit_2pl(_binary(n_items=6), confirmatory(pattern), q=7)


def test_fit_2pl_rejects_infinite_responses():
    y = _binary(n_items=6)
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_2pl(y, 1, q=7, max_iter=5)


def test_fit_2pl_rejects_non_scalar_or_complex_q():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=np.array([7, 7]))
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7j)


def test_fit_2pl_rejects_non_integer_q():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7.5)


def test_fit_2pl_rejects_unsupported_q():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=8)


def test_fit_2pl_rejects_max_iter_out_of_range():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, max_iter=0)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, max_iter=200_000)


def test_fit_2pl_rejects_xi_points_out_of_range_for_qmc():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, node_rule="qmc", xi_points=0)


def test_fit_2pl_rejects_bad_xi_seed():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=True)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=1.5)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=-1)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=2**64)
