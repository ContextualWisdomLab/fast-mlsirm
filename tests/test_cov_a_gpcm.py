"""Coverage for the generalized partial credit model fitter (gpcm.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.gpcm import GpcmFit, fit_gpcm
from fast_mlsirm.models import confirmatory


def _poly(n_cat=3, seed=1, n_persons=45, n_items=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    for k in range(n_cat):
        y[k, :] = k
    return y


def test_fit_gpcm_happy_path_marshals_result():
    fit = fit_gpcm(_poly(), 3, 1, q=7, max_iter=5)
    assert isinstance(fit, GpcmFit)
    assert fit.slope.shape == (4, 1)
    assert fit.step.shape == (4, 2)
    assert fit.theta.shape == (45, 1)
    assert fit.n_cat == 3
    assert fit.n_dims == 1


def test_fit_gpcm_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_gpcm(_poly(), 3, 1, q=7)


def test_fit_gpcm_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_gpcm(np.zeros(4), 3)


def test_fit_gpcm_rejects_dims_over_gh_cap():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(n_items=4), 3, confirmatory(np.eye(4, dtype=np.int64)), q=7)


def test_fit_gpcm_rejects_bad_n_cat_type():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), np.array([3, 3]))
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3j)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3.5)


def test_fit_gpcm_rejects_n_cat_out_of_range():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 1)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 65)


def test_fit_gpcm_rejects_unsupported_q():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=8)


def test_fit_gpcm_rejects_max_iter_out_of_range():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, max_iter=0)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, max_iter=200_000)


def test_fit_gpcm_rejects_xi_points_out_of_range():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_points=0)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_points=200_001)


def test_fit_gpcm_rejects_bad_xi_seed():
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_seed=True)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_seed=1.5)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_seed=-1)
    with pytest.raises(ValueError):
        fit_gpcm(_poly(), 3, 1, q=7, xi_seed=2**64)


def test_fit_gpcm_all_missing_skips_category_check():
    y = np.full((10, 4), np.nan)
    with pytest.raises(ValueError):
        fit_gpcm(y, 3, 1, q=7, max_iter=3)


def test_fit_gpcm_rejects_bad_category_values():
    y = _poly(3)
    y[5, 0] = 0.5
    with pytest.raises(ValueError):
        fit_gpcm(y, 3, 1, q=7, max_iter=5)
    y2 = _poly(3)
    y2[5, 0] = 3
    with pytest.raises(ValueError):
        fit_gpcm(y2, 3, 1, q=7, max_iter=5)
