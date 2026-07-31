"""Coverage for the rating scale model fitter validation paths (rsm.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.rsm import RsmFit, fit_rsm


def _poly(n_cat=3, seed=0, n_persons=30, n_items=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    for k in range(n_cat):
        y[k, :] = k
    return y


def test_fit_rsm_happy_path_explicit_n_cat():
    fit = fit_rsm(_poly(), n_cat=3, q_theta=7, max_iter=5)
    assert isinstance(fit, RsmFit)
    assert fit.item_location.shape == (4,)
    assert fit.thresholds.shape == (2,)
    assert fit.theta.shape == (30,)


def test_fit_rsm_happy_path_inferred_n_cat():
    fit = fit_rsm(_poly(), q_theta=7, max_iter=5)
    assert isinstance(fit, RsmFit)


def test_fit_rsm_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_rsm(_poly(), n_cat=3, q_theta=7)


def test_fit_rsm_rejects_bad_n_cat_type():
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=2.5)
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=True)


def test_fit_rsm_rejects_n_cat_range():
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=1)
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=65)


def test_fit_rsm_rejects_bad_q_theta():
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=3, q_theta=8)


def test_fit_rsm_rejects_bad_max_iter():
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=3, q_theta=7, max_iter=0)
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=3, q_theta=7, max_iter=True)


def test_fit_rsm_rejects_bad_tol():
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=3, q_theta=7, tol=0.0)
    with pytest.raises(ValueError):
        fit_rsm(_poly(), n_cat=3, q_theta=7, tol=np.inf)


def test_fit_rsm_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_rsm(np.zeros(4), n_cat=3, q_theta=7)


def test_fit_rsm_rejects_empty():
    with pytest.raises(ValueError):
        fit_rsm(np.zeros((0, 4)), n_cat=3, q_theta=7)


def test_fit_rsm_rejects_non_finite_observed():
    y = _poly()
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_rsm(y, n_cat=3, q_theta=7)


def test_fit_rsm_rejects_non_integer_or_negative():
    y = _poly()
    y[5, 0] = 0.5
    with pytest.raises(ValueError):
        fit_rsm(y, n_cat=3, q_theta=7)
    y2 = _poly()
    y2[5, 0] = -1.0
    with pytest.raises(ValueError):
        fit_rsm(y2, n_cat=3, q_theta=7)


def test_fit_rsm_inference_no_observed_values():
    with pytest.raises(ValueError):
        fit_rsm(np.full((10, 4), np.nan), q_theta=7)


def test_fit_rsm_inference_single_category():
    with pytest.raises(ValueError):
        fit_rsm(np.zeros((10, 4)), q_theta=7)


def test_fit_rsm_inference_too_many_categories():
    y = np.zeros((10, 4))
    y[0, 0] = 64.0
    with pytest.raises(ValueError):
        fit_rsm(y, q_theta=7)


def test_fit_rsm_rejects_category_over_declared_range():
    y = _poly()
    y[5, 0] = 3.0
    with pytest.raises(ValueError):
        fit_rsm(y, n_cat=3, q_theta=7)


def test_fit_rsm_rejects_item_with_no_observed():
    y = _poly()
    y[:, 3] = np.nan
    with pytest.raises(ValueError):
        fit_rsm(y, n_cat=3, q_theta=7)
