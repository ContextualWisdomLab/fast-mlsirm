"""Coverage for the MH-RM confirmatory IFA fitter validation paths (mhrm.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.mhrm import MhrmFit, fit_mhrm
from fast_mlsirm.models import confirmatory


def _binary(seed=0, n_persons=60, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def _poly(n_cat=3, seed=1, n_persons=60, n_items=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    for k in range(n_cat):
        y[k, :] = k
    return y


def _short(**kw):
    base = dict(max_cycles=40, burn_in=10, mh_steps=2)
    base.update(kw)
    return base


def test_fit_mhrm_2pl_with_standard_errors():
    fit = fit_mhrm(_binary(), 1, family="2pl", estimate_se=True, **_short())
    assert isinstance(fit, MhrmFit)
    assert fit.family == "2pl"
    assert fit.loading.shape == (6, 1)
    assert fit.se_loading.shape == (6, 1)
    assert fit.step.size == 0
    assert fit.n_dims == 1


def test_fit_mhrm_gpcm_with_standard_errors():
    fit = fit_mhrm(_poly(), 1, family="gpcm", n_cat=3, estimate_se=True, **_short())
    assert fit.family == "gpcm"
    assert fit.step.shape == (6, 2)
    assert fit.se_step.shape == (6, 2)


def test_fit_mhrm_gpcm_without_standard_errors():
    fit = fit_mhrm(_poly(), 1, family="gpcm", n_cat=3, estimate_se=False, **_short())
    assert fit.step.shape == (6, 2)
    assert fit.se_loading.size == 0


def test_fit_mhrm_all_missing_skips_category_check():
    fit = fit_mhrm(np.full((10, 4), np.nan), 1, **_short(max_cycles=20, burn_in=5))
    assert isinstance(fit, MhrmFit)


def test_fit_mhrm_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_mhrm(_binary(), 1)


def test_fit_mhrm_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_mhrm(np.zeros(6), 1)


def test_fit_mhrm_rejects_dims_over_cap():
    with pytest.raises(ValueError):
        fit_mhrm(np.zeros((5, 65)), confirmatory(np.eye(65, dtype=np.int64)))


def test_fit_mhrm_rejects_non_integer_cycle_counts():
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, max_cycles=1.5)
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, burn_in=np.array([10]))


def test_fit_mhrm_rejects_bad_seed():
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, seed=True)
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, seed=1.5)
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, seed=-1)
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, seed=2**64)


def test_fit_mhrm_rejects_non_finite_control():
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, tol=np.inf)


def test_fit_mhrm_rejects_bad_family():
    with pytest.raises(ValueError):
        fit_mhrm(_binary(), 1, family="foo")


def test_fit_mhrm_rejects_bad_n_cat_for_gpcm():
    with pytest.raises(ValueError):
        fit_mhrm(_poly(), 1, family="gpcm", n_cat=2.5)
    with pytest.raises(ValueError):
        fit_mhrm(_poly(), 1, family="gpcm", n_cat=1)


def test_fit_mhrm_rejects_bad_2pl_values():
    y = _binary()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        fit_mhrm(y, 1, family="2pl", **_short())


def test_fit_mhrm_rejects_bad_gpcm_values():
    y = _poly()
    y[5, 0] = 0.5
    with pytest.raises(ValueError):
        fit_mhrm(y, 1, family="gpcm", n_cat=3, **_short())
    y2 = _poly()
    y2[5, 0] = 9.0
    with pytest.raises(ValueError):
        fit_mhrm(y2, 1, family="gpcm", n_cat=3, **_short())
