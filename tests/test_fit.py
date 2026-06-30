import numpy as np
import pytest
from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit, _adam, _lbfgs, _unpack
from fast_mlsirm.types import MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad

def test_adam_nan():
    def obj(x):
        return np.nan, np.zeros_like(x), np.nan
    x, _, _, status = _adam(np.zeros(1), obj, FitConfig(), 10)
    assert status == "nan_or_inf"

def test_lbfgs_line_search_failed():
    def obj(x):
        return 10.0, np.ones_like(x), -10.0
    x, _, _, status = _lbfgs(np.zeros(1), obj, FitConfig(), 10)
    assert status == "line_search_failed"

def test_fit_invalid_restarts():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    factor_id = np.array([0, 1])
    with pytest.raises(ValueError):
        fit(y, factor_id, config=FitConfig(n_restarts=0))

def test_adam_max_iter_reached():
    def obj(x):
        return float(x[0]), np.ones_like(x), -10.0
    x, _, _, status = _adam(np.array([100.0]), obj, FitConfig(tolerance=1e-8), 10)
    assert status == "max_iter_reached"

def test_adam_converged():
    def obj(x):
        return 0.0, np.zeros_like(x), 0.0
    x, trace, _, status = _adam(np.array([10.0]), obj, FitConfig(tolerance=1e-1), 100)
    assert status == "converged"

def test_lbfgs_max_iter_reached():
    def obj(x):
        return 10.0, np.ones_like(x) * 1e-5, -10.0
    x, _, _, status = _lbfgs(np.zeros(1), obj, FitConfig(tolerance=1e-8), 10)
    assert status == "max_iter_reached"

def test_lbfgs_converged():
    def obj(x):
        return 0.0, np.zeros_like(x), 0.0
    x, trace, _, status = _lbfgs(np.array([10.0]), obj, FitConfig(tolerance=1e-1), 100)
    assert status == "converged"

def test_make_objective_clip():
    from fast_mlsirm.fit import _make_objective
    from fast_mlsirm.types import MLSIRMParams
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    factor_id = np.array([0, 0])
    params = MLSIRMParams(
        theta=np.zeros((2, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    observed = np.ones_like(y, dtype=bool)
    obj_fn = _make_objective(y, observed, factor_id, params, FitConfig(gradient_clip=1e-8, model="ULS2PLM"))
    x = np.zeros(2*1 + 2 + 2 + 2*1 + 2*1 + 1)
    obj, grad, loglik = obj_fn(x)
    assert np.linalg.norm(grad) <= 1e-8 + 1e-12

def test_unpack_missing_alpha():
    from fast_mlsirm.fit import _unpack
    from fast_mlsirm.types import MLSIRMParams
    template = MLSIRMParams(
        theta=np.zeros((2, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    x = np.zeros(2*1 + 2 + 2*1 + 2*1 + 1)
    res = _unpack(x, template, "ULSRM")
    assert res.alpha.sum() == 0.0

def test_fit_uls2plm_zeroes_factors():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    factor_id = np.array([0, 1])
    res = fit(y, factor_id, config=FitConfig(model="ULS2PLM", max_iter=1, n_restarts=1, optimizer="adam"))
    assert res.model == "ULS2PLM"

def test_adam_nan_return():
    def obj(x):
        return np.inf, np.zeros_like(x), np.inf
    x, _, _, status = _adam(np.zeros(1), obj, FitConfig(), 10)
    assert status == "nan_or_inf"

def test_lbfgs_direction_sy_yy():
    from fast_mlsirm.fit import _lbfgs_direction
    grad = np.array([1.0, 1.0])
    s_hist = [np.array([1.0, 0.0])]
    y_hist = [np.array([1.0, 1.0])]
    rho_hist = [1.0]
    direction = _lbfgs_direction(grad, s_hist, y_hist, rho_hist)
    assert direction.shape == (2,)

def test_lbfgs_direction_history_limit():
    def obj(x):
        return float(x[0]**2), 2*x, float(-x[0]**2)
    _lbfgs(np.array([10.0]), obj, FitConfig(tolerance=1e-8, lbfgs_history=1), 5)

def test_fit_best_is_none():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    factor_id = np.array([0, 0])
    from unittest.mock import patch
    with patch('fast_mlsirm.config.FitConfig.validate'):
        with pytest.raises(RuntimeError, match="Optimization failed to find a valid fit."):
            fit(y, factor_id, config=FitConfig(n_restarts=0))

def test_lbfgs_direction_fallback():
    def obj(x):
        return 10.0, np.array([1.0]), -10.0
    from unittest.mock import patch
    with patch('fast_mlsirm.fit._lbfgs_direction', return_value=np.array([-1.0])):
        x, _, _, status = _lbfgs(np.zeros(1), obj, FitConfig(tolerance=1e-8), 10)
        assert status == "line_search_failed"

def test_lbfgs_direction_history_limit_pop():
    def obj(x):
        # We need a quadratic so gradient changes
        return float(x[0]**2), 2*x, float(-x[0]**2)
    # Give it history size 0 so it pops immediately
    _lbfgs(np.array([10.0]), obj, FitConfig(tolerance=1e-8, lbfgs_history=0), 5)
