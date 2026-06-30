import numpy as np
import pytest

from fast_mlsirm.objective import prepare_response, validate_factor_id, model_flags, linear_predictor, neg_loglik_and_grad
from fast_mlsirm.config import FitConfig, MLS2PLMConfig, PenaltyConfig
from fast_mlsirm.types import MLSIRMParams
from fast_mlsirm.math import standardize, normalize_latent_positions
from fast_mlsirm.diagnostics import predict_proba, align_latent_space, _corr
from fast_mlsirm.fit import fit, _adam, _lbfgs, _pack

def test_missing_obj():
    with pytest.raises(ValueError):
        prepare_response(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        prepare_response(np.array([[1, 2], [3, 4]]), mask=np.array([[True]]))
    with pytest.raises(ValueError):
        prepare_response(np.array([[np.nan]]), mask=None)
    with pytest.raises(ValueError):
        prepare_response(np.array([[2.0]]))
    with pytest.raises(ValueError):
        prepare_response(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        prepare_response(np.array([[np.nan], [1.0]]))
    with pytest.raises(ValueError):
        validate_factor_id(np.array([0]), 2, 1)
    with pytest.raises(ValueError):
        validate_factor_id(np.array([-1]), 1, 1)
    with pytest.raises(ValueError):
        validate_factor_id(np.array([1]), 1, 1)
    params = MLSIRMParams(theta=np.zeros((1, 2)), alpha=np.zeros(1), b=np.zeros(1), xi=np.zeros((1, 1)), zeta=np.zeros((1, 1)), tau=0.0)
    with pytest.raises(ValueError):
        neg_loglik_and_grad(np.array([[1.0]]), np.array([0]), params, FitConfig(model="ULS2PLM"))

def test_missing_cfg():
    with pytest.raises(ValueError):
        MLS2PLMConfig(n_persons=0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(n_dims=0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(items_per_dim=0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(latent_dim=0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(phi=-2.0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(gamma=-1.0).validate()
    with pytest.raises(ValueError):
        MLS2PLMConfig(dtype="int32").validate()
    with pytest.raises(ValueError):
        FitConfig(model="UNKNOWN").validate()
    with pytest.raises(ValueError):
        FitConfig(latent_dim=0).validate()
    with pytest.raises(ValueError):
        FitConfig(optimizer="UNKNOWN").validate()
    with pytest.raises(ValueError):
        FitConfig(max_iter=0).validate()
    with pytest.raises(ValueError):
        FitConfig(n_restarts=0).validate()
    with pytest.raises(ValueError):
        FitConfig(learning_rate=0.0).validate()
    with pytest.raises(ValueError):
        FitConfig(init_gamma=0.0).validate()
    with pytest.raises(ValueError):
        FitConfig(eps_distance=0.0).validate()

def test_missing_diag():
    params = MLSIRMParams(theta=np.zeros((5, 2)), alpha=np.zeros(5), b=np.zeros(5), xi=np.zeros((5, 2)), zeta=np.zeros((5, 2)), tau=0.0)
    probs = predict_proba(params, np.array([0, 1, 0, 1, 0]), persons=np.array([0, 1]), items=np.array([0, 2]))
    assert probs.shape == (2, 2)
    with pytest.raises(ValueError):
        align_latent_space(params.xi, params.zeta, params.xi, params.zeta, method="unknown")
    assert np.isnan(_corr(np.zeros(5), np.random.randn(5)))
    assert np.isnan(_corr(np.random.randn(5), np.zeros(5)))

def test_missing_math():
    assert np.all(standardize(np.zeros(5)) == 0.0)
    params = MLSIRMParams(theta=np.zeros((5, 2)), alpha=np.zeros(5), b=np.zeros(5), xi=np.zeros((0, 2)), zeta=np.zeros((5, 2)), tau=0.0)
    out = normalize_latent_positions(params)
    assert out.xi.size == 0

def test_missing_fit():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    factor_id = np.array([0, 0])
    fit(y, factor_id, config=FitConfig(model="MLSRM", optimizer="adam", max_iter=1, n_restarts=1))
    fit(y, factor_id, config=FitConfig(model="ULSRM", optimizer="adam", max_iter=1, n_restarts=1))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="adam", max_iter=1000, n_restarts=1, tolerance=1.0))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="lbfgs", max_iter=1000, n_restarts=1, tolerance=1.0))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="lbfgs", max_iter=5, n_restarts=1, init_gamma=100.0))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="lbfgs", max_iter=10, n_restarts=1, lbfgs_history=1))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="lbfgs", max_iter=2, n_restarts=1, tolerance=1e-12))
    fit(y, factor_id, config=FitConfig(model="MIRT", optimizer="adam_lbfgs", max_iter=2, n_restarts=1))
    y2 = np.array([[1.0, -1.0], [0.0, 1.0]])
    fit(y2, factor_id, config=FitConfig(model="ULS2PLM", optimizer="adam_lbfgs", max_iter=2, n_restarts=1))
    fit(y, factor_id, config=FitConfig(model="ULS2PLM", optimizer="adam", max_iter=2, n_restarts=1, gradient_clip=1e-8))

    params = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=0.0)
    cfg = FitConfig(model="ULS2PLM")

    def bad_obj(x):
        return np.nan, np.zeros_like(x), np.nan
    x0 = np.zeros(_pack(params, "ULS2PLM").shape)
    _adam(x0, bad_obj, cfg, max_iter=2)

    def bad_obj2(x):
        return 0.0, np.ones_like(x) * 100.0, 0.0
    _lbfgs(x0, bad_obj2, cfg, max_iter=2)

    orig_validate = FitConfig.validate
    FitConfig.validate = lambda self: None
    try:
        with pytest.raises(RuntimeError):
            fit(y, factor_id, config=FitConfig(model="ULS2PLM", n_restarts=0))
    finally:
        FitConfig.validate = orig_validate

def test_lbfgs_bad_direction_trigger(monkeypatch):
    import sys; target = sys.modules['fast_mlsirm.fit']
    monkeypatch.setattr(target, "_lbfgs_direction", lambda g, s, y, r: np.array([-1.0]))
    cfg = FitConfig(model="ULS2PLM")
    target._lbfgs(np.array([1.0]), lambda x: (0.0, np.array([1.0]), 0.0), cfg, max_iter=2)
