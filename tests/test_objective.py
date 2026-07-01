import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad, validate_factor_id


def test_missing_entries_are_excluded():
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.1]]),
        alpha=np.array([0.0, 0.1]),
        b=np.array([0.0, 0.2]),
        xi=np.array([[0.1, 0.2], [-0.2, 0.3]]),
        zeta=np.array([[0.0, 0.0], [0.2, -0.1]]),
        tau=0.0,
    )
    y = np.array([[1.0, -1.0], [0.0, 1.0]])
    full_obj, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), params, FitConfig(max_iter=1))

    y2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, False], [True, True]])
    mask_obj, _, _ = neg_loglik_and_grad(y2, np.array([0, 0]), params, FitConfig(max_iter=1), mask=mask)
    assert np.isclose(full_obj, mask_obj)


def test_gradient_matches_finite_difference():
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.3]], dtype=float),
        alpha=np.array([0.1, -0.2], dtype=float),
        b=np.array([0.3, -0.1], dtype=float),
        xi=np.array([[0.1, 0.2], [-0.2, 0.4]], dtype=float),
        zeta=np.array([[0.0, -0.1], [0.3, -0.4]], dtype=float),
        tau=0.2,
    )
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    config = FitConfig(max_iter=1)
    base, grad, _ = neg_loglik_and_grad(y, np.array([0, 0]), params, config)

    checks = [
        ("theta", (0, 0), grad.theta[0, 0]),
        ("alpha", (0,), grad.alpha[0]),
        ("b", (1,), grad.b[1]),
        ("xi", (0, 1), grad.xi[0, 1]),
        ("zeta", (1, 0), grad.zeta[1, 0]),
    ]
    h = 1e-6
    for name, idx, analytic in checks:
        trial = params.copy()
        arr = getattr(trial, name)
        arr[idx] += h
        got, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), trial, config)
        assert np.isclose((got - base) / h, analytic, atol=2e-5)

    trial = params.copy()
    trial.tau += h
    got, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), trial, config)
    assert np.isclose((got - base) / h, grad.tau, atol=2e-5)


def test_rust_backend_matches_numpy_objective():
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.3]], dtype=float),
        alpha=np.array([0.1, -0.2], dtype=float),
        b=np.array([0.3, -0.1], dtype=float),
        xi=np.array([[0.1, 0.2], [-0.2, 0.4]], dtype=float),
        zeta=np.array([[0.0, -0.1], [0.3, -0.4]], dtype=float),
        tau=0.2,
    )
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, True], [True, False]])
    factors = np.array([0, 0])
    config = FitConfig(max_iter=1)

    numpy_obj, numpy_grad, numpy_loglik = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="numpy")
    rust_obj, rust_grad, rust_loglik = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="rust")

    assert np.isclose(rust_obj, numpy_obj)
    assert np.isclose(rust_loglik, numpy_loglik)
    assert np.allclose(rust_grad.theta, numpy_grad.theta)
    assert np.allclose(rust_grad.alpha, numpy_grad.alpha)
    assert np.allclose(rust_grad.b, numpy_grad.b)
    assert np.allclose(rust_grad.xi, numpy_grad.xi)
    assert np.allclose(rust_grad.zeta, numpy_grad.zeta)
    assert np.isclose(rust_grad.tau, numpy_grad.tau)


def test_rust_core_rejects_shape_mismatch():
    from fast_mlsirm import _core

    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        _core.neg_loglik_and_grad(
            np.zeros((2, 2), dtype=float),
            np.ones((2, 2), dtype=bool),
            np.array([0], dtype=np.int64),
            np.zeros((2, 1), dtype=float),
            np.zeros(2, dtype=float),
            np.zeros(2, dtype=float),
            np.zeros((2, 1), dtype=float),
            np.zeros((2, 1), dtype=float),
            0.0,
            "MLS2PLM",
            1e-8,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.0,
            0.0,
        )


def test_validate_factor_id():
    res = validate_factor_id([0, 1, 0], n_items=3, n_dims=2)
    assert np.array_equal(res, np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        validate_factor_id([0, 1], n_items=3, n_dims=2)

    with pytest.raises(ValueError, match="factor_id values must be in 0..n_dims-1"):
        validate_factor_id([-1, 0, 1], n_items=3, n_dims=2)

    with pytest.raises(ValueError, match="factor_id values must be in 0..n_dims-1"):
        validate_factor_id([0, 2, 0], n_items=3, n_dims=2)

import pytest
from fast_mlsirm.objective import prepare_response, _add_penalty
from fast_mlsirm.config import PenaltyConfig

def test_objective_check_responses_errors():
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        prepare_response(np.array([1, 0]))

    with pytest.raises(ValueError, match="mask shape must match responses"):
        prepare_response(np.zeros((2, 2)), mask=np.zeros((3, 2)))

    with pytest.raises(ValueError, match="responses contain no observed entries"):
        prepare_response(np.full((2, 2), np.nan))

    with pytest.raises(ValueError, match="observed responses must be 0 or 1"):
        prepare_response(np.full((2, 2), 2.0))

    with pytest.raises(ValueError, match="all-missing item found"):
        prepare_response(np.array([[np.nan, 1], [np.nan, 0]]))

    with pytest.raises(ValueError, match="all-missing person found"):
        prepare_response(np.array([[np.nan, np.nan], [1, 0]]))

def test_objective_model_requires_one_trait():
    from fast_mlsirm.objective import neg_loglik_and_grad
    from fast_mlsirm.config import FitConfig
    params = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)

    with pytest.raises(ValueError, match="ULS2PLM requires one trait dimension"):
        neg_loglik_and_grad(np.zeros((2, 2)), np.zeros(2, dtype=int), params, config=FitConfig(model="ULS2PLM"))

def test_objective_add_penalty_uses_space():
    from fast_mlsirm.types import MLSIRMParams
    params = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)
    penalty = PenaltyConfig(
        lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0, lambda_xi=1.0, lambda_zeta=1.0, lambda_tau=1.0,
        mu_alpha=0.0, mu_tau=0.0
    )
    val = _add_penalty(params, penalty, free_alpha=True, uses_space=True)
    assert val > 0.0
