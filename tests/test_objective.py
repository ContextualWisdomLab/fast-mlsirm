import pytest
import numpy as np

from fast_mlsirm import FitConfig, MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad


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
def test_prepare_response_invalid():
    from fast_mlsirm.objective import prepare_response
    # test not 2D
    with pytest.raises(ValueError):
        prepare_response(np.array([1.0, 0.0]))

    # test shape mismatch mask
    with pytest.raises(ValueError):
        prepare_response(np.array([[1.0, 0.0]]), mask=np.array([True]))

    # test no observed entries
    with pytest.raises(ValueError):
        prepare_response(np.array([[-1.0, np.nan]]))

    # test invalid response values
    with pytest.raises(ValueError):
        prepare_response(np.array([[1.0, 2.0]]))

    # test all missing item
    with pytest.raises(ValueError):
        prepare_response(np.array([[1.0, -1.0], [0.0, -1.0]]))

    # test all missing person
    with pytest.raises(ValueError):
        prepare_response(np.array([[-1.0, -1.0], [0.0, 1.0]]))

def test_validate_factor_id_invalid():
    from fast_mlsirm.objective import validate_factor_id
    with pytest.raises(ValueError):
        validate_factor_id(np.array([0, 1]), n_items=3, n_dims=2)
    with pytest.raises(ValueError):
        validate_factor_id(np.array([0, 2]), n_items=2, n_dims=2)

def test_neg_loglik_and_grad_invalid_model():
    from fast_mlsirm.objective import neg_loglik_and_grad
    params = MLSIRMParams(
        theta=np.zeros((2, 2)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0,
    )
    with pytest.raises(ValueError):
        neg_loglik_and_grad(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), params, FitConfig(model="ULSRM"))
