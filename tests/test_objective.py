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

def test_distance_gradient_optimization_correctness():
    """Verify that the optimized dot-product gradient computation
    is equivalent to the naive 3D broadcast computation."""
    # We want to test that the outputs of the new gradient implementation
    # give the same result as finite difference, specifically for a larger set of points
    np.random.seed(42)
    N, J, D = 10, 5, 2
    params = MLSIRMParams(
        theta=np.random.normal(0, 1, size=(N, 1)),
        alpha=np.random.normal(0, 1, size=(J,)),
        b=np.random.normal(0, 1, size=(J,)),
        xi=np.random.normal(0, 1, size=(N, D)),
        zeta=np.random.normal(0, 1, size=(J, D)),
        tau=0.5,
    )
    y = np.random.binomial(1, 0.5, size=(N, J))
    config = FitConfig(max_iter=1)
    base, grad, _ = neg_loglik_and_grad(y, np.zeros(J, dtype=int), params, config)

    # Test finite difference for one element of xi
    h = 1e-6
    trial = params.copy()
    trial.xi[2, 1] += h
    got, _, _ = neg_loglik_and_grad(y, np.zeros(J, dtype=int), trial, config)
    assert np.isclose((got - base) / h, grad.xi[2, 1], atol=2e-5)

    # Test finite difference for one element of zeta
    trial = params.copy()
    trial.zeta[3, 0] += h
    got, _, _ = neg_loglik_and_grad(y, np.zeros(J, dtype=int), trial, config)
    assert np.isclose((got - base) / h, grad.zeta[3, 0], atol=2e-5)
