import numpy as np

from fast_mlsirm import MLS2PLMConfig, simulate
from fast_mlsirm.math import sigmoid


def test_simulate_seed_reproducible():
    config = MLS2PLMConfig(n_persons=20, n_dims=2, items_per_dim=3, latent_dim=2, seed=42)
    a = simulate(config)
    b = simulate(config)

    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.factor_id, b.factor_id)
    assert np.allclose(a.truth.theta, b.truth.theta)
    assert np.allclose(a.truth.a, b.truth.a)
    assert np.allclose(a.truth.b, b.truth.b)
    assert np.allclose(a.truth.xi, b.truth.xi)
    assert np.allclose(a.truth.zeta, b.truth.zeta)


def test_gamma_zero_removes_distance_term():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, gamma=0.0, seed=7))
    manual_eta = data.truth.a[None, :] * data.truth.theta[:, data.factor_id] + data.truth.b[None, :]
    assert np.allclose(data.probabilities, sigmoid(manual_eta))

def test_distance_computation_optimization_correctness():
    """Verify that the optimized dot-product distance computation
    is equivalent to the naive 3D broadcast computation."""
    config = MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, latent_dim=3, seed=42)
    data = simulate(config)

    # Extract ground truth components from data
    xi = data.truth.xi
    zeta = data.truth.zeta

    # Calculate using naive 3D broadcast (the old way)
    diff = xi[:, None, :] - zeta[None, :, :]
    dist_old = np.sqrt(np.sum(diff * diff, axis=2))

    # Calculate using optimized 2D dot product (the new way used in simulate)
    xi_sq = np.sum(xi ** 2, axis=1)
    zeta_sq = np.sum(zeta ** 2, axis=1)
    dist_sq = xi_sq[:, None] + zeta_sq[None, :] - 2 * np.dot(xi, zeta.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist_new = np.sqrt(dist_sq)

    # They should be extremely close
    assert np.allclose(dist_old, dist_new, atol=1e-10)

    # Check that probabilities from simulate match manual calc with dist_new
    manual_eta = data.truth.a[None, :] * data.truth.theta[:, data.factor_id] + data.truth.b[None, :] - config.gamma * dist_new
    assert np.allclose(data.probabilities, sigmoid(manual_eta))
