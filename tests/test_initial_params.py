import importlib

import numpy as np

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import _initial_params
from fast_mlsirm.math import standardize

def test_initial_params_matches_loop():
    n_persons = 50
    n_items = 20
    n_dims = 3
    latent_dim = 2

    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.5, size=(n_persons, n_items)).astype(np.float64)
    observed = rng.binomial(1, 0.8, size=(n_persons, n_items)).astype(bool)
    factor_id = rng.integers(0, n_dims, size=n_items)

    config = FitConfig(latent_dim=latent_dim)

    rng_vect = np.random.default_rng(42)
    params_vect = _initial_params(y, observed, factor_id, n_dims, latent_dim, config, rng_vect)

    theta_loop = np.zeros((n_persons, n_dims), dtype=np.float64)
    for d in range(n_dims):
        items = factor_id == d
        denom = np.maximum(observed[:, items].sum(axis=1), 1)
        x = (y[:, items] * observed[:, items]).sum(axis=1) / denom
        theta_loop[:, d] = standardize(x)

    np.testing.assert_allclose(params_vect.theta, theta_loop, atol=1e-10)

def test_initial_params_missing_dimension():
    n_persons = 10
    n_items = 5
    n_dims = 4
    latent_dim = 1

    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.5, size=(n_persons, n_items)).astype(np.float64)
    observed = np.ones((n_persons, n_items), dtype=bool)
    factor_id = np.array([0, 1, 0, 2, 1])

    config = FitConfig(latent_dim=latent_dim)

    params_vect = _initial_params(y, observed, factor_id, n_dims, latent_dim, config, rng)

    np.testing.assert_allclose(params_vect.theta[:, 3], np.zeros(n_persons))


def test_initial_params_uses_rust_theta_for_rust_backend(monkeypatch):
    fit_module = importlib.import_module("fast_mlsirm.fit")
    n_persons = 3
    n_items = 4
    n_dims = 2
    latent_dim = 1
    y = np.zeros((n_persons, n_items), dtype=np.float64)
    observed = np.ones((n_persons, n_items), dtype=bool)
    factor_id = np.array([0, 1, 0, 1])
    expected_theta = np.arange(n_persons * n_dims, dtype=np.float64).reshape(
        n_persons, n_dims
    )
    calls = []

    def fake_initial_theta_rust(*args):
        calls.append(args)
        return expected_theta

    monkeypatch.setattr(fit_module, "_initial_theta_rust", fake_initial_theta_rust)

    params = _initial_params(
        y,
        observed,
        factor_id,
        n_dims,
        latent_dim,
        FitConfig(latent_dim=latent_dim),
        np.random.default_rng(7),
        backend="rust",
    )

    np.testing.assert_allclose(params.theta, expected_theta)
    assert len(calls) == 1
