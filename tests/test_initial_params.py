import numpy as np

from fast_mlsirm import FitConfig
from fast_mlsirm.fit import _initial_params
from fast_mlsirm.math import standardize


def test_initial_params_vectorized_theta_matches_reference_loop():
    y = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    observed = np.array(
        [
            [True, False, False, True],
            [True, True, False, True],
            [False, False, False, False],
        ]
    )
    factor_id = np.array([0, 0, 1, 1])

    params = _initial_params(
        y,
        observed,
        factor_id,
        n_dims=2,
        latent_dim=1,
        config=FitConfig(seed=7),
        rng=np.random.default_rng(7),
    )

    expected = np.zeros((3, 2))
    for dim in range(2):
        items = factor_id == dim
        denom = np.maximum(observed[:, items].sum(axis=1), 1)
        expected[:, dim] = standardize((y[:, items] * observed[:, items]).sum(axis=1) / denom)

    assert np.allclose(params.theta, expected)
