"""Regression evidence for the NumPy MMLE vectorized M-step fallback."""

import numpy as np

from fast_mlsirm.estimators.mmle import fit_mmle_2pl


def test_vectorized_mmle_matches_pre_vectorization_regression_oracle():
    """Preserve the deterministic scalar-M-step result within numeric tolerance."""
    responses = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, False, True],
            [True, False, True, True],
        ]
    )

    result = fit_mmle_2pl(
        responses,
        observed,
        n_nodes=9,
        max_iter=4,
        tol=1e-12,
        seed=7,
    )

    np.testing.assert_allclose(
        result["a"],
        [1.82408488, 0.001, 0.174714173, 0.997395173],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["b"],
        [0.00217825067, -0.358812835, 1.42022543, 0.000175535508],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["theta"],
        [0.79281557, -0.28868410, 0.23150553, -0.75261783, 0.77473955, -0.75236035],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["loglik_trace"],
        [-14.873979688002828, -14.002066971964293, -13.905641202379064, -13.879977167228605],
        rtol=1e-10,
        atol=1e-10,
    )
    assert result["n_iter"] == 4
    assert result["status"] == "max_iter_reached"
