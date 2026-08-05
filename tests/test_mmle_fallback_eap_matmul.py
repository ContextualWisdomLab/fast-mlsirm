"""Numerical and implementation contracts for fallback MMLE EAP projection."""

from __future__ import annotations

import inspect

import numpy as np

from fast_mlsirm.estimators.mmle import fit_mmle_2pl, gauss_hermite_nodes


def test_one_iteration_eap_matches_independent_weighted_sum_reference() -> None:
    """Dense matrix-vector projection must preserve the original EAP equation."""
    y = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, True, True],
            [True, False, True],
            [True, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )
    n_nodes = 9
    seed = 7

    y_filled = np.where(observed, y, 0.0)
    obs_f = observed.astype(np.float64)
    nodes, weights = gauss_hermite_nodes(n_nodes)
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    rng = np.random.default_rng(seed)
    discrimination = np.ones(y.shape[1]) + 0.01 * rng.standard_normal(y.shape[1])
    intercept = np.log(p_item / (1.0 - p_item))
    logit = nodes[:, None] * discrimination[None, :] + intercept[None, :]
    log_p1 = -np.logaddexp(0.0, -logit)
    log_p0 = -np.logaddexp(0.0, logit)
    log_joint = (
        (y_filled * obs_f) @ log_p1.T
        + ((1.0 - y_filled) * obs_f) @ log_p0.T
        + np.log(weights)[None, :]
    )
    maximum = log_joint.max(axis=1, keepdims=True)
    stabilized = np.exp(log_joint - maximum)
    posterior = stabilized / stabilized.sum(axis=1, keepdims=True)
    expected_theta = (posterior * nodes[None, :]).sum(axis=1)

    result = fit_mmle_2pl(
        y,
        observed,
        n_nodes=n_nodes,
        max_iter=1,
        seed=seed,
    )

    np.testing.assert_allclose(
        np.asarray(result["theta"], dtype=np.float64),
        expected_theta,
        rtol=1e-13,
        atol=1e-13,
    )


def test_eap_projection_retains_the_allocation_bounded_matmul_path() -> None:
    """The fallback must not restore the posterior-by-node temporary array."""
    source = inspect.getsource(fit_mmle_2pl)

    assert "theta = posterior @ nodes" in source
    assert "posterior * nodes[None, :]" not in source
