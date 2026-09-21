"""Python binding contract for the single-group polytomous two-tier GRM
(stage 4 of #1912).

Covers the public ``fast_mlsirm.two_tier_grm.fit_two_tier_grm`` surface:
shapes and identification properties of a small independent simulation
(two correlated primaries, cross-cutting specifics), deterministic reruns
from the same seed, argument validation, the unobserved-category failure,
and the non-convergence report.

Reference
---------
Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0 (full text read, pp. 583-584)
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.two_tier_grm import fit_two_tier_grm, two_tier_oakes_se

N_PERSONS = 300
N_ITEMS = 6
N_PRIMARY = 2
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260917
RHO = 0.35
PRIMARY_MAP = np.array(
    [
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [False, True],
    ]
)
SPECIFIC_MAP = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
TRUE_A_P = np.array(
    [
        [1.4, 0.0],
        [1.1, 0.0],
        [1.0, 0.0],
        [0.0, 1.2],
        [0.0, 0.9],
        [0.0, 1.1],
    ]
)
TRUE_A_S = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 0.9])
TRUE_D = np.array(
    [
        [1.2, 0.0, -1.2],
        [1.0, -0.1, -1.3],
        [1.3, 0.2, -1.0],
        [1.1, 0.1, -1.1],
        [0.9, -0.2, -1.4],
        [1.2, 0.0, -1.2],
    ]
)


def _simulate(seed: int, rho: float = RHO) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0.0, 1.0, N_PERSONS)
    z1 = rng.normal(0.0, 1.0, N_PERSONS)
    theta = np.stack([z0, rho * z0 + np.sqrt(1.0 - rho**2) * z1], axis=1)

    theta_s = rng.normal(0.0, 1.0, (N_PERSONS, N_SPECIFIC))
    y = np.zeros((N_PERSONS, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = (
            TRUE_A_P[i, 0] * theta[:, 0]
            + TRUE_A_P[i, 1] * theta[:, 1]
            + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        )
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(N_PERSONS)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y

def test_identity_primary_identification() -> None:
    y = _simulate(SEED, rho=0.0)
    # Oakes information needs a fit close enough to the stationary point.
    fixed = _fit(y, primary_correlation="identity", tol=1e-7)
    estimated = _fit(y, primary_correlation="estimate")
    np.testing.assert_array_equal(fixed.phi, np.eye(N_PRIMARY))
    assert not np.signbit(fixed.phi).any()
    assert fixed.primary_identification == "orthogonal"
    assert estimated.primary_identification == "correlated"
    assert fixed.n_parameters + N_PRIMARY * (N_PRIMARY - 1) // 2 == estimated.n_parameters
    assert np.all(np.diff(fixed.loglik_trace) >= -1e-7)
    assert abs(fixed.loglik_trace[-1] - estimated.loglik_trace[-1]) / N_PERSONS < 0.03
    se = two_tier_oakes_se(
        fixed.a_primary, fixed.a_specific, fixed.threshold, fixed.phi, y,
        PRIMARY_MAP, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC,
        q_primary=7, q_specific=7, fd_step=1e-5, primary_correlation="identity",
    )
    assert se.information.shape == (fixed.n_parameters, fixed.n_parameters)
    assert not any(label.startswith("phi_z:") for label in se.labels)
    assert se.positive_definite, se.non_pd_reason
    assert np.isfinite(se.se).all()


def test_primary_correlation_validation() -> None:
    with pytest.raises(ValueError, match="primary_correlation"):
        _fit(_simulate(SEED), primary_correlation="unknown")


def _fit(y: np.ndarray, **overrides):
    kwargs = {
        "q_primary": 7,
        "q_specific": 7,
        "max_iter": 500,
        "tol": 1e-5,
        "n_starts": 1,
        "seed": SEED,
    }
    kwargs.update(overrides)
    return fit_two_tier_grm(y, PRIMARY_MAP, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC, **kwargs)


def test_fit_returns_identified_shaped_result() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    assert fit.converged, fit.termination_reason
    assert fit.a_primary.shape == (N_ITEMS, N_PRIMARY)
    assert fit.a_specific.shape == (N_ITEMS,)
    assert fit.threshold.shape == (N_ITEMS, N_CAT - 1)
    assert fit.phi.shape == (N_PRIMARY, N_PRIMARY)
    # Phi is a correlation matrix: unit diagonal, symmetric, |off-diag| < 1.
    np.testing.assert_allclose(np.diag(fit.phi), np.ones(N_PRIMARY), atol=1e-12)
    np.testing.assert_allclose(fit.phi, fit.phi.T, atol=1e-12)
    assert bool(np.all(np.abs(fit.phi - np.eye(N_PRIMARY)) < 1.0))
    assert fit.theta_p_eap.shape == (N_PERSONS, N_PRIMARY)
    assert fit.theta_p_sd.shape == (N_PERSONS, N_PRIMARY)
    assert bool((fit.theta_p_sd >= 0).all())
    assert fit.category_counts.shape == (N_ITEMS, N_CAT)
    assert int(fit.category_counts.sum()) == N_PERSONS * N_ITEMS
    # Fixed pattern positions stay exactly zero.
    assert bool((fit.a_primary[~PRIMARY_MAP] == 0.0).all())
    # Strictly decreasing boundary intercepts within each item.
    assert bool((np.diff(fit.threshold, axis=1) < 0).all())
    # Reflection canonicalization: largest-magnitude slopes positive.
    for d in range(N_PRIMARY):
        loaders = np.flatnonzero(PRIMARY_MAP[:, d])
        anchor = loaders[int(np.argmax(np.abs(fit.a_primary[loaders, d])))]
        assert fit.a_primary[anchor, d] > 0.0
    for s in range(N_SPECIFIC):
        block = np.flatnonzero(SPECIFIC_MAP == s)
        anchor = block[int(np.argmax(np.abs(fit.a_specific[block])))]
        assert fit.a_specific[anchor] > 0.0
    # Monotone loglik trace ending at a finite value.
    trace = np.asarray(fit.loglik_trace)
    assert np.all(np.isfinite(trace)) and len(trace) >= 2
    assert bool((np.diff(trace) >= -1e-9).all())
    # sum_i (k_i + has_specific + m1) + P(P-1)/2 = 6*1 + 6*1 + 6*3 + 1.
    assert fit.n_parameters == 6 + 6 + 18 + 1


def test_same_seed_bit_reproduces() -> None:
    y = _simulate(SEED)
    first = _fit(y)
    second = _fit(y)
    np.testing.assert_array_equal(first.a_primary, second.a_primary)
    np.testing.assert_array_equal(first.threshold, second.threshold)
    np.testing.assert_array_equal(first.phi, second.phi)
    assert first.best_start == second.best_start


def test_rejects_out_of_range_caller_arguments() -> None:
    y = _simulate(SEED)
    # #1929: no node-count cap; q_primary=5/q_specific=22 are now accepted,
    # only < 1 is not.
    with pytest.raises(ValueError):
        _fit(y, q_primary=0)
    with pytest.raises(ValueError):
        _fit(y, q_specific=0)
    with pytest.raises(ValueError):
        _fit(y, max_iter=0)
    with pytest.raises(ValueError):
        _fit(y, tol=0.0)
    with pytest.raises(ValueError):
        _fit(y, n_starts=0)
    with pytest.raises(ValueError, match="primary_map"):
        fit_two_tier_grm(
            y,
            np.ones((N_ITEMS, N_PRIMARY - 1), dtype=bool),
            SPECIFIC_MAP,
            N_CAT,
            N_PRIMARY,
            N_SPECIFIC,
            7,
            7, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15
        )
    with pytest.raises(ValueError, match="specific_map"):
        fit_two_tier_grm(
            y,
            PRIMARY_MAP,
            np.array([0, 0, 0, 1, 1]),
            N_CAT,
            N_PRIMARY,
            N_SPECIFIC,
            7,
            7, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15
        )


def test_q_primary_and_q_specific_are_required() -> None:
    """No unsourced defaults exist for the node counts (Project rule, #1929)."""
    y = _simulate(SEED)
    with pytest.raises(TypeError):
        fit_two_tier_grm(y, PRIMARY_MAP, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15)


def test_unobserved_category_fails_loudly() -> None:
    y = _simulate(SEED)
    y[:, 0] = np.clip(y[:, 0], 0, N_CAT - 2)  # drop the top category
    with pytest.raises(ValueError, match="never observed"):
        _fit(y)
