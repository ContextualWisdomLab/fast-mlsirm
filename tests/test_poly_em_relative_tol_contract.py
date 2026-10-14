"""Relative EM stopping contract of the polytomous EM fits (#2121).

The Rust core (``crates/mlsirm-core/src/poly.rs``, ``checked_em_delta``) stops
when ``final_delta = loglik_t - loglik_{t-1}`` is at most
``stopping_tolerance = tol * (1 + |loglik_{t-1}|)``, so ``tol`` is a relative
log-likelihood criterion, and it rejects any decrease larger than
``32 * eps * (1 + |loglik_{t-1}|)`` as non-monotone. These tests pin that
contract on every Python polytomous EM entry point and require each docstring
to state it, so ``final_delta`` is never compared with the raw ``tol``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fast_mlsirm.polytomous import (
    fit_nominal_polytomous,
    fit_poly_fipc,
    fit_polytomous,
)

N_ITEMS = 6
N_CAT = 3
TOL = 1e-6
SLOPES = np.array([1.4, 1.1, 0.9, 1.2, 1.0, 0.8])
INTERCEPTS = np.array(
    [[1.2, -0.2], [1.0, 0.1], [1.3, -0.4], [0.9, 0.0], [1.1, -0.3], [1.4, -0.1]]
)


def _simulate(seed: int, n_persons: int, mean: float, sd: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean, sd, n_persons)
    y = np.zeros((n_persons, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        p1 = 1.0 / (1.0 + np.exp(-(SLOPES[i] * theta + INTERCEPTS[i, 0])))
        p2 = 1.0 / (1.0 + np.exp(-(SLOPES[i] * theta + INTERCEPTS[i, 1])))
        u = rng.uniform(0.0, 1.0, n_persons)
        y[:, i] = (u < p1).astype(np.int64) + (u < p2).astype(np.int64)
    return y


def _fit_grm():
    return fit_polytomous(
        _simulate(11, 500, 0.0, 1.0),
        N_CAT,
        model="grm",
        q_theta=31,
        max_iter=500,
        tol=TOL,
    )


def _fit_gpcm():
    return fit_polytomous(
        _simulate(12, 500, 0.0, 1.0),
        N_CAT,
        model="gpcm",
        q_theta=31,
        max_iter=500,
        tol=TOL,
    )


def _fit_nominal():
    return fit_nominal_polytomous(
        _simulate(13, 500, 0.0, 1.0), N_CAT, q_theta=31, max_iter=500, tol=TOL
    )


def _fit_fipc():
    reference = _fit_grm()
    assert reference.converged, reference.termination_reason
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:4] = True
    return fit_poly_fipc(
        _simulate(14, 400, 0.4, 1.1),
        N_CAT,
        anchor,
        np.asarray(reference.slope),
        np.asarray(reference.cat_params),
        q_theta=31,
        max_iter=500,
        tol=TOL,
    )


FITS = {
    "fit_polytomous_grm": _fit_grm,
    "fit_polytomous_gpcm": _fit_gpcm,
    "fit_nominal_polytomous": _fit_nominal,
    "fit_poly_fipc": _fit_fipc,
}


@pytest.mark.parametrize("name", sorted(FITS))
def test_converged_fit_reports_relative_stopping_tolerance(name: str) -> None:
    fit = FITS[name]()

    assert fit.converged, fit.termination_reason
    trace = np.asarray(fit.loglik_trace, dtype=np.float64)
    assert trace.size >= 2
    previous = float(trace[-2])
    # The Rust core computes exactly ``tol * (1.0 + previous.abs())`` in
    # binary64, so the reported value is bit-identical to this expression.
    assert fit.stopping_tolerance == TOL * (1.0 + abs(previous))
    assert fit.final_delta == float(trace[-1]) - previous
    assert fit.final_delta <= fit.stopping_tolerance
    # Monotone guard: no recorded EM step may decrease the likelihood by more
    # than the core's 32 * eps relative allowance.
    steps = np.diff(trace)
    allowance = 32.0 * np.finfo(np.float64).eps * (1.0 + np.abs(trace[:-1]))
    assert np.all(steps >= -allowance)
    # The criterion is relative: with |loglik| in the thousands the accepted
    # change exceeds the raw ``tol`` by orders of magnitude.
    assert fit.stopping_tolerance > 100.0 * TOL


@pytest.mark.parametrize(
    "function", [fit_polytomous, fit_nominal_polytomous, fit_poly_fipc]
)
def test_docstring_states_relative_stopping_contract(function) -> None:
    doc = inspect.getdoc(function) or ""
    assert "tol * (1 + |loglik_{t-1}|)" in doc
    assert "stopping_tolerance" in doc
    assert "non-monotone" in doc
    assert "tighter ``tol``" in doc
