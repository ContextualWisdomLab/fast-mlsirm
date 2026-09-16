# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stopping rule, per-replicate reporting, validation, and CPU/GPU parity
for the joint person bootstrap.

Implementation basis: Andrews, D. W. K., & Buchinsky, M. (2000). A
three-step method for choosing the number of bootstrap repetitions.
*Econometrica, 68*(1), 23–51. https://www.jstor.org/stable/2999474
"""

import numpy as np
import pytest

from fast_mlsirm.bifactor_bootstrap import (
    _endpoint_movement,
    run_bifactor_bootstrap,
)


def _small_problem(seed=777, n_persons=80, n_items=6, n_cat=3, n_dims=3):
    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1
        loading_pattern[i, 1 + (i % (n_dims - 1))] = 1
    rng = np.random.default_rng(seed)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1
    return responses, loading_pattern, n_cat, group_ids


def test_endpoint_movement_exact_values():
    """_endpoint_movement measures max endpoint shift over half-width."""
    prev_lo = np.array([0.0, 1.0, 5.0])
    prev_hi = np.array([2.0, 3.0, 5.0])  # third entry has zero half-width
    new_lo = np.array([0.1, 1.5, 5.0])
    new_hi = np.array([2.1, 3.0, 5.0])
    # entry 0: max(0.1, 0.1)/1.05 = 0.0952; entry 1: max(0.5, 0.0)/0.75 = 2/3
    assert _endpoint_movement(prev_lo, prev_hi, new_lo, new_hi) == pytest.approx(2.0 / 3.0)
    # Identical endpoints -> zero movement.
    assert _endpoint_movement(new_lo, new_hi, new_lo, new_hi) == pytest.approx(0.0)
    # All-zero half-widths carry no MC uncertainty.
    z = np.array([1.0, 1.0])
    assert _endpoint_movement(z, z, z, z) == pytest.approx(0.0)


def test_ratio_zero_runs_all_replicates():
    """mc_stopping_ratio=0 disables early stopping (full completion)."""
    responses, loading_pattern, n_cat, group_ids = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=12,
        batch_size=4,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=300.0,
        n_jobs=1,
        base_seed=123,
        max_iter=8,
        tol=1e-3,
        qmc_draws=200,
    )
    assert res.n_requested == 12
    assert res.n_replicates == 12
    assert res.stopped_early is False
    assert len(res.converged) == 12
    assert int(np.sum(res.converged)) == res.n_converged
    assert res.replicate_slopes.shape[0] == res.n_converged


def test_per_replicate_convergence_reported_and_never_substituted():
    """Failed replicates are flagged, excluded, never imputed."""
    responses, loading_pattern, n_cat, group_ids = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=12,
        batch_size=4,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=300.0,
        n_jobs=2,
        base_seed=123,
        max_iter=8,
        tol=1e-3,
        qmc_draws=200,
    )
    assert res.converged.dtype == bool
    assert res.replicate_loglik.shape == (res.n_converged,)
    assert np.all(np.isfinite(res.replicate_loglik))
    # Summary statistics come only from converged replicates.
    assert res.replicate_slopes.shape[0] == res.n_converged
    assert res.replicate_thresholds.shape[0] == res.n_converged
    if res.n_converged > 1:
        assert np.all(np.isfinite(res.se_slope))
        assert np.all(np.isfinite(res.ci_lower_slope))
        assert np.all(res.ci_lower_slope <= res.ci_upper_slope)


def test_stopping_rule_invariants():
    """Early stopping (if fired) preserves reporting invariants."""
    responses, loading_pattern, n_cat, group_ids = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=40,
        batch_size=5,
        mc_stopping_ratio=0.5,
        compute_budget_seconds=600.0,
        n_jobs=2,
        base_seed=123,
        max_iter=8,
        tol=1e-3,
        qmc_draws=200,
    )
    assert res.n_replicates <= res.n_requested
    assert len(res.converged) == res.n_replicates
    if res.stopped_early:
        assert res.n_replicates < res.n_requested


def test_bootstrap_argument_validation():
    """Caller arguments have validated ranges (no silent misconfiguration)."""
    responses, loading_pattern, n_cat, group_ids = _small_problem()
    good = dict(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=4,
        batch_size=4,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=30.0,
        n_jobs=1,
        base_seed=1,
        max_iter=3,
        tol=1e-3,
        qmc_draws=200,
    )
    bad_cases = [
        {"n_replicates": 0},
        {"n_replicates": -3},
        {"n_replicates": 4.5},
        {"n_replicates": True},
        {"batch_size": 0},
        {"batch_size": -1},
        {"mc_stopping_ratio": -0.1},
        {"mc_stopping_ratio": 1.0},
        {"mc_stopping_ratio": float("nan")},
        {"mc_stopping_ratio": float("inf")},
        {"compute_budget_seconds": 0.0},
        {"compute_budget_seconds": -2.0},
        {"compute_budget_seconds": float("nan")},
        {"device": "tpu"},
        {"ci_level": 0.0},
        {"ci_level": 1.0},
        {"ci_level": 1.5},
        {"qmc_draws": 0},
    ]
    for bad in bad_cases:
        kwargs = dict(good)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            run_bifactor_bootstrap(**kwargs)


def test_cpu_gpu_bootstrap_replicate_parity():
    """Same base_seed gives replicate-by-replicate CPU/GPU agreement."""
    responses, loading_pattern, n_cat, _ = _small_problem(
        seed=999, n_persons=40, n_items=4, n_dims=2
    )
    kwargs = dict(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        n_replicates=6,
        batch_size=6,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=300.0,
        n_jobs=1,
        base_seed=42,
        max_iter=8,
        tol=1e-3,
        qmc_draws=241,
    )
    res_cpu = run_bifactor_bootstrap(**kwargs, device="cpu")
    res_gpu = run_bifactor_bootstrap(**kwargs, device="gpu")
    assert res_cpu.n_converged == res_gpu.n_converged
    assert res_cpu.n_replicates == res_gpu.n_replicates
    np.testing.assert_allclose(
        res_cpu.replicate_slopes, res_gpu.replicate_slopes, atol=1e-3
    )
    np.testing.assert_allclose(
        res_cpu.replicate_thresholds, res_gpu.replicate_thresholds, atol=1e-3
    )
    np.testing.assert_allclose(
        res_cpu.replicate_loglik, res_gpu.replicate_loglik, atol=1e-2
    )
