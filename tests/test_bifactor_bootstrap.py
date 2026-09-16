# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for joint person bootstrap replicate parallelization."""

import numpy as np
import pytest

from fast_mlsirm.bifactor_bootstrap import BifactorBootstrapResult, run_bifactor_bootstrap


def test_bifactor_bootstrap_shapes_and_se() -> None:
    """Run a 10-replicate person bootstrap and verify result structure and empirical SEs."""
    n_persons = 80
    n_items = 6
    n_cat = 3
    n_dims = 3

    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1
        loading_pattern[i, 1 + (i // 3)] = 1

    rng = np.random.default_rng(777)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2:] = 1

    boot_res = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=10, batch_size=5, mc_stopping_ratio=0.1, compute_budget_seconds=30.0,
        n_jobs=2,
        base_seed=123,
        max_iter=15,
        tol=1e-3,
        qmc_draws=200,
    )

    assert isinstance(boot_res, BifactorBootstrapResult)
    assert boot_res.n_replicates == 10
    assert boot_res.n_converged >= 8

    # Dimensions: B_conv x items x dims
    assert boot_res.replicate_slopes.ndim == 3
    assert boot_res.replicate_slopes.shape[1:] == (6, 3)
    assert boot_res.replicate_thresholds.shape[1:] == (6, 2)
    assert boot_res.replicate_group_means.shape[1:] == (2, 3)
    assert boot_res.replicate_group_variances.shape[1:] == (2, 3)

    # Standard error shapes
    assert boot_res.se_slope.shape == (6, 3)
    assert boot_res.se_threshold.shape == (6, 2)
    assert boot_res.se_group_means.shape == (2, 3)
    assert boot_res.se_group_variances.shape == (2, 3)

    # Reference group moments have zero SE (fixed across all replicates)
    np.testing.assert_allclose(boot_res.se_group_means[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(boot_res.se_group_variances[0], 0.0, atol=1e-10)

    # Free parameters must have finite non-negative SEs
    assert np.all(np.isfinite(boot_res.se_slope))
    assert np.all(boot_res.se_slope >= 0.0)
    assert np.all(np.isfinite(boot_res.se_threshold))
    assert np.all(boot_res.se_threshold >= 0.0)
    assert boot_res.wall_clock_seconds > 0.0
    assert boot_res.throughput_replicates_per_second > 0.0


def test_bifactor_bootstrap_reproducibility() -> None:
    """Identical seeds must yield bit-exact bootstrap replicate estimates within 1e-6."""
    n_persons = 40
    n_items = 4
    n_cat = 3
    n_dims = 2

    loading_pattern = np.ones((n_items, n_dims), dtype=np.uint8)
    rng = np.random.default_rng(999)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)

    res_run1 = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        n_replicates=4, batch_size=4, mc_stopping_ratio=0.001, compute_budget_seconds=30.0,
        n_jobs=1,
        base_seed=42,
        max_iter=10,
        tol=1e-3,
        qmc_draws=100,
    )

    res_run2 = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        n_replicates=4, batch_size=4, mc_stopping_ratio=0.001, compute_budget_seconds=30.0,
        n_jobs=2,  # Multi-threaded execution
        base_seed=42,
        max_iter=10,
        tol=1e-3,
        qmc_draws=100,
    )

    assert res_run1.n_converged == res_run2.n_converged
    np.testing.assert_allclose(
        res_run1.replicate_slopes,
        res_run2.replicate_slopes,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        res_run1.replicate_thresholds,
        res_run2.replicate_thresholds,
        atol=1e-6,
    )
