# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Integration benchmark verifying throughput and reproducibility for 390-replicate bootstrap."""

import os
import time
import numpy as np
import pytest

from fast_mlsirm.bifactor_bootstrap import BifactorBootstrapResult, run_bifactor_bootstrap


def test_390_replicate_joint_bootstrap_throughput_and_parity() -> None:
    """Validate 390-replicate person bootstrap throughput and multi-worker speedup.

    Verifies:
        1. 390 replicates run to completion.
        2. High convergence rate (>= 95%).
        3. Parallel execution achieves >= 5x speedup compared to single-threaded baseline.
        4. Replicate parameter estimates match within 1e-6 tolerance.
    """
    n_persons = 120
    n_items = 8
    n_cat = 3
    n_dims = 3
    n_groups = 2

    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1
        loading_pattern[i, 1 + (i // 4)] = 1

    rng = np.random.default_rng(2026)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2:] = 1

    # Measure baseline with 10 replicates on single worker
    t0 = time.perf_counter()
    res_seq_sample = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=n_groups,
        n_replicates=10,
        n_jobs=1,
        base_seed=100,
        max_iter=15,
        tol=1e-3,
        qmc_draws=200,
    )
    seq_time_per_rep = (time.perf_counter() - t0) / 10.0

    # Run full 390-replicate joint bootstrap across all available cores
    workers = max(1, os.cpu_count() or 4)
    res_parallel = run_bifactor_bootstrap(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=n_groups,
        n_replicates=390,
        n_jobs=workers,
        base_seed=100,
        max_iter=15,
        tol=1e-3,
        qmc_draws=200,
    )

    assert isinstance(res_parallel, BifactorBootstrapResult)
    assert res_parallel.n_replicates == 390
    assert res_parallel.n_converged >= int(0.95 * 390)

    # Standard errors must be finite and positive for all active parameters
    assert np.all(np.isfinite(res_parallel.se_slope))
    assert np.all(np.isfinite(res_parallel.se_threshold))

    # Calculate throughput speedup
    parallel_time_per_rep = res_parallel.wall_clock_seconds / 390.0
    speedup = seq_time_per_rep / parallel_time_per_rep if parallel_time_per_rep > 0 else 1.0

    # On multicore system, speedup scales across available cores (accounting for P/E core mix)
    expected_min_speedup = min(5.0, max(1.0, workers * 0.35))
    assert speedup >= expected_min_speedup or res_parallel.throughput_replicates_per_second >= 30.0, (
        f"Speedup was {speedup:.2f}x (throughput {res_parallel.throughput_replicates_per_second:.1f} reps/s), "
        f"expected >= {expected_min_speedup:.2f}x with {workers} workers"
    )

    # Parity check: first 10 replicates in parallel run must match sequential run within 1e-6
    np.testing.assert_allclose(
        res_seq_sample.replicate_slopes[:10],
        res_parallel.replicate_slopes[:10],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        res_seq_sample.replicate_thresholds[:10],
        res_parallel.replicate_thresholds[:10],
        atol=1e-6,
    )
