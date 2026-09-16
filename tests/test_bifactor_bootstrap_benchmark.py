# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Measured CPU vs GPU wall time for the joint person bootstrap.

No study-specific replicate target is encoded here: the replicate count is a
caller-style setting (overridable via ``STAGE5_BOOTSTRAP_REPS``) and this
benchmark records measured wall times rather than asserting
machine-specific throughput thresholds.

Quadrature is precision: the benchmark uses 241 Halton draws (above the
121-draw floor for study settings) with no upper cap.

Implementation basis: Andrews, D. W. K., & Buchinsky, M. (2000). A
three-step method for choosing the number of bootstrap repetitions.
*Econometrica, 68*(1), 23–51. https://www.jstor.org/stable/2999474
"""

import os
import time

import numpy as np

from fast_mlsirm.bifactor_bootstrap import BifactorBootstrapResult, run_bifactor_bootstrap


def _problem():
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
    group_ids[n_persons // 2 :] = 1
    return responses, loading_pattern, n_cat, group_ids, n_groups


def test_joint_bootstrap_cpu_vs_gpu_wall_time_and_parity() -> None:
    """Measure CPU vs GPU bootstrap wall time; assert completion and parity.

    Verifies:
        1. The requested replicates run to completion on both devices.
        2. Same-seed CPU and GPU runs agree replicate-by-replicate.
        3. Measured (not estimated) wall times are reported for the PR record.
    """
    responses, loading_pattern, n_cat, group_ids, n_groups = _problem()
    n_replicates = int(os.environ.get("STAGE5_BOOTSTRAP_REPS", "8"))
    workers = max(1, os.cpu_count() or 4)

    common = dict(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=n_groups,
        n_replicates=n_replicates,
        batch_size=n_replicates,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=1200.0,
        base_seed=100,
        max_iter=8,
        tol=1e-3,
        qmc_draws=241,
    )

    t0 = time.perf_counter()
    res_cpu = run_bifactor_bootstrap(**common, device="cpu", n_jobs=workers)
    cpu_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    res_gpu = run_bifactor_bootstrap(**common, device="gpu", n_jobs=workers)
    gpu_time = time.perf_counter() - t1

    for res in (res_cpu, res_gpu):
        assert isinstance(res, BifactorBootstrapResult)
        assert res.n_replicates == n_replicates
        assert res.n_converged >= 1
        assert np.all(np.isfinite(res.replicate_loglik))

    # Replicate-by-replicate device parity (single-precision E-step level).
    assert res_cpu.n_converged == res_gpu.n_converged
    np.testing.assert_allclose(
        res_cpu.replicate_slopes, res_gpu.replicate_slopes, atol=1e-3
    )
    np.testing.assert_allclose(
        res_cpu.replicate_thresholds, res_gpu.replicate_thresholds, atol=1e-3
    )

    print(
        f"\n[bootstrap B={n_replicates} qn=241 workers={workers}] "
        f"CPU wall {cpu_time:.3f}s ({cpu_time / n_replicates:.3f}s/rep) vs "
        f"GPU wall {gpu_time:.3f}s ({gpu_time / n_replicates:.3f}s/rep); "
        f"speedup {cpu_time / gpu_time:.2f}x"
    )
