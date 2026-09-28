# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Measured CPU vs GPU wall time for the joint person bootstrap.

No study-specific replicate target is encoded here: the replicate count is a
caller-style setting (overridable via ``STAGE5_BOOTSTRAP_REPS``) and this
benchmark records measured wall times rather than asserting
machine-specific throughput thresholds.

Quadrature follows the merged estimator contract (arbitrary-``n``
Gauss-Hermite rules generated on demand via Golub & Welsch, 1969, issue
#1929; no defaults); the smoke benchmark uses a small grid while the
study-precision benchmark (``test_joint_bootstrap_cpu_vs_gpu_wall_time_q121``)
runs at the maintainer-standard 121-point grid.

The 1e-6 absolute replicate bound is a regression check for this fixed
synthetic fixture. It detects the 4.04e-6 drift recorded at PR #2246 head
050351ed; it is not a uniform error bound for all response matrices.
NumPy 2.5 ``assert_allclose`` treats matching NaNs as equal by default, so
replicate parameter arrays must first be finite:
https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose .

The endpoint-movement early stop is a heuristic, not an Andrews–Buchinsky
``(pdb, τ)`` accuracy rule.
"""

import os
import time

import numpy as np
import pytest

from fast_mlsirm.bifactor_bootstrap import BifactorBootstrapResult, run_bifactor_bootstrap

from bifactor_gpu_evidence import missing_hardware_bootstrap


def _problem():
    n_persons = 120
    n_items = 8
    n_cat = 3
    n_specific = 2

    smap = np.zeros(n_items, dtype=np.int64)
    smap[n_items // 2 :] = 1

    rng = np.random.default_rng(2026)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1
    return responses, smap, n_cat, n_specific, group_ids


def test_joint_bootstrap_cpu_vs_gpu_wall_time_and_parity() -> None:
    """Measure CPU vs GPU bootstrap wall time; assert completion and parity.

    Verifies:
        1. The requested replicates run to completion on both devices.
        2. Same-seed CPU and GPU runs agree replicate-by-replicate.
        3. Measured (not estimated) wall times are reported for the PR record.
    """
    responses, smap, n_cat, n_specific, group_ids = _problem()
    n_replicates = int(os.environ.get("STAGE5_BOOTSTRAP_REPS", "6"))
    workers = max(1, os.cpu_count() or 4)

    common = dict(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=n_replicates,
        batch_size=n_replicates,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=3600.0,
        q_general=11,
        q_specific=11,
        base_seed=100,
        ci_level=0.95,
        max_iter=10,
        tol=1e-3,
        n_starts=1,
    )

    t0 = time.perf_counter()
    res_cpu = run_bifactor_bootstrap(**common, device="cpu", n_jobs=workers)
    cpu_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        res_gpu = run_bifactor_bootstrap(**common, device="gpu", n_jobs=workers)
    except RuntimeError as exc:
        if missing_hardware_bootstrap(exc):
            return
        raise
    gpu_time = time.perf_counter() - t1

    for res in (res_cpu, res_gpu):
        assert isinstance(res, BifactorBootstrapResult)
        assert res.n_replicates == n_replicates
        assert res.n_converged >= 1
        assert len(res.replicate_ids) == len(res.replicate_errors) == n_replicates
        assert np.all(np.isfinite(res.replicate_loglik))
        for estimates in (
            res.replicate_a_general, res.replicate_a_specific, res.replicate_threshold
        ):
            assert np.all(np.isfinite(estimates))
        print(
            f"[bootstrap q=11] {res.device}: converged={res.n_converged}/{n_replicates}; "
            f"ids={res.converged_replicate_ids}; errors={res.replicate_errors}"
        )

    # Replicate-by-replicate device parity (single-precision E-step level).
    assert res_cpu.n_converged == res_gpu.n_converged
    assert res_cpu.converged_replicate_ids == res_gpu.converged_replicate_ids
    np.testing.assert_allclose(
        res_cpu.replicate_a_general, res_gpu.replicate_a_general, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        res_cpu.replicate_a_specific, res_gpu.replicate_a_specific, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        res_cpu.replicate_threshold, res_gpu.replicate_threshold, atol=1e-6, rtol=0
    )

    print(
        f"\n[bootstrap B={n_replicates} q=11 workers={workers}] "
        f"CPU wall {cpu_time:.3f}s ({cpu_time / n_replicates:.3f}s/rep) vs "
        f"GPU wall {gpu_time:.3f}s ({gpu_time / n_replicates:.3f}s/rep); "
        f"speedup {cpu_time / gpu_time:.2f}x"
    )


@pytest.mark.skipif(
    os.environ.get("STAGE5_HIGH_Q") != "1",
    reason="study-precision grid (121 nodes); rerun with STAGE5_HIGH_Q=1",
)
def test_joint_bootstrap_cpu_vs_gpu_wall_time_q121() -> None:
    """Measure CPU vs GPU bootstrap wall time at the 121-point study grid.

    Same completion/parity contract as the smoke benchmark, at the
    maintainer-standard minimum-precision node count (issue #1929; Golub &
    Welsch, 1969 arbitrary-``n`` rules). Gated behind ``STAGE5_HIGH_Q=1``
    like the high-``q`` parity tests: each replicate fits a 14,641-node
    grid, so the replicate count defaults to 2 (overridable via
    ``STAGE5_HIGH_Q_BOOTSTRAP_REPS``) and wall times are recorded for the
    PR record rather than asserted against machine-specific thresholds.
    """
    n_persons = 48
    n_items = 6
    n_cat = 3
    n_specific = 2

    smap = np.zeros(n_items, dtype=np.int64)
    smap[n_items // 2 :] = 1

    rng = np.random.default_rng(2026)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1

    n_replicates = int(os.environ.get("STAGE5_HIGH_Q_BOOTSTRAP_REPS", "2"))
    workers = max(1, os.cpu_count() or 4)

    common = dict(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=n_replicates,
        batch_size=n_replicates,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=3600.0,
        q_general=121,
        q_specific=121,
        base_seed=100,
        ci_level=0.95,
        max_iter=15,
        tol=1e-3,
        n_starts=1,
    )

    t0 = time.perf_counter()
    res_cpu = run_bifactor_bootstrap(**common, device="cpu", n_jobs=workers)
    cpu_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        res_gpu = run_bifactor_bootstrap(**common, device="gpu", n_jobs=workers)
    except RuntimeError as exc:
        if missing_hardware_bootstrap(exc):
            return
        raise
    gpu_time = time.perf_counter() - t1

    for res in (res_cpu, res_gpu):
        assert isinstance(res, BifactorBootstrapResult)
        assert res.n_replicates == n_replicates
        assert res.n_converged >= 1
        assert len(res.replicate_ids) == len(res.replicate_errors) == n_replicates
        assert np.all(np.isfinite(res.replicate_loglik))
        for estimates in (
            res.replicate_a_general, res.replicate_a_specific, res.replicate_threshold
        ):
            assert np.all(np.isfinite(estimates))
        print(
            f"[bootstrap q=121] {res.device}: converged={res.n_converged}/{n_replicates}; "
            f"ids={res.converged_replicate_ids}; errors={res.replicate_errors}"
        )

    # Replicate-by-replicate device parity (single-precision E-step level).
    assert res_cpu.n_converged == res_gpu.n_converged
    assert res_cpu.converged_replicate_ids == res_gpu.converged_replicate_ids
    np.testing.assert_allclose(
        res_cpu.replicate_a_general, res_gpu.replicate_a_general, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        res_cpu.replicate_a_specific, res_gpu.replicate_a_specific, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        res_cpu.replicate_threshold, res_gpu.replicate_threshold, atol=1e-6, rtol=0
    )

    print(
        f"\n[bootstrap B={n_replicates} q=121 workers={workers}] "
        f"CPU wall {cpu_time:.3f}s ({cpu_time / n_replicates:.3f}s/rep) vs "
        f"GPU wall {gpu_time:.3f}s ({gpu_time / n_replicates:.3f}s/rep); "
        f"speedup {cpu_time / gpu_time:.2f}x"
    )
