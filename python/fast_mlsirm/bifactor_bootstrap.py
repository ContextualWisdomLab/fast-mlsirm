# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Joint person bootstrap replication for polytomous bifactor models.

Provides parallelized bootstrap resampling (with optional group stratification),
parameter extraction, convergence tracking, and empirical standard error estimation.
Designed for 390-replicate empirical validation with CPU/GPU execution parity.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import os
import time
import numpy as np

from .polytomous_bifactor import PolytomousBifactorFit, fit_polytomous_bifactor


@dataclass
class BifactorBootstrapResult:
    """Summary and replicate storage of a joint person bootstrap run."""

    n_replicates: int
    n_converged: int
    replicate_slopes: np.ndarray  # (B_conv, n_items, n_dims)
    replicate_thresholds: np.ndarray  # (B_conv, n_items, n_cat - 1)
    replicate_group_means: np.ndarray  # (B_conv, n_groups, n_dims)
    replicate_group_variances: np.ndarray  # (B_conv, n_groups, n_dims)
    replicate_loglik: np.ndarray  # (B_conv,)
    se_slope: np.ndarray  # (n_items, n_dims)
    se_threshold: np.ndarray  # (n_items, n_cat - 1)
    se_group_means: np.ndarray  # (n_groups, n_dims)
    se_group_variances: np.ndarray  # (n_groups, n_dims)
    wall_clock_seconds: float
    throughput_replicates_per_second: float
    device: str = "cpu"


def _generate_bootstrap_indices(
    n_persons: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap sample indices, with stratification if multiple groups."""
    if group_ids is None or n_groups <= 1:
        return rng.integers(0, n_persons, size=n_persons, endpoint=False)

    sample_indices = np.empty(n_persons, dtype=np.int64)
    for g in range(n_groups):
        g_mask = np.flatnonzero(group_ids == g)
        if len(g_mask) == 0:
            continue
        g_sample = rng.choice(g_mask, size=len(g_mask), replace=True)
        sample_indices[g_mask] = g_sample
    return sample_indices


def _fit_single_replicate(
    rep_idx: int,
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    seed: int,
    max_iter: int,
    tol: float,
    ridge: float,
    newton_iter: int,
    qmc_draws: int,
    slope_bound: float | None,
) -> tuple[int, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute one bootstrap resample and fit."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_persons = responses.shape[0]

    indices = _generate_bootstrap_indices(n_persons, group_ids, n_groups, rng)
    y_boot = responses[indices]
    g_boot = group_ids[indices] if group_ids is not None else None

    # Distinct seed for QMC draws to avoid alignment artifacts
    qmc_seed = (seed ^ 0x5DEE_CE66_D) & 0xFFFF_FFFF_FFFF_FFFF

    try:
        fit = fit_polytomous_bifactor(
            responses=y_boot,
            loading_pattern=loading_pattern,
            n_cat=n_cat,
            group_ids=g_boot,
            n_groups=n_groups,
            max_iter=max_iter,
            tol=tol,
            ridge=ridge,
            newton_iter=newton_iter,
            qmc_draws=qmc_draws,
            seed=qmc_seed,
            slope_bound=slope_bound,
            compute_oakes_se=False,  # SEs estimated empirically from bootstrap
        )
        return (
            rep_idx,
            fit.converged,
            fit.slope,
            fit.threshold,
            fit.group_means,
            fit.group_variances,
            fit.loglik,
        )
    except Exception:
        n_items, n_dims = loading_pattern.shape
        return (
            rep_idx,
            False,
            np.full((n_items, n_dims), np.nan),
            np.full((n_items, n_cat - 1), np.nan),
            np.full((n_groups, n_dims), np.nan),
            np.full((n_groups, n_dims), np.nan),
            float("-inf"),
        )


def run_bifactor_bootstrap(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    group_ids: np.ndarray | None = None,
    n_groups: int = 1,
    n_replicates: int = 390,
    n_jobs: int = -1,
    base_seed: int = 42,
    device: str = "cpu",
    slope_bound: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    ridge: float = 1e-6,
    newton_iter: int = 10,
    qmc_draws: int = 2000,
) -> BifactorBootstrapResult:
    """Run joint person bootstrap replications with parallel workers.

    Parameters:
        responses: Persons x items array of response categories.
        loading_pattern: Items x dims array in {0, 1}.
        n_cat: Number of response categories.
        group_ids: Optional 1-D group membership indices.
        n_groups: Number of groups.
        n_replicates: Total bootstrap replicates (e.g. 390).
        n_jobs: Number of parallel workers (-1 for all logical cores).
        base_seed: Master seed for deterministic replication.
        device: 'cpu' or 'gpu' execution device.
        slope_bound: Upper bound on slope magnitude.
        max_iter: Max EM iterations per replicate.
        tol: Convergence tolerance.
        ridge: Ridge stabilization penalty.
        newton_iter: Newton steps per M-step.
        qmc_draws: Halton draws per person.

    Returns:
        BifactorBootstrapResult with replicate matrices, empirical SEs, and timing.
    """
    start_time = time.perf_counter()

    if n_jobs <= 0:
        n_jobs = max(1, os.cpu_count() or 1)

    y_arr = np.asarray(responses, dtype=np.float64)
    lp_arr = np.asarray(loading_pattern, dtype=np.uint8)
    n_items, n_dims = lp_arr.shape
    g_arr = np.asarray(group_ids, dtype=np.int64) if group_ids is not None else None

    # Prepare replicate tasks with deterministic seeds
    tasks = []
    for b in range(n_replicates):
        # 64-bit linear congruential step for uncorrelated replicate seeds
        rep_seed = int((base_seed + b * 0x9E37_79B9_7F4A_7C15) & 0x7FFF_FFFF_FFFF_FFFF)
        tasks.append((
            b,
            y_arr,
            lp_arr,
            n_cat,
            g_arr,
            n_groups,
            rep_seed,
            max_iter,
            tol,
            ridge,
            newton_iter,
            qmc_draws,
            slope_bound,
        ))

    results = [None] * n_replicates

    # ThreadPoolExecutor is safe and GIL-free because fast_mlsirm_core uses py.detach
    if n_jobs == 1 or n_replicates == 1:
        for t in tasks:
            res = _fit_single_replicate(*t)
            results[res[0]] = res
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_fit_single_replicate, *t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                results[res[0]] = res

    # Collect converged replicates
    converged_slopes = []
    converged_thresholds = []
    converged_means = []
    converged_vars = []
    converged_logliks = []

    for r in results:
        if r is not None and r[1]:  # converged
            converged_slopes.append(r[2])
            converged_thresholds.append(r[3])
            converged_means.append(r[4])
            converged_vars.append(r[5])
            converged_logliks.append(r[6])

    n_conv = len(converged_slopes)
    if n_conv > 1:
        slopes_mat = np.stack(converged_slopes, axis=0)
        thresh_mat = np.stack(converged_thresholds, axis=0)
        means_mat = np.stack(converged_means, axis=0)
        vars_mat = np.stack(converged_vars, axis=0)
        loglik_vec = np.array(converged_logliks, dtype=np.float64)

        # Empirical standard errors: ddof=1 sample standard deviation
        se_slope = np.std(slopes_mat, axis=0, ddof=1)
        se_thresh = np.std(thresh_mat, axis=0, ddof=1)
        se_means = np.std(means_mat, axis=0, ddof=1)
        se_vars = np.std(vars_mat, axis=0, ddof=1)
    else:
        slopes_mat = np.empty((0, n_items, n_dims), dtype=np.float64)
        thresh_mat = np.empty((0, n_items, n_cat - 1), dtype=np.float64)
        means_mat = np.empty((0, n_groups, n_dims), dtype=np.float64)
        vars_mat = np.empty((0, n_groups, n_dims), dtype=np.float64)
        loglik_vec = np.empty(0, dtype=np.float64)

        se_slope = np.full((n_items, n_dims), np.nan)
        se_thresh = np.full((n_items, n_cat - 1), np.nan)
        se_means = np.full((n_groups, n_dims), np.nan)
        se_vars = np.full((n_groups, n_dims), np.nan)

    elapsed = time.perf_counter() - start_time
    throughput = n_replicates / elapsed if elapsed > 0 else 0.0

    return BifactorBootstrapResult(
        n_replicates=n_replicates,
        n_converged=n_conv,
        replicate_slopes=slopes_mat,
        replicate_thresholds=thresh_mat,
        replicate_group_means=means_mat,
        replicate_group_variances=vars_mat,
        replicate_loglik=loglik_vec,
        se_slope=se_slope,
        se_threshold=se_thresh,
        se_group_means=se_means,
        se_group_variances=se_vars,
        wall_clock_seconds=elapsed,
        throughput_replicates_per_second=throughput,
        device=device,
    )
