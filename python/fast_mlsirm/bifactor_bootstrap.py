# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Joint person bootstrap for polytomous bifactor models.

Resamples persons (stratified by group when multiple groups are present),
refits the polytomous bifactor graded response model on each replicate, and
reports empirical standard errors and percentile intervals from the converged
replicates. Replicate count, batch size, Monte Carlo stopping ratio, and
compute budget are caller arguments; this module defines no study-specific
defaults for them.

Implementation basis
--------------------
The replicate-number stopping rule is a sequential application of the accuracy
framework of Andrews and Buchinsky (2000, §§ 2–4): a finite-``B`` bootstrap
quantity is accurate when its percentage deviation from the ideal (``B`` → ∞)
bootstrap quantity is small. The caller-supplied ``mc_stopping_ratio`` plays
the role of their percentage-deviation bound ``pdb`` (expressed as a
fraction): after each batch, the percentile interval endpoints of every free
parameter are recomputed from all converged replicates so far, and the run
stops once the maximum endpoint movement relative to the interval half-width
falls below ``mc_stopping_ratio``. Because the ideal endpoints are unknown
mid-run, successive-batch endpoint movement is used as the observable proxy;
the compute budget always caps the run. Bias-corrected-and-accelerated (BCa)
intervals are out of scope.

References
----------
- Andrews, D. W. K., & Buchinsky, M. (2000). A three-step method for choosing
  the number of bootstrap repetitions. *Econometrica, 68*(1), 23–51.
  https://www.jstor.org/stable/2999474 (full text: Cowles Foundation Paper
  No. 1001, http://dido.econ.yale.edu/~dwka/pub/p1001.pdf; see eqs.
  (4.1)–(4.4) for the batch-size formulae and § 6 for the 95% interval
  simulations motivating the default ``ci_level``).
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
  analysis. *Psychometrika, 57*(3), 423–436.
  https://doi.org/10.1007/BF02295430
- Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
  item bifactor analysis. *Psychological Methods, 16*(3), 221–248.
  https://doi.org/10.1037/a0023350
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import math
import os
import time
import numpy as np

from .polytomous_bifactor import fit_polytomous_bifactor


@dataclass
class BifactorBootstrapResult:
    """Summary and replicate storage of a joint person bootstrap run."""

    n_requested: int
    n_replicates: int  # replicates actually completed (≤ n_requested)
    n_converged: int
    converged: np.ndarray  # bool per completed replicate, in replicate order
    stopped_early: bool  # True when the MC stopping rule fired
    ci_level: float
    replicate_slopes: np.ndarray  # (B_conv, n_items, n_dims)
    replicate_thresholds: np.ndarray  # (B_conv, n_items, n_cat - 1)
    replicate_group_means: np.ndarray  # (B_conv, n_groups, n_dims)
    replicate_group_variances: np.ndarray  # (B_conv, n_groups, n_dims)
    replicate_loglik: np.ndarray  # (B_conv,)
    se_slope: np.ndarray  # (n_items, n_dims)
    se_threshold: np.ndarray  # (n_items, n_cat - 1)
    se_group_means: np.ndarray  # (n_groups, n_dims)
    se_group_variances: np.ndarray  # (n_groups, n_dims)
    ci_lower_slope: np.ndarray
    ci_upper_slope: np.ndarray
    ci_lower_threshold: np.ndarray
    ci_upper_threshold: np.ndarray
    ci_lower_group_means: np.ndarray
    ci_upper_group_means: np.ndarray
    ci_lower_group_variances: np.ndarray
    ci_upper_group_variances: np.ndarray
    wall_clock_seconds: float
    throughput_replicates_per_second: float
    device: str = "cpu"


def _generate_bootstrap_indices(
    n_persons: int,
    group_ids: np.ndarray | None,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap sample indices, with stratification if multiple groups.

    Implementation basis: nonparametric iid person resampling within each
    known group (the resampling scheme to which Andrews and Buchinsky (2000,
    § 2) apply their replicate-number results for iid data; Andrews, D. W. K.,
    & Buchinsky, M. (2000). A three-step method for choosing the number of
    bootstrap repetitions. *Econometrica, 68*(1), 23–51.
    https://www.jstor.org/stable/2999474).
    """
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
    device: str,
) -> tuple[int, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute one bootstrap resample and fit.

    The replicate seed derives deterministically from the caller-supplied
    ``base_seed`` and the replicate index, so CPU and GPU runs with the same
    ``base_seed`` draw identical resamples and identical QMC shifts and match
    replicate-by-replicate up to device precision.
    """
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
            device=device,
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


def _stack_monitor_vector(
    slopes: list[np.ndarray],
    thresholds: list[np.ndarray],
    means: list[np.ndarray],
    variances: list[np.ndarray],
) -> np.ndarray:
    """Stack converged replicate estimates into one row-per-replicate matrix."""
    rows = [
        np.concatenate([s.ravel(), t.ravel(), m.ravel(), v.ravel()])[None, :]
        for s, t, m, v in zip(slopes, thresholds, means, variances)
    ]
    return np.concatenate(rows, axis=0)


def _endpoint_movement(
    prev_lo: np.ndarray,
    prev_hi: np.ndarray,
    new_lo: np.ndarray,
    new_hi: np.ndarray,
) -> float:
    """Maximum endpoint movement relative to the new interval half-width.

    Entries with zero half-width (parameters that are constant across
    replicates, e.g. fixed reference-group moments) carry no Monte Carlo
    uncertainty and are excluded. This is the observable proxy for the
    percentage deviation of finite-``B`` interval endpoints from their ideal
    counterparts in Andrews and Buchinsky (2000, §§ 2–4).
    """
    half = (new_hi - new_lo) / 2.0
    live = half > 0
    if not np.any(live):
        return 0.0
    move = np.maximum(np.abs(new_lo[live] - prev_lo[live]),
                      np.abs(new_hi[live] - prev_hi[live])) / half[live]
    return float(np.max(move))


def run_bifactor_bootstrap(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    n_replicates: int,
    batch_size: int,
    mc_stopping_ratio: float,
    compute_budget_seconds: float,
    group_ids: np.ndarray | None = None,
    n_groups: int = 1,
    n_jobs: int = -1,
    base_seed: int = 42,
    device: str = "cpu",
    ci_level: float = 0.95,
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
        n_replicates: Total bootstrap replicates requested (caller-supplied;
            no study-specific default is defined by this module).
        batch_size: Replicates per batch; the stopping rule and the compute
            budget are evaluated at batch boundaries.
        mc_stopping_ratio: Bound on the Monte Carlo error of the percentile
            interval endpoints relative to the interval half-width, in the
            role of the percentage-deviation bound ``pdb`` of Andrews and
            Buchinsky (2000, §§ 2–4). Must satisfy ``0 <= ratio < 1``;
            ``0`` disables early stopping (the run completes all requested
            replicates within budget).
        compute_budget_seconds: Wall-clock budget; batch execution stops when
            the elapsed time reaches this bound.
        group_ids: Optional 1-D group membership indices.
        n_groups: Number of groups.
        n_jobs: Number of parallel workers (-1 for all logical cores).
        base_seed: Master seed for deterministic replication.
        device: 'cpu', 'gpu', or 'auto' execution device.
        ci_level: Nominal level of the reported percentile intervals and of
            the endpoints monitored by the stopping rule (0 < level < 1).
        slope_bound: Upper bound on slope magnitude.
        max_iter: Max EM iterations per replicate.
        tol: Convergence tolerance.
        ridge: Ridge stabilization penalty.
        newton_iter: Newton steps per M-step.
        qmc_draws: Halton draws per person.

    Returns:
        BifactorBootstrapResult with replicate matrices, empirical SEs,
        percentile intervals, per-replicate convergence flags, and timing.
        Replicates that fail to converge (or raise) are reported in
        ``converged`` and excluded from all summary statistics; failed
        replicates are never substituted or imputed.

    References:
        Andrews, D. W. K., & Buchinsky, M. (2000). A three-step method for
        choosing the number of bootstrap repetitions. *Econometrica, 68*(1),
        23–51. https://www.jstor.org/stable/2999474
    """
    if isinstance(n_replicates, bool) or not isinstance(n_replicates, int) or n_replicates < 1:
        raise ValueError(f"n_replicates must be a positive integer, got {n_replicates!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        isinstance(mc_stopping_ratio, bool)
        or not isinstance(mc_stopping_ratio, (float, int))
        or not math.isfinite(mc_stopping_ratio)
        or not 0.0 <= mc_stopping_ratio < 1.0
    ):
        raise ValueError(
            "mc_stopping_ratio must satisfy 0 <= ratio < 1 "
            f"(0 disables early stopping), got {mc_stopping_ratio!r}"
        )
    if (
        isinstance(compute_budget_seconds, bool)
        or not isinstance(compute_budget_seconds, (float, int))
        or not math.isfinite(compute_budget_seconds)
        or compute_budget_seconds <= 0
    ):
        raise ValueError(
            f"compute_budget_seconds must be positive and finite, got {compute_budget_seconds!r}"
        )
    if (
        isinstance(ci_level, bool)
        or not isinstance(ci_level, (float, int))
        or not math.isfinite(ci_level)
        or not 0.0 < ci_level < 1.0
    ):
        raise ValueError(f"ci_level must satisfy 0 < level < 1, got {ci_level!r}")
    if not isinstance(device, str) or device.strip().lower() not in ("cpu", "gpu", "auto"):
        raise ValueError(f"device must be one of 'cpu', 'gpu', 'auto'; got {device!r}")
    if isinstance(qmc_draws, bool) or not isinstance(qmc_draws, int) or qmc_draws < 1:
        raise ValueError(f"qmc_draws must be a positive integer, got {qmc_draws!r}")
    device = device.strip().lower()

    start_time = time.perf_counter()

    if n_jobs <= 0:
        n_jobs = max(1, os.cpu_count() or 1)

    y_arr = np.asarray(responses, dtype=np.float64)
    lp_arr = np.asarray(loading_pattern, dtype=np.uint8)
    n_items, n_dims = lp_arr.shape
    g_arr = np.asarray(group_ids, dtype=np.int64) if group_ids is not None else None

    alpha = 1.0 - float(ci_level)
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

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
            device,
        ))

    results: list = [None] * n_replicates
    converged_flags = np.zeros(0, dtype=bool)
    converged_slopes: list[np.ndarray] = []
    converged_thresholds: list[np.ndarray] = []
    converged_means: list[np.ndarray] = []
    converged_vars: list[np.ndarray] = []
    converged_logliks: list[float] = []

    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    completed_reps = 0
    stopped_early = False
    prev_lo: np.ndarray | None = None
    prev_hi: np.ndarray | None = None
    batches_completed = 0

    for batch_pos, batch in enumerate(batches):
        # Check compute budget
        elapsed_so_far = time.perf_counter() - start_time
        if elapsed_so_far >= compute_budget_seconds:
            break

        if n_jobs == 1 or len(batch) == 1:
            for t in batch:
                res = _fit_single_replicate(*t)
                results[res[0]] = res
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_fit_single_replicate, *t) for t in batch]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    results[res[0]] = res

        batch_flags = np.array(
            [results[t[0]][1] for t in batch], dtype=bool
        )
        converged_flags = np.concatenate([converged_flags, batch_flags])
        completed_reps += len(batch)
        batches_completed += 1

        for t in batch:
            r = results[t[0]]
            if r is not None and r[1]:  # converged
                converged_slopes.append(r[2])
                converged_thresholds.append(r[3])
                converged_means.append(r[4])
                converged_vars.append(r[5])
                converged_logliks.append(r[6])

        # Monte Carlo stopping check at batch boundaries (needs two endpoint
        # estimates to measure movement, so it can only fire from the second
        # completed batch on, and never on the final batch).
        last_batch = batch_pos == len(batches) - 1
        if (
            mc_stopping_ratio > 0
            and batches_completed >= 2
            and not last_batch
            and converged_slopes
        ):
            stacked = _stack_monitor_vector(
                converged_slopes, converged_thresholds, converged_means, converged_vars
            )
            new_lo = np.percentile(stacked, lo_q, axis=0)
            new_hi = np.percentile(stacked, hi_q, axis=0)
            if prev_lo is not None and prev_hi is not None:
                movement = _endpoint_movement(prev_lo, prev_hi, new_lo, new_hi)
                if movement < mc_stopping_ratio:
                    stopped_early = True
                    break
            prev_lo, prev_hi = new_lo, new_hi

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

        ci_lower_slope = np.percentile(slopes_mat, lo_q, axis=0)
        ci_upper_slope = np.percentile(slopes_mat, hi_q, axis=0)
        ci_lower_thresh = np.percentile(thresh_mat, lo_q, axis=0)
        ci_upper_thresh = np.percentile(thresh_mat, hi_q, axis=0)
        ci_lower_means = np.percentile(means_mat, lo_q, axis=0)
        ci_upper_means = np.percentile(means_mat, hi_q, axis=0)
        ci_lower_vars = np.percentile(vars_mat, lo_q, axis=0)
        ci_upper_vars = np.percentile(vars_mat, hi_q, axis=0)
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

        ci_lower_slope = np.full((n_items, n_dims), np.nan)
        ci_upper_slope = np.full((n_items, n_dims), np.nan)
        ci_lower_thresh = np.full((n_items, n_cat - 1), np.nan)
        ci_upper_thresh = np.full((n_items, n_cat - 1), np.nan)
        ci_lower_means = np.full((n_groups, n_dims), np.nan)
        ci_upper_means = np.full((n_groups, n_dims), np.nan)
        ci_lower_vars = np.full((n_groups, n_dims), np.nan)
        ci_upper_vars = np.full((n_groups, n_dims), np.nan)

    elapsed = time.perf_counter() - start_time
    throughput = completed_reps / elapsed if elapsed > 0 else 0.0

    return BifactorBootstrapResult(
        n_requested=n_replicates,
        n_replicates=completed_reps,
        n_converged=n_conv,
        converged=converged_flags,
        stopped_early=stopped_early,
        ci_level=float(ci_level),
        replicate_slopes=slopes_mat,
        replicate_thresholds=thresh_mat,
        replicate_group_means=means_mat,
        replicate_group_variances=vars_mat,
        replicate_loglik=loglik_vec,
        se_slope=se_slope,
        se_threshold=se_thresh,
        se_group_means=se_means,
        se_group_variances=se_vars,
        ci_lower_slope=ci_lower_slope,
        ci_upper_slope=ci_upper_slope,
        ci_lower_threshold=ci_lower_thresh,
        ci_upper_threshold=ci_upper_thresh,
        ci_lower_group_means=ci_lower_means,
        ci_upper_group_means=ci_upper_means,
        ci_lower_group_variances=ci_lower_vars,
        ci_upper_group_variances=ci_upper_vars,
        wall_clock_seconds=elapsed,
        throughput_replicates_per_second=throughput,
        device=device,
    )
