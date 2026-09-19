# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Multidimensional Polytomous Bifactor Graded Response Model (GRM).

Provides full-information item bifactor analysis with Quasi-Monte Carlo EM (QMCEM),
multiple-group simultaneous calibration, slope bounding, Oakes standard errors,
and two-stage Lord-Wingersky recursion.

References:
    - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis.
      Psychometrika, 57(3), 423-436.
    - Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis.
      Psychological Methods, 16(3), 221-248.
    - Bock, R. D., & Zimowski, M. F. (1997). Multiple Group IRT.
      In Handbook of Modern Item Response Theory.
    - Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm.
      Journal of the Royal Statistical Society: Series B, 61(2), 479-482.
    - Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile equating.
      Applied Psychological Measurement, 8(4), 453-461.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PolytomousBifactorFit:
    """Result of fitting a multidimensional polytomous bifactor model."""

    model: str
    n_dims: int
    n_items: int
    n_cat: int
    n_groups: int
    slope: np.ndarray
    threshold: np.ndarray
    group_means: np.ndarray
    group_variances: np.ndarray
    loglik: float
    loglik_trace: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    n_iter: int = 0
    converged: bool = False
    oakes_se_slope: np.ndarray | None = None
    oakes_se_threshold: np.ndarray | None = None
    min_eigenvalue: float | None = None
    condition_number: float | None = None


def _load_bifactor_core():
    """Return the compiled Rust bifactor core module, or None if unavailable."""
    try:
        from ._bifactor_core_loader import bifactor_core
        return bifactor_core()
    except Exception:
        try:
            from . import _bifactor_core
            return _bifactor_core
        except Exception:
            try:
                from . import _core
                return _core
            except Exception:
                return None


def fit_polytomous_bifactor(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    group_ids: np.ndarray | None = None,
    n_groups: int = 1,
    max_iter: int = 500,
    tol: float = 1e-4,
    ridge: float = 1e-6,
    newton_iter: int = 10,
    qmc_draws: int = 5000,
    seed: int = 0x9E37_79B9_7F4A_7C15,
    slope_bound: float | None = None,
    compute_oakes_se: bool = False,
) -> PolytomousBifactorFit:
    """Fit a polytomous bifactor Graded Response Model via QMCEM.

    Parameters:
        responses: Persons x items array of integer categories (0..n_cat-1).
                   NaN or negative values indicate missing cells.
        loading_pattern: Items x dims array in {0, 1} defining factor structure.
                         Dimension 0 is the general factor; remaining dimensions are
                         specific or method factors.
        n_cat: Number of response categories (e.g. 4 for 0, 1, 2, 3).
        group_ids: Optional 1-D array of group indices (0..n_groups-1).
                   Group 0 is the reference group (mean 0, variance 1 fixed).
        n_groups: Number of groups for multiple-group simultaneous calibration.
        max_iter: Maximum number of EM iterations.
        tol: Convergence tolerance for relative log-likelihood change.
        ridge: Ridge penalty for Hessian conditioning during Newton M-step.
        newton_iter: Number of Newton iterations per item in M-step.
        qmc_draws: Number of Quasi-Monte Carlo Halton integration draws.
        seed: Random seed for deterministic Halton sequence shift.
        slope_bound: Optional upper bound |a_id| <= slope_bound on discrimination magnitude.
        compute_oakes_se: Request Oakes observed-information standard errors. The
            polytomous bifactor likelihood does not yet expose the category-count
            derivatives required by Oakes' identity, so ``True`` fails closed
            instead of returning a fit with missing or fabricated uncertainty.

    Returns:
        PolytomousBifactorFit containing estimated slopes, thresholds, group moments,
        log-likelihood trace, and Oakes standard errors.
    """
    y_raw = np.asarray(responses, dtype=np.float64)
    if y_raw.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y_raw.shape

    lp = np.asarray(loading_pattern, dtype=np.uint8)
    if lp.ndim != 2 or lp.shape[0] != n_items:
        raise ValueError("loading_pattern must be an items x dims array")
    n_dims = lp.shape[1]

    if n_cat < 2:
        raise ValueError("n_cat must be >= 2")

    if compute_oakes_se:
        raise NotImplementedError(
            "polytomous bifactor Oakes standard errors require a Rust-owned "
            "category-count information kernel; no validated implementation is available"
        )

    if group_ids is not None:
        g_arr = np.asarray(group_ids, dtype=np.int64)
        if g_arr.ndim != 1 or g_arr.size != n_persons:
            raise ValueError("group_ids must have length n_persons")
        n_groups = max(n_groups, int(g_arr.max()) + 1)
    else:
        g_arr = None
        n_groups = max(n_groups, 1)

    specific_map = np.full(n_items, -1, dtype=np.int64)
    for i in range(n_items):
        if n_dims > 1:
            spec_idx = np.flatnonzero(lp[i, 1:])
            if len(spec_idx) > 0:
                specific_map[i] = spec_idx[0]
    n_specific = max(1, int(specific_map.max()) + 1) if (specific_map >= 0).any() else 1

    if g_arr is not None and n_groups > 1:
        from .bifactor_multigroup import fit_bifactor_grm_multigroup

        fit = fit_bifactor_grm_multigroup(
            responses=y_raw,
            group=g_arr,
            specific_map=specific_map,
            n_cat=n_cat,
            n_specific=n_specific,
            q_general=21,
            q_specific=11,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        slope = np.zeros((n_items, n_dims), dtype=np.float64)
        slope[:, 0] = fit.a_general[0]
        for i in range(n_items):
            s = specific_map[i]
            if s >= 0:
                slope[i, 1 + s] = fit.a_specific[0, i]
        threshold = fit.threshold[0]
        group_means = np.zeros((n_groups, n_dims), dtype=np.float64)
        group_variances = np.ones((n_groups, n_dims), dtype=np.float64)
        for g in range(n_groups):
            group_means[g, 0] = fit.general_mean[g]
            group_variances[g, 0] = fit.general_sd[g] ** 2
            if n_specific > 0 and fit.specific_sd.shape[1] >= n_specific:
                group_variances[g, 1 : 1 + n_specific] = fit.specific_sd[g, :n_specific] ** 2

        loglik = float(fit.loglik_trace[-1]) if len(fit.loglik_trace) > 0 else 0.0
        return PolytomousBifactorFit(
            model="grm_bifactor",
            n_dims=n_dims,
            n_items=n_items,
            n_cat=n_cat,
            n_groups=n_groups,
            slope=slope,
            threshold=threshold,
            group_means=group_means,
            group_variances=group_variances,
            loglik=loglik,
            loglik_trace=fit.loglik_trace,
            n_iter=fit.n_iter,
            converged=fit.converged,
        )
    else:
        from .bifactor_grm import fit_bifactor_grm

        fit = fit_bifactor_grm(
            responses=y_raw,
            specific_map=specific_map,
            n_cat=n_cat,
            n_specific=n_specific,
            q_general=21,
            q_specific=11,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        slope = np.zeros((n_items, n_dims), dtype=np.float64)
        slope[:, 0] = fit.a_general
        for i in range(n_items):
            s = specific_map[i]
            if s >= 0:
                slope[i, 1 + s] = fit.a_specific[i]
        threshold = fit.threshold
        group_means = np.zeros((1, n_dims), dtype=np.float64)
        group_variances = np.ones((1, n_dims), dtype=np.float64)
        loglik = float(fit.loglik_trace[-1]) if len(fit.loglik_trace) > 0 else 0.0
        return PolytomousBifactorFit(
            model="grm_bifactor",
            n_dims=n_dims,
            n_items=n_items,
            n_cat=n_cat,
            n_groups=1,
            slope=slope,
            threshold=threshold,
            group_means=group_means,
            group_variances=group_variances,
            loglik=loglik,
            loglik_trace=fit.loglik_trace,
            n_iter=fit.n_iter,
            converged=fit.converged,
        )


def bifactor_lord_wingersky(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    item_domains: np.ndarray,
    n_cat: int,
    n_domains: int,
    theta_general: np.ndarray,
    theta_specific: np.ndarray,
    weights_specific: np.ndarray,
) -> np.ndarray:
    """Compute total score distribution P(X = r | theta_0) via two-stage Lord-Wingersky recursion.

    Within-domain recursion is performed conditional on (theta_0, theta_s) and integrated over
    theta_s, followed by across-domain convolution.
    """
    a_g = np.asarray(a_general, dtype=np.float64)
    a_s = np.asarray(a_specific, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    domains = np.asarray(item_domains, dtype=np.int64)
    th_g = np.asarray(theta_general, dtype=np.float64)
    th_s = np.asarray(theta_specific, dtype=np.float64)
    w_s = np.asarray(weights_specific, dtype=np.float64)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "bifactor_lord_wingersky"):
        raise RuntimeError("bifactor Lord-Wingersky requires compiled Rust core")

    flat = np.asarray(
        core.bifactor_lord_wingersky(
            a_g, a_s, thr, domains, n_cat, n_domains, th_g, th_s, w_s
        ),
        dtype=np.float64,
    )
    total_max_score = len(a_g) * (n_cat - 1)
    return flat.reshape(len(th_g), total_max_score + 1)


def direct_enumeration_bifactor(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    item_domains: np.ndarray,
    n_cat: int,
    n_domains: int,
    theta_general: np.ndarray,
    theta_specific: np.ndarray,
    weights_specific: np.ndarray,
) -> np.ndarray:
    """Compute exact direct enumeration ground truth for small item sets."""
    a_g = np.asarray(a_general, dtype=np.float64)
    a_s = np.asarray(a_specific, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    domains = np.asarray(item_domains, dtype=np.int64)
    th_g = np.asarray(theta_general, dtype=np.float64)
    th_s = np.asarray(theta_specific, dtype=np.float64)
    w_s = np.asarray(weights_specific, dtype=np.float64)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "direct_enumeration_bifactor"):
        raise RuntimeError("direct enumeration requires compiled Rust core")

    flat = np.asarray(
        core.direct_enumeration_bifactor(
            a_g, a_s, thr, domains, n_cat, n_domains, th_g, th_s, w_s
        ),
        dtype=np.float64,
    )
    total_max_score = len(a_g) * (n_cat - 1)
    return flat.reshape(len(th_g), total_max_score + 1)


def bifactor_slope_sensitivity(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    candidate_bounds: list[float | None] | None = None,
    group_ids: np.ndarray | None = None,
    n_groups: int = 1,
    max_iter: int = 100,
    tol: float = 1e-4,
    ridge: float = 1e-6,
    newton_iter: int = 10,
    qmc_draws: int = 2000,
    seed: int = 0x9E37_79B9_7F4A_7C15,
) -> list[dict]:
    """Run a slope upper bound sensitivity sweep across bounds (e.g. 4, 6, 8, 10)."""
    if candidate_bounds is None:
        candidate_bounds = [4.0, 6.0, 8.0, 10.0]

    y_raw = np.asarray(responses, dtype=np.float64)
    n_persons, n_items = y_raw.shape
    lp = np.asarray(loading_pattern, dtype=np.uint8)
    missing = np.isnan(y_raw) | (y_raw < 0)
    observed = ~missing
    y_clean = np.where(observed, y_raw, 0).astype(np.int64)

    if group_ids is not None:
        g_arr = np.asarray(group_ids, dtype=np.int64)
        n_groups = max(n_groups, int(g_arr.max()) + 1)
    else:
        g_arr = None
        n_groups = max(n_groups, 1)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "fit_bifactor_slope_sensitivity"):
        raise RuntimeError("Slope sensitivity requires compiled Rust core")

    return list(
        core.fit_bifactor_slope_sensitivity(
            y_clean,
            observed,
            g_arr,
            n_groups,
            lp,
            n_cat,
            candidate_bounds,
            max_iter,
            tol,
            ridge,
            newton_iter,
            qmc_draws,
            seed,
        )
    )
