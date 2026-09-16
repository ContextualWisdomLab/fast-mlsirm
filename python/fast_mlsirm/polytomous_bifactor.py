# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Multidimensional Polytomous Bifactor Graded Response Model (GRM).

Provides full-information item bifactor analysis with Quasi-Monte Carlo EM (QMCEM),
multiple-group simultaneous calibration, slope bounding, Oakes standard errors,
and two-stage Lord-Wingersky recursion.

References:
    - Samejima, F. (1969). Estimation of latent ability using a response
      pattern of graded scores. *Psychometrika, 34*, 1–97.
      https://doi.org/10.1007/BF03372160
    - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis.
      *Psychometrika, 57*(3), 423-436. https://doi.org/10.1007/BF02295430
    - Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis.
      *Psychological Methods, 16*(3), 221-248. https://doi.org/10.1037/a0023350
    - Bock, R. D., & Zimowski, M. F. (1997). Multiple Group IRT.
      In W. J. van der Linden & R. K. Hambleton (Eds.),
      *Handbook of Modern Item Response Theory*. Springer.
    - Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm.
      *Journal of the Royal Statistical Society: Series B, 61*(2), 479-482.
      https://doi.org/10.1111/1467-9868.00188
    - Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of
      Monte Carlo EM. *Computational Statistics & Data Analysis, 48*(4), 685-701.
      https://doi.org/10.1016/j.csda.2004.03.019
    - Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile equating.
      *Applied Psychological Measurement, 8*(4), 453-461.
      https://doi.org/10.1177/014662168400800409
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
    compute_oakes_se: bool = True,
    device: str = "cpu",
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
            Quadrature count is precision: it is caller-controlled with no
            upper cap. Study-setting fits use at least 121 draws and check
            stabilization at higher counts (e.g. 241, 481); the GPU E-step
            kernels accumulate in f32, so CPU/GPU parity targets the looser
            single-precision agreement documented in
            `tests/test_bifactor_gpu.py`.
        seed: Random seed for deterministic Halton sequence shift.
        slope_bound: Optional upper bound |a_id| <= slope_bound on discrimination magnitude.
        compute_oakes_se: Whether to compute Oakes observed information standard errors.
        device: 'cpu' runs the f64 scalar E-step; 'gpu' runs the WGSL f32
            E-step kernels and falls back to CPU (with a warning) when no GPU
            adapter is available; 'auto' prefers GPU without warning.

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

    missing = np.isnan(y_raw) | (y_raw < 0)
    observed = ~missing
    y_clean = np.where(observed, y_raw, 0).astype(np.int64)

    if group_ids is not None:
        g_arr = np.asarray(group_ids, dtype=np.int64)
        if g_arr.ndim != 1 or g_arr.size != n_persons:
            raise ValueError("group_ids must have length n_persons")
        n_groups = max(n_groups, int(g_arr.max()) + 1)
    else:
        g_arr = None
        n_groups = max(n_groups, 1)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "fit_bifactor_grm"):
        raise RuntimeError("Polytomous bifactor fitting requires compiled Rust core")

    result = core.fit_bifactor_grm(
        y_clean,
        observed,
        g_arr,
        n_groups,
        lp,
        n_cat,
        max_iter,
        tol,
        ridge,
        newton_iter,
        qmc_draws,
        seed,
        slope_bound,
        compute_oakes_se,
        device,
    )

    return PolytomousBifactorFit(
        model="grm_bifactor",
        n_dims=n_dims,
        n_items=n_items,
        n_cat=n_cat,
        n_groups=n_groups,
        slope=np.asarray(result["slope"], dtype=np.float64).reshape(n_items, n_dims),
        threshold=np.asarray(result["threshold"], dtype=np.float64).reshape(n_items, n_cat - 1),
        group_means=np.asarray(result["group_means"], dtype=np.float64).reshape(n_groups, n_dims),
        group_variances=np.asarray(result["group_variances"], dtype=np.float64).reshape(n_groups, n_dims),
        loglik=float(result["loglik"]),
        loglik_trace=np.asarray(result["loglik_trace"], dtype=np.float64),
        n_iter=int(result["n_iter"]),
        converged=bool(result["converged"]),
        oakes_se_slope=np.asarray(result["oakes_se_slope"], dtype=np.float64).reshape(n_items, n_dims) if "oakes_se_slope" in result else None,
        oakes_se_threshold=np.asarray(result["oakes_se_threshold"], dtype=np.float64).reshape(n_items, n_cat - 1) if "oakes_se_threshold" in result else None,
        min_eigenvalue=float(result["min_eigenvalue"]) if "min_eigenvalue" in result else None,
        condition_number=float(result["condition_number"]) if "condition_number" in result else None,
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
    device: str = "cpu",
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
            device,
        )
    )
