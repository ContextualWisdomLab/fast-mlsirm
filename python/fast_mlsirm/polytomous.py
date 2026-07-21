"""Unidimensional polytomous item-response fitting (GRM / GPCM).

Thin orchestration over the Rust compute path (``mlsirm_core::poly``): all
numerical work — the Bock-Aitkin marginal-EM loop, the category cells, and the
Newton M-step — runs in Rust. This is the classic (no latent-space) polytomous
model; the latent-space polytomous LSIRM extension slots the same category cell
into the marginal (theta, xi) quadrature and is the next milestone (see
``docs/papers/gpcm-nominal-design-spec.md``).

``GRM`` (Samejima cumulative logit) is the default; ``GPCM`` (Muraki
adjacent-category) is available for partial-credit scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    MAX_MAX_ITER,
    MAX_POLYTOMOUS_CATEGORIES,
    MAX_SIM_CELLS,
    MAX_SIM_PERSONS,
)

__all__ = [
    "PolytomousFit",
    "fit_polytomous",
    "score_polytomous",
    "information_polytomous",
    "PolyLsirmFit",
    "fit_lsirm_polytomous",
    "polytomous_information_criteria",
]

VALID_POLY_MODELS = {"grm", "gpcm"}
MAX_POLY_QUADRATURE_POINTS = 4_096
MAX_POLY_BOOTSTRAP_REPLICATES = 10_000
MAX_POLY_CAT_ITEMS = 10_000


def _bounded_integer(value, name: str, lower: int, upper: int) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or not lower <= int(value) <= upper
    ):
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return int(value)


def _quadrature_points(value) -> int:
    return _bounded_integer(value, "q_theta", 1, MAX_POLY_QUADRATURE_POINTS)


@dataclass
class PolytomousFit:
    """Result of :func:`fit_polytomous`.

    ``slope`` is the per-item discrimination ``a_i``. ``cat_params`` is
    ``n_items x (n_cat - 1)``: GPCM additive category intercepts, or GRM
    cumulative thresholds ``beta_{i,k}`` (ordered decreasing). ``thresholds``
    is the GPCM Muraki step reparametrization ``b_{i,k} = c_{i,k-1} - c_{i,k}``
    (``None`` for GRM, whose ``cat_params`` are already thresholds).
    """

    model: str
    slope: np.ndarray
    cat_params: np.ndarray
    loglik: float
    n_iter: int
    converged: bool = False
    termination_reason: str = "not_fitted"
    loglik_trace: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    final_delta: float = np.nan
    stopping_tolerance: float = np.nan
    thresholds: np.ndarray | None = None


def _core_module():
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover - core built in CI
        return None


def _poly_int_and_mask(responses: np.ndarray, n_cat: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate polytomous responses (``NaN`` = missing) and return
    ``(int64 categories with missing filled to 0, boolean observed mask)``."""
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or not 2 <= int(n_cat) <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(f"n_cat must be an integer between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    n_cat = int(n_cat)
    yf = np.asarray(responses, dtype=np.float64)
    if yf.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if np.any(np.isinf(yf)):
        raise ValueError("responses may only use NaN for missing values")
    observed = ~np.isnan(yf)
    obs_vals = yf[observed]
    if obs_vals.size and (
        np.any(obs_vals != np.floor(obs_vals)) or obs_vals.min() < 0 or obs_vals.max() >= n_cat
    ):
        raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")
    y_int = np.where(observed, yf, 0.0).astype(np.int64)
    return y_int, observed


def _nonnegative_integer_vector(values, name: str) -> np.ndarray:
    """Validate label/index vectors before their irreversible int64 cast."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    numeric = raw.astype(np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    return raw.astype(np.int64)


def fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    model: str = "grm",
    q_theta: int = 21,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> PolytomousFit:
    """Fit a unidimensional GRM or GPCM by marginal MLE (compute in Rust).

    ``responses`` is a persons x items array of integer categories
    ``0..n_cat-1``; ``NaN`` marks a missing response (marginalized out of the
    likelihood). ``model`` is ``"grm"`` (default) or ``"gpcm"``.
    ``theta ~ N(0, 1)`` on a ``q_theta``-node Gauss-Hermite grid. The returned
    convergence fields describe the observed-data likelihood at the returned
    parameter state; reaching ``max_iter`` is reported as nonconvergence.
    ``n_cat`` is limited to 2..64 and ``max_iter`` to 1..100,000.

    References
    ----------
    Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
    from incomplete data via the EM algorithm. *Journal of the Royal
    Statistical Society: Series B (Methodological), 39*(1), 1–22.
    https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

    Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
    *The Annals of Statistics, 11*(1), 95–103.
    https://doi.org/10.1214/aos/1176346060
    """
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or not 2 <= int(n_cat) <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(f"n_cat must be an integer between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or not 1 <= int(max_iter) <= MAX_MAX_ITER
    ):
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    y_int, observed = _poly_int_and_mask(responses, n_cat)

    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_unidim"):
        raise RuntimeError("fit_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_poly_unidim(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        obs_arg,
        m,
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    slope = np.asarray(res["slope"], dtype=np.float64)
    cat_params = np.asarray(res["cat_params"], dtype=np.float64)
    thresholds = None
    if m == "gpcm":
        # Muraki step difficulties from additive intercepts (baseline 0 prepended)
        c = np.concatenate([np.zeros((n_items, 1)), cat_params], axis=1)
        thresholds = c[:, :-1] - c[:, 1:]
    return PolytomousFit(
        model=m,
        slope=slope,
        cat_params=cat_params,
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        thresholds=thresholds,
    )


def score_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
) -> dict[str, np.ndarray]:
    """EAP trait scores for polytomous responses given a fitted model (compute
    in Rust). ``responses`` is persons x items of integer categories; ``fit`` is
    a :class:`PolytomousFit` from :func:`fit_polytomous`. ``NaN`` marks a
    missing response. The posterior mean and standard deviation are evaluated
    on a standard-normal quadrature grid (Bock & Mislevy, 1982). Returns
    ``{"theta_eap", "theta_sd"}``.

    References
    ----------
    Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
    a microcomputer environment. *Applied Psychological Measurement, 6*(4),
    431–444. https://doi.org/10.1177/014662168200600405
    """
    if (
        not isinstance(q_theta, int)
        or isinstance(q_theta, bool)
        or q_theta not in {7, 11, 15, 21, 31, 41}
    ):
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")

    slope = np.asarray(fit.slope, dtype=np.float64)
    cat_params = np.asarray(fit.cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.size == 0:
        raise ValueError("fit.slope must be a non-empty 1-D array")
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        raise ValueError("fit.cat_params must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(cat_params)):
        raise ValueError("fit item parameters must be finite")
    model = str(fit.model).lower()
    if model not in VALID_POLY_MODELS:
        raise ValueError(f"fit.model must be one of {sorted(VALID_POLY_MODELS)}")

    n_items = slope.shape[0]
    n_cat = cat_params.shape[1] + 1
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "score_poly_eap"):
        raise RuntimeError("score_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.score_poly_eap(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        slope,
        cat_params.reshape(-1),
        obs_arg,
        model,
        q_theta,
    )
    return {
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "theta_sd": np.asarray(res["theta_sd"], dtype=np.float64),
    }


def information_polytomous(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Item and test information curves for a fitted polytomous model (compute
    in Rust). ``theta`` is a 1-D grid of trait values. Returns
    ``{"item_info"` (n_theta x n_items), ``"test_info"`` (n_theta)}``. The
    model-specific information functions follow Samejima (1969) for the GRM
    and Muraki (1993) for the GPCM.

    References
    ----------
    Muraki, E. (1993). Information functions of the generalized partial credit
    model. *Applied Psychological Measurement, 17*(4), 351–363.
    https://doi.org/10.1177/014662169301700403

    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1–97.
    https://doi.org/10.1007/BF03372160
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim != 1 or th.size == 0 or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a non-empty finite 1-D grid")
    slope = np.asarray(fit.slope, dtype=np.float64)
    cat_params = np.asarray(fit.cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.size == 0:
        raise ValueError("fit.slope must be a non-empty 1-D array")
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        raise ValueError("fit.cat_params must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(cat_params)):
        raise ValueError("fit item parameters must be finite")
    model = str(fit.model).lower()
    if model not in VALID_POLY_MODELS:
        raise ValueError(f"fit.model must be one of {sorted(VALID_POLY_MODELS)}")
    core = _core_module()
    if core is None or not hasattr(core, "poly_information_curves"):
        raise RuntimeError("information_polytomous requires the compiled Rust core")

    n_items = slope.shape[0]
    n_cat = cat_params.shape[1] + 1
    flat = core.poly_information_curves(
        th,
        slope,
        cat_params.reshape(-1),
        int(n_items),
        int(n_cat),
        model,
    )
    item_info = np.asarray(flat, dtype=np.float64).reshape(th.size, n_items)
    return {"item_info": item_info, "test_info": item_info.sum(axis=1)}


@dataclass
class PolyLsirmFit:
    """Result of :func:`fit_lsirm_polytomous` — a latent-space polytomous LSIRM.

    ``slope``/``cat_params`` are the item parameters; ``zeta`` is the
    ``n_items x latent_dim`` item interaction-map positions (identified up to
    rotation/reflection/translation — compare via distances). ``theta_eap`` /
    ``theta_sd`` are per-person EAP trait scores and SDs; ``xi_eap`` is the
    ``n_persons x latent_dim`` person positions.
    """

    model: str
    slope: np.ndarray
    cat_params: np.ndarray
    zeta: np.ndarray
    theta_eap: np.ndarray
    theta_sd: np.ndarray
    xi_eap: np.ndarray
    loglik: float
    n_iter: int


def fit_lsirm_polytomous(
    responses: np.ndarray,
    n_cat: int,
    latent_dim: int = 2,
    model: str = "grm",
    q_theta: int = 11,
    q_xi: int = 11,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> PolyLsirmFit:
    """Fit a latent-space polytomous LSIRM (GRM/GPCM cell in an interaction map)
    by marginal EM — all compute in the Rust core (``poly_marginal``). The
    distance weight is fixed to 1 (Go et al. 2024 identification); positions are
    identified up to rotation/reflection/translation. ``NaN`` marks missing.
    ``n_cat`` is limited to 2..64 and ``max_iter`` to 1..100,000.
    """
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or not 2 <= int(n_cat) <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(f"n_cat must be an integer between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    if not isinstance(latent_dim, int) or not (1 <= latent_dim <= 3):
        raise ValueError("latent_dim must be an integer in 1..3")
    if q_theta not in {7, 11, 15, 21, 31, 41} or q_xi not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta/q_xi must be one of 7, 11, 15, 21, 31, 41")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or not 1 <= int(max_iter) <= MAX_MAX_ITER
    ):
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    y_int, observed = _poly_int_and_mask(responses, n_cat)
    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_lsirm"):
        raise RuntimeError("fit_lsirm_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_poly_lsirm(
        y_int.reshape(-1), int(n_persons), int(n_items), int(n_cat), int(latent_dim),
        obs_arg, m, int(q_theta), int(q_xi), int(max_iter), float(tol),
    )
    return PolyLsirmFit(
        model=m,
        slope=np.asarray(res["slope"], dtype=np.float64),
        cat_params=np.asarray(res["cat_params"], dtype=np.float64),
        zeta=np.asarray(res["zeta"], dtype=np.float64).reshape(n_items, latent_dim),
        theta_eap=np.asarray(res["theta_eap"], dtype=np.float64),
        theta_sd=np.asarray(res["theta_sd"], dtype=np.float64),
        xi_eap=np.asarray(res["xi_eap"], dtype=np.float64).reshape(n_persons, latent_dim),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
    )


def polytomous_information_criteria(fit, n_persons: int) -> dict[str, float]:
    """Relative model-selection indices for a polytomous fit (Kang, Cohen &
    Sung 2009, *Model Selection Indices for Polytomous Items*). Given a fitted
    :class:`PolytomousFit` or :class:`PolyLsirmFit` and the calibration sample
    size, returns ``AIC``, ``BIC``, ``CAIC``, ``AICc``, and the sample-size
    adjusted ``SABIC`` (all "smaller is better"), plus the free-parameter count.

    The parameter count is read from the fitted arrays: ``slope`` +
    ``cat_params`` (+ item positions ``zeta`` for the latent-space model).
    """
    if not isinstance(n_persons, int) or n_persons < 2:
        raise ValueError("n_persons must be an integer >= 2")
    k = int(np.asarray(fit.slope).size + np.asarray(fit.cat_params).size)
    zeta = getattr(fit, "zeta", None)
    if zeta is not None:
        k += int(np.asarray(zeta).size)
    ll = float(fit.loglik)
    n = int(n_persons)
    m2ll = -2.0 * ll
    aic = m2ll + 2.0 * k
    bic = m2ll + k * np.log(n)
    caic = m2ll + k * (np.log(n) + 1.0)
    aicc = aic + (2.0 * k * (k + 1.0)) / max(n - k - 1, 1)
    sabic = m2ll + k * np.log((n + 2.0) / 24.0)
    return {
        "n_parameters": k,
        "aic": float(aic),
        "bic": float(bic),
        "caic": float(caic),
        "aicc": float(aicc),
        "sabic": float(sabic),
    }


def item_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
    min_expected: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generalized S-X² item-fit statistic for an ordered polytomous fit
    (compute in Rust). Groups persons by summed score, compares observed to
    model-expected category proportions formed from the generalized
    Lord-Wingersky recursion, and returns per-item ``statistic``, ``df``,
    ``p_value``, and ``n_cells`` (the retained cell count, the reference df at
    known parameters). ``responses`` is persons x items of integer categories
    with ``NaN`` for missing; only persons complete on every item enter the
    summed-score table. At ``n_cat = 2`` this equals the binary Orlando-Thissen
    S-X². ``min_expected`` is the minimum expected cell frequency below which
    adjacent categories are collapsed.

    References (APA 7th ed.):
        Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X²
            item fit index for polytomous IRT models. *Journal of Educational
            Measurement, 45*(4), 391-406.
            https://doi.org/10.1111/j.1745-3984.2008.00070.x
        Kang, T., & Chen, T. T. (2011). Performance of the generalized S-X²
            item fit index for the graded response model. *Asia Pacific
            Education Review, 12*(1), 89-96.
            https://doi.org/10.1007/s12564-010-9082-4
        Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
            dichotomous item response theory models. *Applied Psychological
            Measurement, 24*(1), 50-64.
            https://doi.org/10.1177/01466216000241003
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _quadrature_points(q_theta)
    if not np.isfinite(min_expected) or min_expected <= 0:
        raise ValueError("min_expected must be positive")
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_item_fit_sx2"):
        raise RuntimeError("item_fit_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_item_fit_sx2(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
        float(min_expected),
    )
    return {
        "statistic": np.asarray(res["statistic"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "n_cells": np.asarray(res["n_cells"], dtype=np.int64),
    }


def m2_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
) -> dict[str, float]:
    """Polytomous M2 limited-information goodness-of-fit for a fitted GRM/GPCM
    (compute in Rust). Extends the binary M2 to ordered categories via the
    cumulative marginals ``P(Y_i >= c)`` and ``P(Y_i >= c, Y_j >= d)``; equals
    the binary M2 at ``n_cat = 2``. ``responses`` is persons x items of integer
    categories with ``NaN`` for missing (complete cases only enter the
    statistic). Returns ``m2``, ``df``, ``p_value``, ``rmsea2`` and its 90%
    interval (``rmsea2_ci_lower``/``rmsea2_ci_upper``), ``srmsr``, and
    ``cfi``/``tli`` from a complete-independence M2 baseline (``null_m2`` and
    ``null_df``), plus the ``n_moments``/``n_parameters``/``n_complete``
    counts. Requires at least 3 items and ``n_moments > n_parameters``. A fit
    carrying a known non-converged status is rejected because the reference
    distribution and derived fit indices require a completed calibration.

    References (APA 7th ed.):
        Cai, L., Chung, S. W., & Lee, T. (2023). Incremental model fit assessment
            in the case of categorical data: Tucker–Lewis index for item response
            theory modeling. *Prevention Science, 24*(3), 455–466.
            https://doi.org/10.1007/s11121-021-01253-4

        Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in
            categorical data analysis. *Multivariate Behavioral Research,
            49*(4), 305-328. https://doi.org/10.1080/00273171.2014.911075
    """
    q_theta = _quadrature_points(q_theta)
    if hasattr(fit, "converged") and not bool(fit.converged):
        reason = getattr(fit, "termination_reason", "unknown")
        n_iter = getattr(fit, "n_iter", "unknown")
        final_delta = getattr(fit, "final_delta", float("nan"))
        stopping_tolerance = getattr(fit, "stopping_tolerance", float("nan"))
        raise RuntimeError(
            "m2_polytomous requires a converged fit; "
            f"termination_reason={reason}, n_iter={n_iter}, "
            f"final_delta={final_delta}, stopping_tolerance={stopping_tolerance}"
        )

    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_m2"):
        raise RuntimeError("m2_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_m2(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
    )
    return {k: float(v) if k not in ("n_moments", "n_parameters", "n_complete")
            else int(v) for k, v in res.items()}


def local_dependence_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
) -> dict[str, np.ndarray]:
    """Item-pair local-dependence diagnostics for a fitted GRM/GPCM (compute in
    Rust; Chen & Thissen, 1997). For every item pair it compares the observed
    ``K x K`` contingency table against the model-implied joint under local
    independence and returns per-pair arrays: ``item_i``/``item_j`` (the pair),
    ``x2`` (Pearson) and ``g2`` (likelihood-ratio) statistics, ``p_value`` on
    ``chi2(df)`` with the shared ``df = (n_cat - 1) ** 2``, ``cramers_v`` effect
    size, ``max_abs_std_resid``, and ``n_pair`` (pairwise-complete sample size).
    A large ``x2``/``cramers_v`` on a pair flags residual association beyond the
    fitted trait (a local-dependence violation). ``responses`` is persons x
    items of integer categories with ``NaN`` for missing. The reference is
    heuristic and slightly conservative (Liu & Maydeu-Olivares, 2013), so read
    it as a diagnostic screen.

    References (APA 7th ed.):
        Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item
            pairs using item response theory. *Journal of Educational and
            Behavioral Statistics, 22*(3), 265-289.
            https://doi.org/10.3102/10769986022003265
        Liu, Y., & Maydeu-Olivares, A. (2013). Local dependence diagnostics in
            IRT modeling of binary data. *Educational and Psychological
            Measurement, 73*(2), 254-274.
            https://doi.org/10.1177/0013164412453841
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _quadrature_points(q_theta)
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_local_dependence"):
        raise RuntimeError("local_dependence_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_local_dependence(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
    )
    return {
        "item_i": np.asarray(res["item_i"], dtype=np.int64),
        "item_j": np.asarray(res["item_j"], dtype=np.int64),
        "x2": np.asarray(res["x2"], dtype=np.float64),
        "g2": np.asarray(res["g2"], dtype=np.float64),
        "df": float(res["df"]),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "cramers_v": np.asarray(res["cramers_v"], dtype=np.float64),
        "max_abs_std_resid": np.asarray(res["max_abs_std_resid"], dtype=np.float64),
        "n_pair": np.asarray(res["n_pair"], dtype=np.int64),
    }


@dataclass
class NominalFit:
    """Result of :func:`fit_nominal_polytomous`. ``scores`` and ``intercepts``
    are each ``n_items x (n_cat - 1)``: the free category scoring values
    ``a_{i,1}..a_{i,K-1}`` and intercepts ``c_{i,1}..c_{i,K-1}`` of the nominal
    model ``P(Y=k|theta) = softmax_k(a_k*theta + c_k)`` (baseline
    ``a_0 = c_0 = 0``). Parameters are identified up to the reflection
    ``(a_k, theta) -> (-a_k, -theta)``.
    """

    scores: np.ndarray
    intercepts: np.ndarray
    loglik: float
    n_iter: int
    converged: bool = False
    termination_reason: str = "not_fitted"
    loglik_trace: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    final_delta: float = np.nan
    stopping_tolerance: float = np.nan


def fit_nominal_polytomous(
    responses: np.ndarray,
    n_cat: int,
    q_theta: int = 21,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> NominalFit:
    """Fit the unidimensional nominal categories model by marginal MLE (compute
    in Rust; Bock, 1972; Thissen, Cai & Bock, 2010). Each item has a free scoring
    function ``a_k`` and intercept ``c_k`` per category,
    ``P(Y=k|theta) = softmax_k(a_k*theta + c_k)``, identified by ``a_0=c_0=0``
    with ``theta ~ N(0,1)``. The generalized partial credit model is the special
    case ``a_k = a*k``, so the nominal model nests it. ``responses`` is persons x
    items of integer categories ``0..n_cat-1``; ``NaN`` marks a missing response.
    As a repository-level convergence contract, the returned trace evaluates the
    observed-data log-likelihood at every returned parameter state;
    ``converged=False`` with ``termination_reason="max_iter"`` distinguishes an
    exhausted iteration budget from tolerance-based convergence.

    References (APA 7th ed.):
        Bock, R. D. (1972). Estimating item parameters and latent ability when
            responses are scored in two or more nominal categories.
            *Psychometrika, 37*(1), 29–51. https://doi.org/10.1007/BF02291411
        Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item
            response model. In *Handbook of polytomous item response theory
            models* (pp. 43-75). Routledge.
    """
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or not 2 <= int(n_cat) <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(f"n_cat must be an integer between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or not 1 <= int(max_iter) <= MAX_MAX_ITER
    ):
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[0] == 0 or y_int.shape[1] == 0:
        raise ValueError("responses must contain at least one person and one item")
    missing_items = np.flatnonzero(~observed.any(axis=0))
    if missing_items.size:
        raise ValueError(f"items with no observed responses: {missing_items.tolist()}")
    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal"):
        raise RuntimeError("fit_nominal_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_nominal(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        obs_arg,
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    return NominalFit(
        scores=np.asarray(res["scores"], dtype=np.float64),
        intercepts=np.asarray(res["intercepts"], dtype=np.float64),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
    )


def person_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    flag_threshold: float = -1.645,
) -> dict[str, np.ndarray]:
    """Person-fit statistics for polytomous responses under a fitted GRM/GPCM
    (compute in Rust). Returns the standardized log-likelihood ``lz`` (Drasgow,
    Levine & Williams, 1985) and its estimated-trait correction ``lz_star``
    (Snijders, 2001) at the EAP trait, plus ``theta_eap`` and a boolean
    ``flagged`` (``lz_star < flag_threshold``, i.e. an aberrant / misfitting
    response pattern). ``responses`` is persons x items of integer categories
    with ``NaN`` for missing; ``prior_mean``/``prior_sd`` set the MAP prior used
    in the Snijders correction. Reduces to the binary l_z at ``n_cat = 2``. Low
    (negative) values indicate poor person fit.

    References (APA 7th ed.):
        Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
            measurement with polychotomous item response models and standardized
            indices. *British Journal of Mathematical and Statistical
            Psychology, 38*(1), 67-86.
            https://doi.org/10.1111/j.2044-8317.1985.tb00817.x
        Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
            statistics with estimated person parameter. *Psychometrika, 66*(3),
            331-342. https://doi.org/10.1007/BF02294437
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _quadrature_points(q_theta)
    if not np.isfinite(prior_mean):
        raise ValueError("prior_mean must be finite")
    if not np.isfinite(prior_sd) or prior_sd <= 0:
        raise ValueError("prior_sd must be finite and > 0")
    if not np.isfinite(flag_threshold):
        raise ValueError("flag_threshold must be finite")
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_person_fit"):
        raise RuntimeError("person_fit_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_person_fit(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
        float(prior_mean),
        float(prior_sd),
        float(flag_threshold),
    )
    return {
        "lz": np.asarray(res["lz"], dtype=np.float64),
        "lz_star": np.asarray(res["lz_star"], dtype=np.float64),
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
    }


def cat_simulate_polytomous(
    true_theta: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
    se_threshold: float = 0.3,
    min_items: int = 5,
    max_items: int = 30,
    adaptive: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate a polytomous computerized adaptive test over a fitted GRM/GPCM
    item bank (compute in Rust; Dodd, De Ayala & Koch, 1995). For each true trait
    in ``true_theta`` it selects items by maximum Fisher information at the
    running EAP estimate (or at random when ``adaptive=False``), generates the
    response at the true trait, and re-estimates the trait after each item,
    stopping once at least ``min_items`` are given and the posterior SD is below
    ``se_threshold`` (or at ``max_items``; set ``se_threshold=0`` with
    ``min_items == max_items`` for a fixed-length CAT). Returns per-simulee
    ``theta_eap``, ``theta_sd`` (the final CAT standard error), and ``n_used``.

    References (APA 7th ed.):
        Dodd, B. G., De Ayala, R. J., & Koch, W. R. (1995). Computerized
            adaptive testing with polytomous items. *Applied Psychological
            Measurement, 19*(1), 5-22.
            https://doi.org/10.1177/014662169501900103
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    tt = np.asarray(true_theta, dtype=np.float64).ravel()
    if tt.size == 0 or not np.all(np.isfinite(tt)):
        raise ValueError("true_theta must be a non-empty finite 1-D array")
    q_theta = _quadrature_points(q_theta)
    min_items = _bounded_integer(min_items, "min_items", 1, MAX_POLY_CAT_ITEMS)
    max_items = _bounded_integer(max_items, "max_items", min_items, MAX_POLY_CAT_ITEMS)
    effective_max_items = min(max_items, n_items)
    if min_items > effective_max_items:
        raise ValueError("min_items must not exceed the fitted item-bank size")
    if not np.isfinite(se_threshold) or se_threshold < 0:
        raise ValueError("se_threshold must be finite and >= 0")
    if tt.size > MAX_SIM_PERSONS or tt.size * effective_max_items > MAX_SIM_CELLS:
        raise ValueError("polytomous CAT simulation exceeds the aggregate work limit")

    core = _core_module()
    if core is None or not hasattr(core, "poly_cat_simulate"):
        raise RuntimeError("cat_simulate_polytomous requires the compiled Rust core")

    res = core.poly_cat_simulate(
        tt,
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        int(n_items),
        int(n_cat),
        fit.model,
        int(q_theta),
        float(se_threshold),
        int(min_items),
        int(max_items),
        bool(adaptive),
        int(seed),
    )
    return {
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "theta_sd": np.asarray(res["theta_sd"], dtype=np.float64),
        "n_used": np.asarray(res["n_used"], dtype=np.int64),
    }


def dif_polytomous(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str = "gpcm",
    studied_items: np.ndarray | None = None,
    q_theta: int = 21,
    max_iter: int = 200,
    tol: float = 1e-5,
    fdr_q: float = 0.05,
) -> dict[str, np.ndarray]:
    """Likelihood-ratio DIF sweep for polytomous items via a two-group marginal-EM
    fit (compute in Rust; Thissen, Steinberg & Wainer, 1993). Group 0 is the
    reference (latent ``N(0, 1)``); each other group's latent ``N(mu_g,
    sigma_g^2)`` is estimated, so genuine ability differences between groups
    (impact) are absorbed rather than mistaken for DIF. It fits the *compact*
    model (all items group-invariant) once, then per studied item the *augmented*
    model (that item's parameters freed per group) with every other item as the
    anchor; ``LR = 2 * (loglik_aug - loglik_compact)`` is referred to
    ``chi2((n_groups - 1) * n_cat)``. Returns per-item arrays: ``item`` (index),
    ``lr``, ``df``, ``p_value``, ``flagged_bh`` (Benjamini-Hochberg FDR at
    ``fdr_q``), and ``effect_size`` (the unsigned across-group range of the item's
    mean category location -- a DIF magnitude >= 0, monotone in uniform DIF, not a
    direction). If an item's augmented fit fails to converge (e.g. GRM thresholds
    disorder on a sparse focal category) its ``lr``/``p_value``/``effect_size`` are
    ``NaN`` and it is left unflagged rather than silently reported as clean.

    ``responses`` is persons x items of integer categories (``NaN`` = missing);
    ``group_id`` is a length-persons integer array of group labels (any
    non-negative integers; densified internally, so non-contiguous or 1-based
    codes are fine).
    ``studied_items`` limits the sweep to those column indices (default: all
    items). ``model`` is ``"grm"`` or ``"gpcm"``; GPCM is recommended when focal
    groups have sparse extreme categories (GRM thresholds can become disordered
    on a rarely used category). This is the parametric IRT-LR approach; for an
    observed-score alternative that needs no multi-group calibration see the
    ordinal-logistic DIF of Zumbo (1999).

    References (APA 7th ed.):
        Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of
            differential item functioning using the parameters of item response
            models. In P. W. Holland & H. Wainer (Eds.), *Differential item
            functioning* (pp. 67-113). Erlbaum.
        Woehr, D. J., & Meriac, J. P. (2010). Using polytomous item response
            theory to examine differential item and test functioning: The case
            of work ethic. In J. A. Harkness, M. Braun, B. Edwards, T. P.
            Johnson, L. E. Lyberg, P. P. Mohler, B.-E. Pennell, & T. W. Smith
            (Eds.), *Survey methods in multinational, multiregional, and
            multicultural contexts* (pp. 419-433). Wiley.
            https://doi.org/10.1002/9780470609927.ch22
    """
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or not 2 <= int(n_cat) <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(f"n_cat must be an integer between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if (
        not isinstance(q_theta, (int, np.integer))
        or isinstance(q_theta, (bool, np.bool_))
        or q_theta not in {7, 11, 15, 21, 31, 41}
    ):
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or not 1 <= int(max_iter) <= MAX_MAX_ITER
    ):
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")

    y_int, observed = _poly_int_and_mask(responses, int(n_cat))
    n_persons, n_items = y_int.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    gid_raw = _nonnegative_integer_vector(group_id, "group_id")
    if gid_raw.shape[0] != n_persons:
        raise ValueError("group_id length must match the number of persons")
    # Densify labels so n_groups equals the number of *populated* groups and the
    # LR test's df = (n_groups - 1) * n_cat counts only groups backed by data.
    # Without this, sparse/non-contiguous labels (e.g. {0, 2} after filtering, or
    # 1-based codes) would leave phantom empty groups that inflate df and make the
    # test conservative. np.unique sorts, so the smallest label stays group 0
    # (the pinned N(0,1) reference).
    uniq, gid = np.unique(gid_raw, return_inverse=True)
    gid = gid.astype(np.int64)
    n_groups = uniq.size
    if n_groups < 2:
        raise ValueError("DIF requires at least two groups")

    core = _core_module()
    if core is None or not hasattr(core, "poly_dif"):
        raise RuntimeError("dif_polytomous requires the compiled Rust core")

    studied_arg = None
    if studied_items is not None:
        studied_arg = _nonnegative_integer_vector(studied_items, "studied_items")
        if np.any(studied_arg >= n_items):
            raise ValueError("studied_items entries must be valid item indices")
        if np.unique(studied_arg).size != studied_arg.size:
            raise ValueError("studied_items must not contain duplicates")
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_dif(
        y_int.reshape(-1),
        gid,
        int(n_groups),
        int(n_persons),
        int(n_items),
        int(n_cat),
        obs_arg,
        m,
        studied_arg,
        int(q_theta),
        int(max_iter),
        float(tol),
        float(fdr_q),
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "lr": np.asarray(res["lr"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.int64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "effect_size": np.asarray(res["effect_size"], dtype=np.float64),
    }


def u3_person_fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    cutoff: float | None = None,
) -> dict[str, np.ndarray]:
    """Nonparametric polytomous person-fit U3poly (compute in Rust; Emons, 2008),
    van der Flier's (1982) dichotomous U3 generalized to ordered polytomous items.
    It needs NO fitted IRT model: each item-step response function ``P(Y_i >= m)``
    is estimated by its sample proportion and turned into a logit weight, and a
    person's observed weighted score is compared to the largest and smallest
    weighted scores attainable at that person's total score (the conditioning
    group). Returns per-person ``u3poly`` in ``[0, 1]`` (0 = perfectly
    popularity-consistent, 1 = maximally aberrant; ``NaN`` where undefined),
    ``total_score`` (the summed ordinal score over observed items), and
    ``flagged`` (``u3poly >= cutoff``; all ``False`` when ``cutoff is None``).

    ``responses`` is persons x items of integer categories with ``NaN`` for
    missing (marginalized per person). Items must be keyed so a higher category
    means more of the trait -- recode reverse-keyed items first. U3poly has no
    reliable analytic null, so a critical value should come from
    :func:`u3_cutoff_polytomous` (a simulated reference), not a normal
    approximation; and because a single pooled cutoff cannot fully condition on
    the total score, treat flags near the score extremes cautiously.

    References (APA 7th ed.):
        Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous
            item scores. *Applied Psychological Measurement, 32*(3), 224-247.
            https://doi.org/10.1177/0146621607302479
        van der Flier, H. (1982). Deviant response patterns and comparability of
            test scores. *Journal of Cross-Cultural Psychology, 13*(3), 267-298.
            https://doi.org/10.1177/0022002182013003001
    """
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    n_persons, n_items = y_int.shape
    core = _core_module()
    if core is None or not hasattr(core, "u3_person_fit"):
        raise RuntimeError("u3_person_fit_polytomous requires the compiled Rust core")

    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.u3_person_fit(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        obs_arg,
        None if cutoff is None else float(cutoff),
    )
    return {
        "u3poly": np.asarray(res["u3poly"], dtype=np.float64),
        "total_score": np.asarray(res["total_score"], dtype=np.int64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
    }


def u3_cutoff_polytomous(
    fit: PolytomousFit,
    n_persons: int,
    alpha: float = 0.05,
    n_rep: int = 200,
    seed: int = 0,
) -> float:
    """Simulated ``1 - alpha`` critical value for :func:`u3_person_fit_polytomous`
    (compute in Rust; Emons, 2008, used simulated critical values). A parametric
    bootstrap: ``n_rep`` complete datasets of ``n_persons`` x (fitted item count)
    are generated from the fitted GRM/GPCM ``fit`` at ``theta ~ N(0, 1)``, and the
    empirical ``1 - alpha`` quantile of the pooled U3poly is returned. Because the
    null distribution depends on the latent distribution, this ``N(0, 1)`` cutoff
    is appropriate only when that population assumption is reasonable; for a skewed
    population, calibrate against a matching simulation. The replications are
    complete (full-length) patterns, so the cutoff is calibrated for complete
    responders only -- do not flag persons with substantial missing data against
    it (their U3poly comes from a shorter, coarser null).
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    n_persons = _bounded_integer(n_persons, "n_persons", 1, MAX_SIM_PERSONS)
    n_rep = _bounded_integer(n_rep, "n_rep", 1, MAX_POLY_BOOTSTRAP_REPLICATES)
    if not np.isfinite(alpha) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    if n_persons * n_items * n_rep > MAX_SIM_CELLS:
        raise ValueError("U3 bootstrap exceeds the aggregate work limit")
    core = _core_module()
    if core is None or not hasattr(core, "u3_bootstrap_cutoff"):
        raise RuntimeError("u3_cutoff_polytomous requires the compiled Rust core")
    return float(
        core.u3_bootstrap_cutoff(
            int(n_persons),
            int(n_items),
            int(n_cat),
            fit.slope.astype(np.float64),
            fit.cat_params.reshape(-1).astype(np.float64),
            fit.model,
            float(alpha),
            int(n_rep),
            int(seed),
        )
    )
