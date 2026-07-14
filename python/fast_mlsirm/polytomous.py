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

from dataclasses import dataclass

import numpy as np

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
    yf = np.asarray(responses, dtype=np.float64)
    if yf.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    observed = np.isfinite(yf)
    obs_vals = yf[observed]
    if obs_vals.size and (
        np.any(obs_vals != np.floor(obs_vals)) or obs_vals.min() < 0 or obs_vals.max() >= n_cat
    ):
        raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")
    y_int = np.where(observed, yf, 0.0).astype(np.int64)
    return y_int, observed


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
    ``theta ~ N(0, 1)`` on a ``q_theta``-node Gauss-Hermite grid.
    """
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if not isinstance(n_cat, int) or n_cat < 2:
        raise ValueError("n_cat must be an integer >= 2")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")

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
    missing response. Returns ``{"theta_eap", "theta_sd"}``.
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
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
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
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
    ``{"item_info"` (n_theta x n_items), ``"test_info"`` (n_theta)}``.
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size == 0 or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a non-empty finite 1-D grid")
    core = _core_module()
    if core is None or not hasattr(core, "poly_information_curves"):
        raise RuntimeError("information_polytomous requires the compiled Rust core")

    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    flat = core.poly_information_curves(
        th,
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        int(n_items),
        int(n_cat),
        fit.model,
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
    """
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if not isinstance(n_cat, int) or n_cat < 2:
        raise ValueError("n_cat must be an integer >= 2")
    if not isinstance(latent_dim, int) or not (1 <= latent_dim <= 3):
        raise ValueError("latent_dim must be an integer in 1..3")
    if q_theta not in {7, 11, 15, 21, 31, 41} or q_xi not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta/q_xi must be one of 7, 11, 15, 21, 31, 41")

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
    if min_expected <= 0:
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
    interval (``rmsea2_ci_lower``/``rmsea2_ci_upper``), ``srmsr``, and the
    ``n_moments``/``n_parameters``/``n_complete`` counts. Requires at least 3
    items and ``n_moments > n_parameters``.

    References (APA 7th ed.):
        Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in
            categorical data analysis. *Multivariate Behavioral Research,
            49*(4), 305-328. https://doi.org/10.1080/00273171.2014.911075
    """
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

    References (APA 7th ed.):
        Bock, R. D. (1972). Estimating item parameters and latent ability when
            responses are scored in two or more nominal categories.
            *Psychometrika, 37*(1), 29-51. https://doi.org/10.1007/BF02291411
        Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item
            response model. In *Handbook of polytomous item response theory
            models* (pp. 43-75). Routledge.
    """
    if not isinstance(n_cat, int) or n_cat < 2:
        raise ValueError("n_cat must be an integer >= 2")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")

    y_int, observed = _poly_int_and_mask(responses, n_cat)
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
    if se_threshold < 0 or min_items < 1 or max_items < min_items:
        raise ValueError("require se_threshold >= 0 and 1 <= min_items <= max_items")

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
