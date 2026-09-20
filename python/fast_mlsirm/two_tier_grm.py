"""Single-group polytomous two-tier graded response model (Cai, 2010;
Cai, Yang, & Hansen, 2011; Gibbons et al., 2007).

Each item's ordered categories load a caller-supplied subset of the
correlated primary dimensions plus at most one orthogonal specific factor.
Estimation is Bock-Aitkin marginal maximum likelihood (Cai et al., 2011,
"Maximum Marginal Likelihood Estimation" section) with dimension reduction
over the specific tier; the numerical work runs in Rust
(``mlsirm_core::two_tier_grm``).

Modelling decisions and their sources (every non-obvious choice is cited;
decisions without a paper source are marked as implementation choices):

- Cumulative-logit graded form ``P(Y >= k) = logistic(sum_p a_ip*theta_p +
  a_S*theta_S + d_k)`` with strictly decreasing boundary intercepts (Cai et
  al., 2011, eq. 6), adjacent-difference category probabilities (Cai et al.,
  2011, eq. 7). The logistic link is an implementation choice (Gibbons et
  al., 2007, eq. 9, use the normal ogive) matching the ``mirt`` graded
  comparison.
- Two-tier latent covariance ``Sigma = [[G, 0], [0, diag(S)]]``: primaries
  ``theta_P ~ MVN(0, Phi)`` with ``Phi`` a correlation matrix (unit
  diagonal, free off-diagonals — the single-group identification), specifics
  orthogonal ``N(0, 1)`` (Chalmers, 2026, mirt ``bfactor`` documentation,
  "Details" section, which cites Cai, 2010). The bifactor model is the
  special case of one primary dimension (same source). The two-tier model
  itself is Cai (2010) (abstract read; full text not accessible).
- Caller-supplied confirmatory primary pattern (fixed zeros are never
  estimated); rotation with correlated primaries is the caller's
  identification responsibility (implementation scope choice; Cai, 2010, is a
  confirmatory model). Specific-free items (``specific_map == -1``) allowed.
- Slopes UNCONSTRAINED on the real line so reverse-keyed items are
  representable (implementation choice extending the crate's #1879
  unconstrained-slope contract to the two-tier case).
- Reflection pinned per dimension by the crate's deterministic rule
  (largest-magnitude slope positive; ``poly::canonicalize_slope_reflection``,
  ``grm.rs``) — an implementation choice for reporting, not a paper
  prescription; primary flips jointly re-sign the ``Phi`` row/column (the
  joint flip leaves every likelihood invariant); thresholds are invariant
  under the joint flip.
- At least two loading items per primary and per specific factor required
  (implementation choice, not a paper prescription: no minimum-block-size
  theorem was found in the cited sources, and smaller blocks leave the
  primary correlation and the primary/specific split weakly identified).
- E-step integrates the primaries on a fixed product grid (density-ratio
  reweighting by ``Phi``) and each specific factor within its item block at
  fixed primary nodes, so integration needs only ``P + 1`` dimensions
  (Chalmers, 2026, mirt ``bfactor`` documentation: "requires only ncol(G) +
  1 dimensions for integration ... due to the dimension reduction
  technique"; the per-node factorization follows Gibbons et al., 2007,
  eq. 15). Gauss-Hermite quadrature on fixed grids; the EM node set never
  reparametrizes, so EM is monotone (Cai et al., 2011, "Maximum Marginal
  Likelihood Estimation" section).
- Primary correlations maximize the expected complete-data normal
  log-likelihood over Fisher-``z`` transforms (implementation choice using
  the crate's shared Newton ascent; the EM M-step principle is Cai et al.,
  2011, "Maximum Marginal Likelihood Estimation" section).
- ``seed`` drives ONLY the random-start jitter; quadrature is deterministic,
  so reruns with the same arguments bit-reproduce (implementation choice for
  the #1912 reproducibility requirement).
- Newton M-step depth (10 inner iterations) and ridge (1e-8, Hessian
  conditioning only, not a prior) are fixed implementation choices shared
  with the crate's GRM estimator, not caller arguments.
- Unobserved categories raise instead of imputing (with no observations in
  a category the adjacent boundary pair is unidentified under Cai et al.,
  2011, eq. 7). Missing cells are dropped under MAR (Cai et al., 2011,
  eq. 10). Non-convergence reports ``converged=False`` instead of
  substituting values (implementation choice for the #1912 fail-loud
  requirement).

References (APA 7th ed.):

    Cai, L. (2010). A two-tier full-information item factor analysis model
        with applications. *Psychometrika, 75*(4), 581-612.
        https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text
        not accessible — no equation locator is drawn from it)

    Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
        item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
        https://doi.org/10.1037/a0023350 (full text read)

    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
        Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
        Stover, A. (2007). Full-information item bifactor analysis of graded
        response data. *Applied Psychological Measurement, 31*(1), 4-19.
        https://doi.org/10.1177/0146621606289485 (full text read)

    Chalmers, R. P. (2026). mirt: Multidimensional item response theory
        (Version 1.46.1) [R package].
        https://cran.r-project.org/package=mirt (oracle software;
        ``bfactor`` help topic read)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_integer_control(value: object, name: str) -> int:
    """Normalize a trusted finite integer-valued scalar without callbacks."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite integer") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a finite integer")
    return int(numeric)


def _positive_real_control(value: object, name: str) -> float:
    """Normalize a trusted finite positive real scalar without callbacks."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and > 0") from None
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _u64_seed(value: object) -> int:
    """Normalize the deterministic start seed without callbacks."""

    if isinstance(value, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("seed must be a non-negative integer") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError("seed must be a non-negative integer")
    seed = int(numeric)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


@dataclass
class TwoTierGrmFit:
    """Fitted single-group polytomous two-tier GRM.

    ``a_primary`` the ``n_items x n_primary`` primary slopes (unconstrained,
    ``0`` at fixed pattern positions, reflection-canonicalized per primary);
    ``a_specific`` the ``n_items`` specific slopes (``0`` for specific-free
    items, canonicalized within each block); ``threshold`` the
    ``n_items x (n_cat-1)`` strictly decreasing boundary intercepts; ``phi``
    the ``n_primary x n_primary`` estimated primary correlation matrix (unit
    diagonal); ``theta_p_eap`` / ``theta_p_sd`` the primary-factor EAPs and
    marginal posterior SDs (``n_persons x n_primary``); ``category_counts``
    the observed ``n_items x n_cat`` counts. ``termination_reason`` is
    ``"tolerance_met"`` or ``"max_iter_reached"``; ``best_start`` the winning
    start in ``0..n_starts``.
    """

    a_primary: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    phi: np.ndarray
    theta_p_eap: np.ndarray
    theta_p_sd: np.ndarray
    category_counts: np.ndarray
    n_cat: int
    n_primary: int
    n_specific: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    best_start: int
    n_parameters: int


def fit_two_tier_grm(
    responses: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_primary: int,
    n_specific: int,
    q_primary: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    seed: int,
) -> TwoTierGrmFit:
    """Fit the single-group polytomous two-tier GRM (compute in Rust).

    ``responses`` is a persons x items integer-category array
    (``0..n_cat-1``; ``NaN`` or negative = missing, dropped MAR).
    ``primary_map`` is an ``n_items x n_primary`` boolean array marking the
    free confirmatory primary slopes (each primary needs at least two
    loading items); ``specific_map`` is a length-``n_items`` integer array
    with ``-1`` for specific-free items and ``0..n_specific-1`` otherwise
    (every specific factor needs at least two items).
    ``q_primary``/``q_specific`` are required Gauss-Hermite node counts
    (any ``int >= 1``; #1929 removed the fixed-table cap, so any node count
    the Rust core's arbitrary-``n`` Golub-Welsch quadrature resolves is
    accepted); no default is offered, because no accuracy target is on file
    to source one against (Project rule, issue #1929). Study settings use
    >= 121 nodes per dimension (chosen by precision convergence, e.g. 121
    vs 241 agreement).
    ``n_starts`` deterministic EM starts from ``seed`` keep the best loglik. Out-of-range caller arguments raise ``ValueError``
    (never clamped, and — per the no-magic-caps rule — upper-bounded only
    where a real constraint exists); unobserved categories raise;
    ``max_iter`` exhaustion returns ``converged=False`` instead of
    substituting values.

    See the module docstring for the model, the paper basis of every
    non-obvious decision, and the APA 7th references.
    """
    n_cat_int = _finite_integer_control(n_cat, "n_cat")
    if n_cat_int < 2:
        raise ValueError("n_cat must be >= 2")
    n_primary_int = _finite_integer_control(n_primary, "n_primary")
    if n_primary_int < 1:
        raise ValueError("n_primary must be >= 1")
    n_specific_int = _finite_integer_control(n_specific, "n_specific")
    if n_specific_int < 1:
        raise ValueError("n_specific must be >= 1")
    q_primary_int = _finite_integer_control(q_primary, "q_primary")
    if q_primary_int < 1:
        raise ValueError("q_primary must be >= 1")
    q_specific_int = _finite_integer_control(q_specific, "q_specific")
    if q_specific_int < 1:
        raise ValueError("q_specific must be >= 1")
    max_iter_int = _finite_integer_control(max_iter, "max_iter")
    if max_iter_int < 1:
        raise ValueError("max_iter must be >= 1")
    n_starts_int = _finite_integer_control(n_starts, "n_starts")
    if n_starts_int < 1:
        raise ValueError("n_starts must be >= 1")
    tol_float = _positive_real_control(tol, "tol")
    seed_int = _u64_seed(seed)

    y = np.asarray(responses)
    if np.iscomplexobj(y):
        raise ValueError("responses must be real-valued")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if y.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    y = y.astype(np.float64, copy=False)
    if np.isinf(y).any():
        raise ValueError("responses must not contain infinity")
    n_persons, n_items = y.shape

    pmap = np.asarray(primary_map)
    if pmap.dtype.kind not in ("b", "i", "u"):
        raise ValueError("primary_map must be a boolean array")
    if pmap.ndim != 2 or pmap.shape != (n_items, n_primary_int):
        raise ValueError("primary_map must be an n_items x n_primary boolean array")
    pmap_bool = np.asarray(pmap, dtype=bool)

    smap = np.asarray(specific_map)
    if smap.ndim != 1 or smap.shape[0] != n_items:
        raise ValueError("specific_map must be a 1-D array of length n_items")
    if smap.dtype.kind == "f":
        if not bool(np.isfinite(smap).all()):
            raise ValueError("specific_map entries must be finite integers")
        if bool((smap != np.floor(smap)).any()):
            raise ValueError("specific_map entries must be integers")
    try:
        smap_int = smap.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        raise ValueError("specific_map entries must be integers") from None
    if bool((smap_int < -1).any()) or bool((smap_int >= n_specific_int).any()):
        raise ValueError(
            "specific_map entries must be -1 (specific-free) or in "
            f"0..{n_specific_int - 1}"
        )

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_two_tier_grm"):
        raise RuntimeError("fit_two_tier_grm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_two_tier_grm(
        yy,
        observed.reshape(-1),
        pmap_bool.reshape(-1),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_primary_int),
        int(n_specific_int),
        int(n_cat_int),
        int(q_primary_int),
        int(q_specific_int),
        int(max_iter_int),
        float(tol_float),
        int(n_starts_int),
        int(seed_int),
    )
    return TwoTierGrmFit(
        a_primary=np.asarray(res["a_primary"], dtype=np.float64).reshape(
            n_items, n_primary_int
        ),
        a_specific=np.asarray(res["a_specific"], dtype=np.float64),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(
            n_items, n_cat_int - 1
        ),
        phi=np.asarray(res["phi"], dtype=np.float64).reshape(
            n_primary_int, n_primary_int
        ),
        theta_p_eap=np.asarray(res["theta_p_eap"], dtype=np.float64).reshape(
            n_persons, n_primary_int
        ),
        theta_p_sd=np.asarray(res["theta_p_sd"], dtype=np.float64).reshape(
            n_persons, n_primary_int
        ),
        category_counts=np.asarray(res["category_counts"], dtype=np.int64).reshape(
            n_items, n_cat_int
        ),
        n_cat=int(n_cat_int),
        n_primary=int(n_primary_int),
        n_specific=int(n_specific_int),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        best_start=int(res["best_start"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class TwoTierOakesSe:
    """Observed-information standard errors for the two-tier GRM.

    ``labels`` free-parameter names; ``information`` always present ``k x k``;
    ``vcov``/``se`` are ``None`` when the information is not positive
    definite (never substituted).

    Implementation basis: Oakes, D. (1999). Direct calculation of the
    information matrix via the EM algorithm. *Journal of the Royal
    Statistical Society Series B: Statistical Methodology, 61*(2), 479-482.
    https://doi.org/10.1111/1467-9868.00188 (eq. 6, p. 480); Cai, L., Yang,
    J. S., & Hansen, M. (2011). Generalized full-information item bifactor
    analysis. *Psychological Methods, 16*(3), 221-248.
    https://doi.org/10.1037/a0023350 (eq. 6, p. 227).
    """

    labels: list
    information: np.ndarray
    vcov: np.ndarray | None
    se: np.ndarray | None
    positive_definite: bool
    non_pd_reason: str | None


def two_tier_oakes_se(
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    phi: np.ndarray,
    responses: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_primary: int,
    n_specific: int,
    q_primary: int,
    q_specific: int,
    fd_step: float,
) -> TwoTierOakesSe:
    """Observed-information SEs via Oakes (1999, eq. 6, p. 480) at given
    two-tier parameters (valid at every point, not only the MLE).

    ``q_primary``/``q_specific``/``fd_step`` are REQUIRED (no defaults;
    ADR-0028 / #1929). Non-PD information returns ``se=None``.

    Implementation basis: Oakes (1999, eq. 6, p. 480); Cai et al. (2011,
    eq. 6, p. 227); Gibbons et al. (2007, eq. 15).
    """

    n_cat_int = _finite_integer_control(n_cat, "n_cat")
    if n_cat_int < 2:
        raise ValueError("n_cat must be >= 2")
    n_primary_int = _finite_integer_control(n_primary, "n_primary")
    if n_primary_int < 1:
        raise ValueError("n_primary must be >= 1")
    n_specific_int = _finite_integer_control(n_specific, "n_specific")
    if n_specific_int < 1:
        raise ValueError("n_specific must be >= 1")
    q_primary_int = _finite_integer_control(q_primary, "q_primary")
    if q_primary_int < 1:
        raise ValueError("q_primary must be >= 1")
    q_specific_int = _finite_integer_control(q_specific, "q_specific")
    if q_specific_int < 1:
        raise ValueError("q_specific must be >= 1")
    fd_float = _positive_real_control(fd_step, "fd_step")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    y = y.astype(np.float64, copy=False)
    n_persons, n_items = y.shape

    pmap = np.asarray(primary_map)
    if pmap.ndim != 2 or pmap.shape != (n_items, n_primary_int):
        raise ValueError("primary_map must be an n_items x n_primary boolean array")
    pmap_bool = np.asarray(pmap, dtype=bool)

    smap = np.asarray(specific_map)
    if smap.ndim != 1 or smap.shape[0] != n_items:
        raise ValueError("specific_map must be a 1-D array of length n_items")
    smap_int = smap.astype(np.int64, copy=False)

    ag = np.asarray(a_primary, dtype=np.float64)
    if ag.shape != (n_items, n_primary_int):
        raise ValueError("a_primary must have shape (n_items, n_primary)")
    as_ = np.asarray(a_specific, dtype=np.float64)
    if as_.shape != (n_items,):
        raise ValueError("a_specific must have length n_items")
    th = np.asarray(threshold, dtype=np.float64)
    if th.shape != (n_items, n_cat_int - 1):
        raise ValueError("threshold must have shape (n_items, n_cat - 1)")
    ph = np.asarray(phi, dtype=np.float64)
    if ph.shape != (n_primary_int, n_primary_int):
        raise ValueError("phi must have shape (n_primary, n_primary)")

    observed = np.isfinite(y) & (y >= 0)
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "two_tier_oakes_se"):
        raise RuntimeError("two_tier_oakes_se requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.two_tier_oakes_se(
        ag.reshape(-1),
        as_.reshape(-1),
        th.reshape(-1),
        ph.reshape(-1),
        yy,
        observed.reshape(-1),
        pmap_bool.reshape(-1),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_primary_int),
        int(n_specific_int),
        int(n_cat_int),
        int(q_primary_int),
        int(q_specific_int),
        float(fd_float),
    )
    labels = [str(v) for v in res["labels"]]
    k = len(labels)
    information = np.asarray(res["information"], dtype=np.float64).reshape(k, k)
    vcov_raw = res["vcov"]
    se_raw = res["se"]
    vcov = None if vcov_raw is None else np.asarray(vcov_raw, dtype=np.float64).reshape(k, k)
    se = None if se_raw is None else np.asarray(se_raw, dtype=np.float64)
    reason_raw = res["non_pd_reason"]
    return TwoTierOakesSe(
        labels=labels,
        information=information,
        vcov=vcov,
        se=se,
        positive_definite=bool(res["positive_definite"]),
        non_pd_reason=None if reason_raw is None else str(reason_raw),
    )


@dataclass(frozen=True)
class TwoTierExpectedTotalGivenPrimary:
    """Expected raw total ``E[T | theta_focal]`` after integrating nuisance traits.

    ``theta_focal`` is the caller grid (or per-person focal EAPs); ``expected_total``
    matches it elementwise. Nuisance traits (non-focal primaries with nonzero
    item loadings, plus each item's specific factor when present) are integrated
    as independent ``N(0, 1)`` Gauss-Hermite dimensions — matching the late-life
    G+4+W manuscript contract ``E[T|G]`` with W and S1..S4 as nuisances under
    the orthogonal mirt identification (not substituting ``theta_p_eap`` for
    raw scores, and not using ``Phi`` conditional ``W|G`` in this path).
    """

    theta_focal: np.ndarray
    expected_total: np.ndarray
    focal_primary: int
    q_nuisance: int
    n_items: int
    prior: str


def expected_total_score_two_tier_given_primary(
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    specific_map: np.ndarray,
    theta_focal: np.ndarray,
    *,
    focal_primary: int,
    q_nuisance: int,
) -> TwoTierExpectedTotalGivenPrimary:
    """Expected raw total given one two-tier primary, nuisances integrated out.

    Linearity of expectation: ``E[T|theta_f] = sum_i E[Y_i|theta_f]``. Each
    item's conditional expectation marginalizes independent standard-normal
    nuisances on a product Gauss-Hermite rule with ``q_nuisance`` nodes per
    nuisance dimension (required; no default — issue #1929).

    For adopted emotionality G+4+W: ``focal_primary=0`` (G); non-crossed items
    integrate one specific; wording-crossed items integrate ``(S_d, W)`` in 2-D
    (primary column 1). Reverse-keyed items are carried by unconstrained
    slopes already stored on the fit.

    Parameters
    ----------
    a_primary
        ``n_items x n_primary`` primary slopes (zeros at fixed pattern cells).
    a_specific
        Length-``n_items`` specific slopes (``0`` when specific-free).
    threshold
        ``n_items x (n_cat-1)`` GRM boundary intercepts.
    specific_map
        Length-``n_items`` map (``-1`` specific-free, else ``0..n_specific-1``).
    theta_focal
        Finite 1-D focal trait values (grid or person EAPs).
    focal_primary
        Column index of the conditioned primary (``0`` for G).
    q_nuisance
        Gauss-Hermite nodes per nuisance dimension in ``1..=4096``.
    """
    from .polytomous import (
        MAX_POLY_QUADRATURE_POINTS,
        PolytomousFit,
        _bounded_integer,
        predict_expected_response_polytomous,
    )

    ap = np.asarray(a_primary, dtype=np.float64)
    asp = np.asarray(a_specific, dtype=np.float64)
    th = np.asarray(threshold, dtype=np.float64)
    smap = np.asarray(specific_map, dtype=np.int64)
    grid = np.asarray(theta_focal, dtype=np.float64)

    if ap.ndim != 2 or ap.shape[0] == 0:
        raise ValueError("a_primary must be a non-empty n_items x n_primary array")
    n_items, n_primary = ap.shape
    if asp.shape != (n_items,):
        raise ValueError("a_specific must have shape (n_items,)")
    if smap.shape != (n_items,):
        raise ValueError("specific_map must have shape (n_items,)")
    if th.ndim != 2 or th.shape[0] != n_items:
        raise ValueError("threshold must be n_items x (n_cat-1)")
    if not np.all(np.isfinite(ap)) or not np.all(np.isfinite(asp)) or not np.all(
        np.isfinite(th)
    ):
        raise ValueError("a_primary, a_specific, and threshold must be finite")
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("theta_focal must be a non-empty 1-D array")
    if not np.all(np.isfinite(grid)):
        raise ValueError("theta_focal must be finite")

    focal = _bounded_integer(focal_primary, "focal_primary", 0, n_primary - 1)
    q = _bounded_integer(q_nuisance, "q_nuisance", 1, MAX_POLY_QUADRATURE_POINTS)

    nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(q)
    weights_1d = weights_1d / weights_1d.sum()
    unit_slope = np.ones(1, dtype=np.float64)
    expected_total = np.zeros(grid.size, dtype=np.float64)

    for item in range(n_items):
        a_f = float(ap[item, focal])
        # Independent N(0,1) nuisance loadings on this item (non-focal primaries + specific).
        nuisance_coefs: list[float] = []
        for p in range(n_primary):
            if p == focal:
                continue
            coef = float(ap[item, p])
            if coef != 0.0:
                nuisance_coefs.append(coef)
        if int(smap[item]) >= 0 and float(asp[item]) != 0.0:
            nuisance_coefs.append(float(asp[item]))

        cell = PolytomousFit(
            model="grm",
            slope=unit_slope,
            cat_params=th[item : item + 1],
            loglik=float("nan"),
            n_iter=0,
            converged=True,
            termination_reason="marginalized",
        )

        if not nuisance_coefs:
            base = a_f * grid
            expected_total += predict_expected_response_polytomous(
                cell, base.reshape(-1)
            ).ravel()
            continue

        n_nuis = len(nuisance_coefs)
        # Product Gauss-Hermite over nuisance_coefs (1-D or 2-D for G+4+W crossed).
        if n_nuis == 1:
            mesh = nodes_1d.reshape(1, -1)
            wmesh = weights_1d.copy()
        else:
            grids = np.meshgrid(*([nodes_1d] * n_nuis), indexing="ij")
            mesh = np.stack([g.ravel() for g in grids], axis=0)  # (n_nuis, n_nodes)
            w_grids = np.meshgrid(*([weights_1d] * n_nuis), indexing="ij")
            wmesh = np.prod([w.ravel() for w in w_grids], axis=0)
            wmesh = wmesh / wmesh.sum()

        # base[g, node] = a_f * G[g] + sum_k a_k * z_k[node]
        offset = np.zeros(mesh.shape[1], dtype=np.float64)
        for k, coef in enumerate(nuisance_coefs):
            offset += coef * mesh[k]
        base = a_f * grid[:, None] + offset[None, :]
        expected = predict_expected_response_polytomous(cell, base.reshape(-1))
        expected_total += (
            expected.reshape(base.shape) * wmesh[None, :]
        ).sum(axis=1)

    return TwoTierExpectedTotalGivenPrimary(
        theta_focal=grid.copy(),
        expected_total=expected_total,
        focal_primary=int(focal),
        q_nuisance=int(q),
        n_items=int(n_items),
        prior="independent_N01_nuisance_product_GH",
    )


def expected_total_score_two_tier_from_fit(
    fit: TwoTierGrmFit,
    theta_focal: np.ndarray,
    *,
    focal_primary: int,
    q_nuisance: int,
    specific_map: np.ndarray,
) -> TwoTierExpectedTotalGivenPrimary:
    """Wrapper over :func:`expected_total_score_two_tier_given_primary` for a fit.

    ``specific_map`` is required because :class:`TwoTierGrmFit` does not store
    the confirmatory specific assignment used at fit time.
    """
    return expected_total_score_two_tier_given_primary(
        fit.a_primary,
        fit.a_specific,
        fit.threshold,
        specific_map,
        theta_focal,
        focal_primary=focal_primary,
        q_nuisance=q_nuisance,
    )
