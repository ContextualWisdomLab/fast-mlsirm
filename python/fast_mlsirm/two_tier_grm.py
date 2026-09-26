"""Single-group polytomous two-tier graded response model (Cai, 2010;
Cai, Yang, & Hansen, 2011; Gibbons et al., 2007).

Each item's ordered categories load a caller-supplied subset of the
primary dimensions plus at most one orthogonal specific factor.
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
  diagonal, optionally fixed to identity), specifics
  orthogonal ``N(0, 1)`` (Chalmers, 2026, mirt ``bfactor`` documentation,
  "Details" section, which cites Cai, 2010). The bifactor model is the
  special case of one primary dimension (same source). The two-tier model
  itself is Cai (2010, pp. 583-584, full text read). Fixing ``Phi = I`` is
  an orthogonal-primary restriction of its covariance structure.
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
        https://doi.org/10.1007/s11336-010-9178-0 (full text read)

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
    the ``n_primary x n_primary`` primary correlation matrix (unit
    diagonal); ``theta_p_eap`` / ``theta_p_sd`` the primary-factor EAPs and
    marginal posterior SDs (``n_persons x n_primary``); ``category_counts``
    the observed ``n_items x n_cat`` counts. ``termination_reason`` is
    ``"tolerance_met"`` or ``"max_iter_reached"``; ``best_start`` the winning
    start in ``0..n_starts``. ``primary_identification`` is
    ``"orthogonal"`` when Phi was fixed to I or ``"correlated"`` when
    its off-diagonal entries were estimated (Cai, 2010, pp. 583-584).
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
    primary_identification: str


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
    primary_correlation: str = "estimate",
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
    ``primary_correlation='estimate'`` preserves the existing correlated-primary
    fit; ``'identity'`` fixes Phi exactly to I (Cai, 2010, pp. 583-584).
    In identity mode, distinct free-loading item sets for each primary pair
    are necessary to rule out continuous orthogonal rotations; per-column
    reflection canonicalization handles the remaining sign ambiguity.

    References (APA 7th ed.): Cai, L. (2010). A two-tier full-information item
    factor analysis model with applications. *Psychometrika, 75*(4), 581-612.
    https://doi.org/10.1007/s11336-010-9178-0; Cai, L., Yang, J. S., &
    Hansen, M. (2011). Generalized full-information item bifactor analysis.
    *Psychological Methods, 16*(3), 221-248. https://doi.org/10.1037/a0023350.
    """
    if not isinstance(primary_correlation, str) or primary_correlation not in (
        "estimate", "identity"
    ):
        raise ValueError("primary_correlation must be 'estimate' or 'identity'")
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
    if primary_correlation == "identity":
        for d in range(n_primary_int):
            for other in range(d + 1, n_primary_int):
                if np.array_equal(pmap_bool[:, d], pmap_bool[:, other]):
                    raise ValueError(
                        "identity primary correlation requires distinct free-loading item sets "
                        "for every pair of primary columns; identical free-loading item sets "
                        f"at columns {d} and {other} admit orthogonal rotation"
                    )

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
        primary_correlation,
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
        primary_identification=str(res["primary_identification"]),
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
    primary_correlation: str = "estimate",
) -> TwoTierOakesSe:
    """Observed-information SEs via Oakes (1999, eq. 6, p. 480) at given
    two-tier parameters (valid at every point, not only the MLE).

    ``q_primary``/``q_specific``/``fd_step`` are REQUIRED (no defaults;
    ADR-0028 / #1929). Non-PD information returns ``se=None``.

    Implementation basis: Oakes (1999, eq. 6, p. 480); Cai et al. (2011,
    eq. 6, p. 227); Gibbons et al. (2007, eq. 15).
    ``primary_correlation='identity'`` excludes fixed Phi from the information
    matrix (Cai, 2010, pp. 583-584; Oakes, 1999, eq. 6, p. 480).

    References (APA 7th ed.): Cai, L. (2010). A two-tier full-information item
    factor analysis model with applications. *Psychometrika, 75*(4), 581-612.
    https://doi.org/10.1007/s11336-010-9178-0; Oakes, D. (1999). Direct
    calculation of the information matrix via the EM algorithm. *Journal of
    the Royal Statistical Society: Series B, 61*(2), 479-482.
    https://doi.org/10.1111/1467-9868.00188.
    """
    if not isinstance(primary_correlation, str) or primary_correlation not in (
        "estimate", "identity"
    ):
        raise ValueError("primary_correlation must be 'estimate' or 'identity'")

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
        primary_correlation,
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
    matches it elementwise.

    **Prior contract (explicit, fail-closed).** Non-focal primaries and specifics
    are integrated as *independent* Gaussians under
    ``nuisance_prior="independent_standardized"``, each with its own reference
    mean/sd vector (``primary_ref_*`` / ``specific_ref_*``). This path does
    **not** implement ``Phi``-conditional nuisance. ``Phi == I`` on a fit is a
    necessary numeric gate only — not proof that the fit was estimated under
    orthogonal identification; :func:`expected_total_score_two_tier_from_fit`
    additionally requires an explicit consumer confirmation flag.

    Example (not a universal library contract): a general + method-factor
    two-primary model with specifics, fitted under orthogonal identification,
    uses ``focal_primary=0`` (general) with the method factor and specifics
    as independent reference nuisances.
    """

    theta_focal: np.ndarray
    expected_total: np.ndarray
    focal_primary: int
    q_nuisance: int
    n_items: int
    prior: str
    primary_ref_mean: np.ndarray
    primary_ref_sd: np.ndarray
    specific_ref_mean: np.ndarray
    specific_ref_sd: np.ndarray


def _require_identity_phi(phi: np.ndarray, *, atol: float = 0.0) -> None:
    """Fail closed unless ``phi`` is exactly the identity (unit diagonal, zero off)."""
    p = np.asarray(phi, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("phi must be a square primary correlation matrix")
    eye = np.eye(p.shape[0], dtype=np.float64)
    if not np.allclose(p, eye, atol=atol, rtol=0.0):
        raise ValueError(
            "expected_total_score_two_tier_from_fit requires Phi == I as a "
            "necessary numeric gate for orthogonal primary identification. "
            "Correlated Phi needs conditional nuisance integration (not "
            "implemented); do not pass estimated Phi≈I as a substitute."
        )


def _as_ref_mean(value: object, n: int, name: str) -> np.ndarray:
    """Scalar broadcasts; array must be shape ``(n,)`` and finite."""
    if n < 0:
        raise ValueError(f"{name}: n must be >= 0")
    if n == 0:
        arr = np.asarray(value, dtype=np.float64)
        if np.ndim(arr) == 0:
            return np.zeros(0, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size != 0:
            raise ValueError(f"{name} must be empty or scalar when n=0")
        return np.zeros(0, dtype=np.float64)
    if np.isscalar(value) or (isinstance(value, np.ndarray) and np.ndim(value) == 0):
        out = np.full(n, float(value), dtype=np.float64)  # type: ignore[arg-type]
    else:
        out = np.asarray(value, dtype=np.float64)
        if out.shape != (n,):
            raise ValueError(f"{name} must be scalar or shape ({n},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite (NaN/Inf rejected)")
    return out


def _as_ref_sd(value: object, n: int, name: str) -> np.ndarray:
    """Like :func:`_as_ref_mean` but every entry must be finite and ``> 0``."""
    out = _as_ref_mean(value, n, name)
    if out.size and np.any(out <= 0.0):
        raise ValueError(f"{name} entries must be > 0")
    return out


def _as_specific_map_int64(
    specific_map: object,
    *,
    n_items: int,
    n_specific: int | None = None,
) -> np.ndarray:
    """Validate ``specific_map`` before ``int64`` cast (no silent truncation/wrap).

    Rejects non-finite floats and non-integral values such as ``0.5`` before
    ``astype(np.int64)``. Also rejects unsigned / oversized values that cannot
    be represented in ``int64`` without wraparound (e.g. ``uint64`` max → ``-1``,
    ``2**63`` → ``int64`` min). Entries must be ``-1`` (specific-free) or
    integers in ``0..2**63-1``; when ``n_specific`` is given, also
    ``< n_specific``.
    """
    smap = np.asarray(specific_map)
    if smap.ndim != 1 or smap.shape[0] != n_items:
        raise ValueError("specific_map must be a 1-D array of length n_items")
    if smap.dtype.kind == "f":
        if not bool(np.isfinite(smap).all()):
            raise ValueError("specific_map entries must be finite integers")
        if bool((smap != np.floor(smap)).any()):
            raise ValueError("specific_map entries must be integers")
    elif smap.dtype.kind not in ("b", "i", "u"):
        raise ValueError("specific_map entries must be integers")

    # Pre-cast range check via Python int (no dtype wraparound).
    i64_max = int(np.iinfo(np.int64).max)
    for x in smap.ravel():
        try:
            iv = int(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("specific_map entries must be integers") from exc
        if iv < -1 or iv > i64_max:
            raise ValueError(
                "specific_map entries must be -1 (specific-free) or integers "
                f"in 0..{i64_max} without wrapping into int64"
            )
        if n_specific is not None and iv >= int(n_specific):
            raise ValueError(
                "specific_map entries must be -1 (specific-free) or in "
                f"0..{int(n_specific) - 1}"
            )

    try:
        smap_int = smap.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("specific_map entries must be integers") from exc
    # Post-cast sentinel/range (defensive; pre-cast already enforced).
    if bool((smap_int < -1).any()):
        raise ValueError(
            "specific_map entries must be -1 (specific-free) or >= 0"
        )
    if n_specific is not None and bool((smap_int >= int(n_specific)).any()):
        raise ValueError(
            "specific_map entries must be -1 (specific-free) or in "
            f"0..{int(n_specific) - 1}"
        )
    return smap_int


def _probabilists_gauss_hermite(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite rule for any ``q >= 1`` via Golub & Welsch.

    Nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix of the
    ``He_n`` recurrence (``alpha_k = 0``, ``beta_k = k``); weights are the
    squared first eigenvector components, normalized to sum to one (Golub &
    Welsch, 1969, eq. 2.1-2.2, pp. 222-223; Section 3, p. 225). Unlike the
    closed-form weight formula in ``numpy.polynomial.hermite_e.hermegauss``,
    which returns NaN weights from about ``q = 481`` because ``1 / He_{n-1}^2``
    overflows, tail weights here underflow to ``0.0`` and never become NaN.
    This mirrors ``crates/mlsirm-core/src/quadrature.rs``
    ``gauss_hermite_probabilists`` (#1929).

    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature
    rules. *Mathematics of Computation, 23*(106), 221-230.
    https://doi.org/10.1090/S0025-5718-69-99647-1
    """
    if q == 1:
        return np.zeros(1), np.ones(1)
    # Allocation guard instead of a node-count cap (cf. the Rust core's
    # checked_mul guard, #1929): the dense q x q float64 Jacobi matrix must be
    # representable, and allocation/eigensolver failures fail loudly.
    itemsize = np.dtype(np.float64).itemsize
    if q > int(np.iinfo(np.intp).max) // (q * itemsize):
        raise ValueError(
            f"q_nuisance={q} needs a {q}x{q} Jacobi matrix that is not "
            "representable on this platform"
        )
    try:
        off = np.sqrt(np.arange(1, q, dtype=np.float64))
        jacobi = np.diag(off, 1) + np.diag(off, -1)
        nodes, vectors = np.linalg.eigh(jacobi)
    except (MemoryError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"q_nuisance={q}: Gauss-Hermite rule construction failed "
            f"({type(exc).__name__})"
        ) from exc
    weights = vectors[0, :] ** 2
    weights = weights / weights.sum()
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        raise ValueError(f"Gauss-Hermite rule with q={q} is not finite")
    return nodes, weights


def expected_total_score_two_tier_given_primary(
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    specific_map: np.ndarray,
    theta_focal: np.ndarray,
    *,
    focal_primary: int,
    q_nuisance: int,
    nuisance_prior: str,
    primary_ref_mean: object = 0.0,
    primary_ref_sd: object = 1.0,
    specific_ref_mean: object = 0.0,
    specific_ref_sd: object = 1.0,
) -> TwoTierExpectedTotalGivenPrimary:
    """Expected raw total given one two-tier primary, nuisances integrated out.

    Linearity of expectation: ``E[T|theta_f] = sum_i E[Y_i|theta_f]``. Each
    item's conditional expectation marginalizes independent Gaussian nuisances.
    Because those nuisances enter only through the linear predictor
    ``L = sum_k a_k Z_k`` and independent Gaussians yield
    ``L ~ N(sum a_k mu_k, sum (a_k sigma_k)^2)``, the product rule collapses to
    a single 1-D Gauss-Hermite integral with ``q_nuisance`` nodes (required; no
    default — issue #1929). This avoids ``q^n`` meshgrid allocation. Any exact
    integer ``q_nuisance >= 1`` is accepted with no node-count cap (maintainer
    steering 2026-09-16/17, item 1). Resource safety comes from allocation
    guards instead, mirroring the Rust core's ``checked_mul`` guard: the
    prediction-cell admission check runs before the rule is built, the dense
    ``q x q`` Jacobi matrix must be representable, and an allocation or
    eigensolver failure raises ``ValueError``.

    Reference distributions are **per primary / per specific dimension**. A
    scalar mean/sd broadcasts identical values across dimensions (producer
    contracts that fix every nuisance to N(0,1) may pass scalars and should
    record that basis). Distinct W vs S reference variances must pass arrays;
    bundling unequal variances into one scalar changes the integral.

    Example (general + method factor + specifics, orthogonal ID):
    ``focal_primary=0`` (general); items without a method-factor loading
    integrate one specific; method-loaded items integrate ``(S_d, W)`` jointly. Reverse keys use unconstrained (possibly negative)
    slopes. This example is not a universal contract for all two-tier fits.

    Parameters
    ----------
    nuisance_prior
        Must be ``"independent_standardized"``.
    primary_ref_mean, primary_ref_sd
        Length-``n_primary`` (or scalar broadcast). Focal slot is unused.
    specific_ref_mean, specific_ref_sd
        Length-``n_specific`` (or scalar broadcast), indexed by ``specific_map``.
    """
    from .polytomous import (
        _NUMPY_INTEGER_SCALAR_TYPES,
        PolytomousFit,
        _bounded_integer,
        _is_exact_type,
        predict_expected_response_polytomous,
    )
    from ._polytomous_prediction_admission import _raise_if_oversized_prediction_grid

    if nuisance_prior != "independent_standardized":
        raise ValueError(
            "nuisance_prior must be 'independent_standardized' (explicit orthogonal "
            "nuisance contract); Phi-conditional nuisance is not implemented"
        )

    ap = np.asarray(a_primary, dtype=np.float64)
    asp = np.asarray(a_specific, dtype=np.float64)
    th = np.asarray(threshold, dtype=np.float64)
    grid = np.asarray(theta_focal, dtype=np.float64)

    if ap.ndim != 2 or ap.shape[0] == 0:
        raise ValueError("a_primary must be a non-empty n_items x n_primary array")
    n_items, n_primary = ap.shape
    if asp.shape != (n_items,):
        raise ValueError("a_specific must have shape (n_items,)")
    smap = _as_specific_map_int64(specific_map, n_items=n_items)
    if th.ndim != 2 or th.shape[0] != n_items or th.shape[1] < 1:
        raise ValueError("threshold must be n_items x (n_cat-1) with n_cat>=2")
    if not np.all(np.isfinite(ap)) or not np.all(np.isfinite(asp)) or not np.all(
        np.isfinite(th)
    ):
        raise ValueError("a_primary, a_specific, and threshold must be finite")
    if np.any(np.diff(th, axis=1) >= 0.0):
        raise ValueError(
            "threshold rows must be strictly decreasing (GRM category support)"
        )
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("theta_focal must be a non-empty 1-D array")
    if not np.all(np.isfinite(grid)):
        raise ValueError("theta_focal must be finite")

    focal = _bounded_integer(focal_primary, "focal_primary", 0, n_primary - 1)
    # Exact integer >= 1, no node-count cap (maintainer steering item 1).
    q_type = type(q_nuisance)
    if q_type is int:
        q = q_nuisance
    elif _is_exact_type(q_type, _NUMPY_INTEGER_SCALAR_TYPES):
        q = int(q_nuisance)
    else:
        raise ValueError("q_nuisance must be an integer >= 1")
    if q < 1:
        raise ValueError("q_nuisance must be an integer >= 1")

    # When n_specific is not supplied by the fit wrapper, derive from the map.
    # Fail closed on huge representable indices that would allocate max(map)+1
    # reference vectors (confirmatory: at most one distinct specific id per item
    # ⇒ derived n_specific cannot exceed n_items).
    n_specific = int(smap.max()) + 1 if np.any(smap >= 0) else 0
    if n_specific > n_items:
        raise ValueError(
            "specific_map implies n_specific="
            f"{n_specific} > n_items={n_items}; pass a dense 0..K-1 map "
            "or use from_fit (which supplies fit.n_specific)"
        )
    p_mean = _as_ref_mean(primary_ref_mean, n_primary, "primary_ref_mean")
    p_sd = _as_ref_sd(primary_ref_sd, n_primary, "primary_ref_sd")
    s_mean = _as_ref_mean(specific_ref_mean, n_specific, "specific_ref_mean")
    s_sd = _as_ref_sd(specific_ref_sd, n_specific, "specific_ref_sd")

    # Admission before any O(q) / O(q^2) allocation (was after the rule).
    _raise_if_oversized_prediction_grid(
        int(grid.size) * int(q) * (int(th.shape[1]) + 1)
    )
    unit_nodes, unit_weights = _probabilists_gauss_hermite(q)
    unit_slope = np.ones(1, dtype=np.float64)
    expected_total = np.zeros(grid.size, dtype=np.float64)

    # Independent Gaussian nuisances enter only through the linear predictor
    # L = sum_k a_k Z_k. For independent Z_k ~ N(mu_k, sigma_k^2),
    # L ~ N(sum a_k mu_k, sum (a_k sigma_k)^2), so the product GH meshgrid
    # collapses to a single 1-D GH axis (exact continuous integral; finite-q
    # product vs collapse agree to ~1e-10 at q=21 in measured fixtures).
    for item in range(n_items):
        a_f = float(ap[item, focal])
        nuisance: list[tuple[float, float, float]] = []
        for p in range(n_primary):
            if p == focal:
                continue
            coef = float(ap[item, p])
            if coef != 0.0:
                nuisance.append((coef, float(p_mean[p]), float(p_sd[p])))
        sid = int(smap[item])
        if sid >= 0 and float(asp[item]) != 0.0:
            nuisance.append(
                (float(asp[item]), float(s_mean[sid]), float(s_sd[sid]))
            )

        cell = PolytomousFit(
            model="grm",
            slope=unit_slope,
            cat_params=th[item : item + 1],
            loglik=float("nan"),
            n_iter=0,
            converged=True,
            termination_reason="marginalized",
        )

        if not nuisance:
            base = a_f * grid
            expected_total += predict_expected_response_polytomous(
                cell, base.reshape(-1)
            ).ravel()
            continue

        mu_L = 0.0
        var_L = 0.0
        for coef, mu, sigma in nuisance:
            mu_L += coef * mu
            var_L += (coef * sigma) ** 2
        sd_L = float(np.sqrt(var_L))
        nodes = mu_L + sd_L * unit_nodes
        base = a_f * grid[:, None] + nodes[None, :]
        expected = predict_expected_response_polytomous(cell, base.reshape(-1))
        expected_total += (
            expected.reshape(base.shape) * unit_weights[None, :]
        ).sum(axis=1)

    return TwoTierExpectedTotalGivenPrimary(
        theta_focal=grid.copy(),
        expected_total=expected_total,
        focal_primary=int(focal),
        q_nuisance=int(q),
        n_items=int(n_items),
        prior="independent_standardized",
        primary_ref_mean=p_mean.copy(),
        primary_ref_sd=p_sd.copy(),
        specific_ref_mean=s_mean.copy(),
        specific_ref_sd=s_sd.copy(),
    )


def expected_total_score_two_tier_from_fit(
    fit: TwoTierGrmFit,
    theta_focal: np.ndarray,
    *,
    focal_primary: int,
    q_nuisance: int,
    specific_map: np.ndarray,
    orthogonal_primary_identification: bool,
    primary_ref_mean: object = 0.0,
    primary_ref_sd: object = 1.0,
    specific_ref_mean: object = 0.0,
    specific_ref_sd: object = 1.0,
) -> TwoTierExpectedTotalGivenPrimary:
    """Fit wrapper with dual gates: ``Phi == I`` and consumer ID confirmation.

    ``fit.phi == I`` is necessary but **not sufficient** evidence that the fit
    was estimated under orthogonal primary identification (a numeric matrix can
    be identity for other reasons). Callers must pass
    ``orthogonal_primary_identification=True`` only when the consuming research
    / estimation contract itself fixes orthogonal primaries. ``TwoTierGrmFit``
    does not yet carry identification metadata; this flag is the explicit
    consumer confirmation until such metadata exists.
    """
    if orthogonal_primary_identification is not True:
        raise ValueError(
            "orthogonal_primary_identification must be True: Phi==I alone does "
            "not prove the fit used orthogonal primary identification; the "
            "consumer must confirm that research/estimation contract"
        )
    if int(fit.n_primary) != int(np.asarray(fit.a_primary).shape[1]):
        raise ValueError("fit.n_primary inconsistent with a_primary shape")
    n_items = int(np.asarray(fit.a_primary).shape[0])
    smap = _as_specific_map_int64(
        specific_map, n_items=n_items, n_specific=int(fit.n_specific)
    )
    n_specific = int(smap.max()) + 1 if np.any(smap >= 0) else 0
    if n_specific != int(fit.n_specific):
        raise ValueError(
            f"specific_map implies n_specific={n_specific} but fit.n_specific="
            f"{fit.n_specific}"
        )
    _require_identity_phi(fit.phi, atol=0.0)
    return expected_total_score_two_tier_given_primary(
        fit.a_primary,
        fit.a_specific,
        fit.threshold,
        smap,
        theta_focal,
        focal_primary=focal_primary,
        q_nuisance=q_nuisance,
        nuisance_prior="independent_standardized",
        primary_ref_mean=primary_ref_mean,
        primary_ref_sd=primary_ref_sd,
        specific_ref_mean=specific_ref_mean,
        specific_ref_sd=specific_ref_sd,
    )
