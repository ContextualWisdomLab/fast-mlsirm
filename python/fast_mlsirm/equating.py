"""Observed-score equating (Kolen & Brennan, 2014): the raw-score complement to
the IRT scale linking in :mod:`fast_mlsirm.linking`. Equivalent-groups mean /
linear / equipercentile equating and NEAT chained / frequency-estimation
equating, all computed in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EG_METHOD_ALIASES = frozenset(
    {"mean", "m", "linear", "lin", "l", "equipercentile", "equip", "ep"}
)


def _require_equating_method(method, *, name: str = "method") -> str:
    """Return a trusted equating method identity without caller-controlled str().

    Hostile objects can implement ``__str__`` / ``__repr__`` for side effects or
    denial-of-service; accept only exact ``str`` identities so validation never
    executes those callbacks before the Rust core is reached.
    """
    if type(method) is not str:
        raise ValueError(f"{name} must be a str method identity")
    normalized = method.lower().replace("-", "").replace("_", "")
    if normalized not in _EG_METHOD_ALIASES:
        raise ValueError(
            f"{name} must be one of mean, linear, or equipercentile"
        )
    return method


def _require_optional_score_ceiling(value, name: str) -> int | None:
    """Return a trusted optional score ceiling without caller-controlled int().

    Accept plain ``int`` and NumPy integer scalars only. Arbitrary objects that
    implement ``__int__`` / ``__index__`` are rejected before conversion so
    validation cannot be hijacked by hostile callables.
    """
    if value is None:
        return None
    if type(value) is int:
        normalized = value
    elif isinstance(value, np.integer):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer score ceiling")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer score ceiling")
    return normalized


@dataclass
class EquateResult:
    """Observed-score equating result: the conversion table
    ``y_equivalents[i] = e_Y(x_scores[i])`` (unrounded), the form moments in
    ``moments`` (``mu_x``/``sigma_x``/``mu_y``/``sigma_y``/``mu_eq``/``sigma_eq``),
    and (for the moment methods) the linear ``slope``/``intercept`` (``NaN`` for
    equipercentile / NEAT). For frequency estimation the ``mu_x``/``sigma_x``/
    ``mu_y``/``sigma_y`` are the *synthetic-population* moments (the densities FE
    actually equates), not the raw form marginals, so they are not directly
    comparable to a chained or EG result's moments."""

    x_scores: np.ndarray
    y_equivalents: np.ndarray
    method: str
    design: str  # "EG" or "NEAT"
    moments: dict[str, float]  # mu_x, sigma_x, mu_y, sigma_y, mu_eq, sigma_eq
    slope: float
    intercept: float
    n_x: int
    n_y: int
    h_x: float = float("nan")  # Gaussian-kernel bandwidths (NaN unless kernel)
    h_y: float = float("nan")


def _infer_k(scores: np.ndarray, k, name: str) -> int:
    """Return the maximum score ``k``, inferring it from the data when unset.

    An explicit ``k`` is preferred: inferring the ceiling from observed scores
    under-counts it when the top score was never earned.
    """
    k = _require_optional_score_ceiling(k, name)
    if k is not None:
        return k
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name}: score vector must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: scores must be finite")
    # Inferring the maximum score from the observed data under-counts the true
    # ceiling when the top score was never earned, which shifts the whole
    # percentile-rank scale; pass an explicit k for anything but exploratory use.
    return int(np.round(arr.max()))


def _build(res, method: str, design: str) -> EquateResult:
    """Build an :class:`EquateResult` from a Rust equating result dict."""
    return EquateResult(
        x_scores=np.asarray(res["x_scores"], dtype=np.float64),
        y_equivalents=np.asarray(res["y_equivalents"], dtype=np.float64),
        method=method,
        design=design,
        moments={
            "mu_x": float(res["mu_x"]), "sigma_x": float(res["sigma_x"]),
            "mu_y": float(res["mu_y"]), "sigma_y": float(res["sigma_y"]),
            "mu_eq": float(res["mu_eq"]), "sigma_eq": float(res["sigma_eq"]),
        },
        slope=float(res["slope"]),
        intercept=float(res["intercept"]),
        n_x=int(res["n_x"]),
        n_y=int(res["n_y"]),
        h_x=float(res.get("h_x", float("nan"))),
        h_y=float(res.get("h_y", float("nan"))),
    )


def equate_observed_scores(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    method: str = "equipercentile",
    k_x: int | None = None,
    k_y: int | None = None,
) -> EquateResult:
    """Equivalent-groups (or single-group) observed-score equating of form X onto
    form Y (compute in Rust; Kolen & Brennan, 2014). ``x_scores``/``y_scores`` are
    raw integer total-score vectors from the two groups. ``method`` is
    ``"mean"``, ``"linear"``, or ``"equipercentile"`` (the default; whole-
    distribution matching via Kolen-Brennan uniform-kernel continuization).
    ``k_x``/``k_y`` are the maximum possible scores (number of items); if omitted
    they are inferred from the largest observed score, which is only safe when the
    top score was actually earned -- pass them explicitly otherwise. Returns an
    :class:`EquateResult` whose ``y_equivalents`` is the unrounded conversion
    table for scores ``0..k_x``.

    References (APA 7th ed.):
        Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
            linking: Methods and practices* (3rd ed.). Springer.
            https://doi.org/10.1007/978-1-4939-0317-7
    """
    from .fitstats import _core_module

    method = _require_equating_method(method)
    k_x = _require_optional_score_ceiling(k_x, "k_x")
    k_y = _require_optional_score_ceiling(k_y, "k_y")
    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores"):
        raise RuntimeError("equate_observed_scores requires the compiled Rust core")
    xs = np.asarray(x_scores, dtype=np.float64).ravel()
    ys = np.asarray(y_scores, dtype=np.float64).ravel()
    kx = _infer_k(xs, k_x, "k_x")
    ky = _infer_k(ys, k_y, "k_y")
    res = core.equate_observed_scores(xs, ys, int(kx), int(ky), method=method)
    return _build(res, method, "EG")


def equate_neat(
    x_total: np.ndarray,
    x_anchor: np.ndarray,
    y_total: np.ndarray,
    y_anchor: np.ndarray,
    method: str = "chained",
    k_x: int | None = None,
    k_y: int | None = None,
    k_v: int | None = None,
    w1: float = 0.5,
) -> EquateResult:
    """NEAT (common-item non-equivalent groups) observed-score equating (compute
    in Rust; Kolen & Brennan, 2014). Population 1 takes form X plus the anchor V
    (``x_total``, ``x_anchor``); population 2 takes form Y plus the anchor V
    (``y_total``, ``y_anchor``). ``method`` is ``"chained"`` (chained
    equipercentile, no population assumption) or ``"frequency_estimation"``
    (post-stratification, assuming population-invariant score-given-anchor
    conditionals). ``w1`` is the population-1 synthetic-population weight (used by
    frequency estimation only). ``k_x``/``k_y``/``k_v`` are the maximum X/Y/anchor
    scores; inferred from the data if omitted (pass them when the ceiling may be
    unobserved).

    Frequency estimation assumes the two groups share the anchor's support; where
    they do not, the synthetic densities are renormalized, so a poorly overlapping
    anchor degrades gracefully toward each group's own marginal rather than
    erroring. Chained equating makes no such assumption.

    References (APA 7th ed.):
        Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
            linking: Methods and practices* (3rd ed.). Springer.
            https://doi.org/10.1007/978-1-4939-0317-7
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_neat"):
        raise RuntimeError("equate_neat requires the compiled Rust core")
    xt = np.asarray(x_total, dtype=np.float64).ravel()
    xa = np.asarray(x_anchor, dtype=np.float64).ravel()
    yt = np.asarray(y_total, dtype=np.float64).ravel()
    ya = np.asarray(y_anchor, dtype=np.float64).ravel()
    kx = _infer_k(xt, k_x, "k_x")
    ky = _infer_k(yt, k_y, "k_y")
    kv = _infer_k(np.concatenate([xa, ya]), k_v, "k_v")
    res = core.equate_neat(
        xt, xa, yt, ya, int(kx), int(ky), int(kv), method=str(method), w1=float(w1)
    )
    return _build(res, str(method), "NEAT")


def equate_neat_linear(
    x_total: np.ndarray,
    x_anchor: np.ndarray,
    y_total: np.ndarray,
    y_anchor: np.ndarray,
    method: str = "tucker",
    anchor_kind: str = "internal",
    k_x: int | None = None,
    k_y: int | None = None,
    w1: float = 0.5,
) -> EquateResult:
    """Tucker & Levine linear observed-score NEAT equating (compute in Rust; Kolen
    & Brennan, 2014, §4.3-4.4) -- the linear counterpart to :func:`equate_neat`'s
    equipercentile methods. Population 1 takes form X plus the anchor V; population
    2 takes form Y plus the anchor V. ``method`` is ``"tucker"`` (equal
    total-on-anchor regression across populations) or ``"levine"`` (classical-
    congeneric). ``anchor_kind`` is ``"internal"`` (anchor items count toward the
    total) or ``"external"`` (separate section) and affects the Levine gamma only
    (Tucker is anchor-kind-invariant). ``w1`` is the population-1 synthetic weight.
    With equal anchor moments in the two groups every variant collapses to the
    equivalent-groups linear equating. Returns an :class:`EquateResult` whose
    ``slope``/``intercept`` are the linear conversion and whose moments are the
    synthetic-population moments.

    References (APA 7th ed.):
        Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
            linking: Methods and practices* (3rd ed.). Springer.
            https://doi.org/10.1007/978-1-4939-0317-7
        Brennan, R. L. (2006). *Chained linear equating* (CASMA Technical Report
            No. 3). University of Iowa.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_neat_linear"):
        raise RuntimeError("equate_neat_linear requires the compiled Rust core")
    xt = np.asarray(x_total, dtype=np.float64).ravel()
    xa = np.asarray(x_anchor, dtype=np.float64).ravel()
    yt = np.asarray(y_total, dtype=np.float64).ravel()
    ya = np.asarray(y_anchor, dtype=np.float64).ravel()
    kx = _infer_k(xt, k_x, "k_x")
    ky = _infer_k(yt, k_y, "k_y")
    res = core.equate_neat_linear(
        xt, xa, yt, ya, int(kx), int(ky),
        method=str(method), anchor_kind=str(anchor_kind), w1=float(w1),
    )
    return _build(res, f"{method}-{anchor_kind}", "NEAT")


def loglinear_smooth(counts: np.ndarray, degree: int = 6) -> dict:
    """Univariate log-linear presmoothing of a score-frequency distribution
    (compute in Rust; Holland & Thayer, 2000; Kolen & Brennan, 2014, ch. 3): fit
    ``log m_x = sum_j beta_j q_j(x)`` by Poisson ML so the smoothed density
    preserves the first ``degree`` sample moments exactly while damping sampling
    noise. ``counts`` are raw frequencies over scores ``0..=k`` (length ``k+1``);
    ``degree = k`` reproduces the raw relative frequencies. Returns a dict with
    ``probs`` (smoothed density), ``log_lik``, ``aic``, ``bic`` (comparable across
    degrees on the same data), ``moments`` (fitted moments on the ``u = x/k`` scale,
    orders ``1..=degree``), ``converged``, ``iters``, ``termination_reason``,
    ``final_gradient_max``, and ``gradient_tolerance``.

    References (APA 7th ed.):
        Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear
            models for discrete test score distributions. *Journal of Educational
            and Behavioral Statistics, 25*(2), 133-183.
            https://doi.org/10.3102/10769986025002133
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "loglinear_smooth"):
        raise RuntimeError("loglinear_smooth requires the compiled Rust core")
    c = np.asarray(counts, dtype=np.float64).ravel()
    # the model preserves at most k = len(counts)-1 moments; clamp so the default
    # degree works on short forms (k < 6) instead of erroring
    deg = max(1, min(int(degree), c.size - 1))
    res = core.loglinear_smooth(c, deg)
    return {
        "probs": np.asarray(res["probs"], dtype=np.float64),
        "log_lik": float(res["log_lik"]),
        "aic": float(res["aic"]),
        "bic": float(res["bic"]),
        "moments": np.asarray(res["moments"], dtype=np.float64),
        "converged": bool(res["converged"]),
        "iters": int(res["iters"]),
        "termination_reason": str(res["termination_reason"]),
        "final_gradient_max": float(res["final_gradient_max"]),
        "gradient_tolerance": float(res["gradient_tolerance"]),
    }


def equate_observed_scores_kernel(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    continuization: str = "gaussian",
    k_x: int | None = None,
    k_y: int | None = None,
    smooth_x: int | None = None,
    smooth_y: int | None = None,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
) -> EquateResult:
    """Equivalent-groups equating with optional log-linear presmoothing and a
    choice of continuization kernel (compute in Rust; Kolen & Brennan, 2014; von
    Davier, Holland & Thayer, 2004). ``continuization`` is ``"uniform"`` (the
    Kolen-Brennan equipercentile, identical to
    :func:`equate_observed_scores`) or ``"gaussian"`` (kernel equating).
    ``smooth_x``/``smooth_y`` presmooth each form (``None`` = raw frequencies, each
    ``>= 1`` when given); ``bandwidth_x``/``bandwidth_y`` fix the Gaussian bandwidth
    (``None`` = penalty-selected). The chosen bandwidths are returned on
    ``EquateResult.h_x``/``h_y`` (``NaN`` for the uniform kernel). This entry point
    defaults to the Gaussian kernel (unlike the plain
    :func:`equate_observed_scores`, whose equipercentile is the uniform kernel).
    Presmoothing must actually satisfy its stopping criterion. If the Poisson
    log-linear fit does not converge, this function raises ``ValueError`` instead of
    constructing an equating table from an unfinished density; use
    :func:`loglinear_smooth` directly to inspect ``converged`` and ``iters``.

    References (APA 7th ed.):
        von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel
            method of test equating*. Springer. https://doi.org/10.1007/b97446
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores_ext"):
        raise RuntimeError("equate_observed_scores_kernel requires the compiled Rust core")
    xs = np.asarray(x_scores, dtype=np.float64).ravel()
    ys = np.asarray(y_scores, dtype=np.float64).ravel()
    for nm, sv in (("smooth_x", smooth_x), ("smooth_y", smooth_y)):
        if sv is not None and int(sv) < 1:
            raise ValueError(f"{nm} must be >= 1")
    kx = _infer_k(xs, k_x, "k_x")
    ky = _infer_k(ys, k_y, "k_y")
    res = core.equate_observed_scores_ext(
        xs, ys, int(kx), int(ky),
        continuization=str(continuization),
        smooth_degree_x=None if smooth_x is None else int(smooth_x),
        smooth_degree_y=None if smooth_y is None else int(smooth_y),
        bandwidth_x=None if bandwidth_x is None else float(bandwidth_x),
        bandwidth_y=None if bandwidth_y is None else float(bandwidth_y),
    )
    return _build(res, f"{continuization}-kernel", "EG")


def equating_standard_errors(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    method: str = "equipercentile",
    route: str = "bootstrap",
    k_x: int | None = None,
    k_y: int | None = None,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Standard errors of equating (SEE) for the equivalent-groups design (compute
    in Rust; Kolen & Brennan, 2014, ch. 7): the sampling error of the equated score
    at each raw score point. ``route="bootstrap"`` (the default) resamples
    examinees per group independently with replacement, re-equates ``n_boot`` times,
    and returns the per-score bootstrap SD and a percentile CI -- it works for every
    ``method`` (``"mean"``/``"linear"``/``"equipercentile"``). ``route="analytic"``
    returns the closed-form delta-method (normal-theory) SEE for ``"mean"``/
    ``"linear"`` only. Returns a dict with ``x_scores``, ``y_equivalents`` (the
    point estimate), ``se``, ``ci_lo``, ``ci_hi`` (all length ``k_x+1``), ``n_boot``
    (0 for the analytic route), and ``ci_level``.

    References (APA 7th ed.):
        Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
            linking: Methods and practices* (3rd ed.). Springer.
            https://doi.org/10.1007/978-1-4939-0317-7
        Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*.
            Chapman & Hall.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None:
        raise RuntimeError("equating_standard_errors requires the compiled Rust core")
    xs = np.asarray(x_scores, dtype=np.float64).ravel()
    ys = np.asarray(y_scores, dtype=np.float64).ravel()
    kx = _infer_k(xs, k_x, "k_x")
    ky = _infer_k(ys, k_y, "k_y")
    if route == "bootstrap":
        if not hasattr(core, "bootstrap_see"):
            raise RuntimeError("bootstrap SEE requires the compiled Rust core")
        res = core.bootstrap_see(
            xs, ys, int(kx), int(ky),
            method=str(method), n_boot=int(n_boot), ci_level=float(ci_level), seed=int(seed),
        )
    elif route == "analytic":
        if not hasattr(core, "analytic_see"):
            raise RuntimeError("analytic SEE requires the compiled Rust core")
        res = core.analytic_see(xs, ys, int(kx), int(ky), method=str(method), ci_level=float(ci_level))
    else:
        raise ValueError("route must be 'bootstrap' or 'analytic'")
    return {
        "x_scores": np.asarray(res["x_scores"], dtype=np.float64),
        "y_equivalents": np.asarray(res["y_equivalents"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "ci_lo": np.asarray(res["ci_lo"], dtype=np.float64),
        "ci_hi": np.asarray(res["ci_hi"], dtype=np.float64),
        "n_boot": int(res["n_boot"]),
        "ci_level": float(res["ci_level"]),
    }


@dataclass
class CircleArcResult:
    """Circle-arc equating result. ``equated`` are the reference-form
    equivalents of the requested scores. ``xc``/``yc``/``r2`` describe the
    fitted circle -- in raw coordinates for method ``"arc1"``, in the
    transformed (``y* = y - L(x)``) coordinates for ``"arc2"``; all three are
    ``NaN`` when ``collinear`` is ``True`` (the estimate degenerates to the
    straight line through the points). ``middle`` is the ``(x2, y2)`` middle
    point used."""

    equated: np.ndarray
    xc: float
    yc: float
    r2: float
    collinear: bool
    middle: tuple[float, float]
    method: str


def _ca_point(p, name: str) -> tuple[float, float]:
    """Validate and unpack an ``(x, y)`` anchor point into a float pair."""
    try:
        x, y = p
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an (x, y) pair") from None
    if np.iscomplexobj(np.asarray(x)) or np.iscomplexobj(np.asarray(y)):
        raise ValueError(f"{name} must be real-valued")
    try:
        return (float(x), float(y))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an (x, y) pair of numbers") from None


def circle_arc_equate(
    scores: np.ndarray,
    low: tuple[float, float],
    middle: tuple[float, float],
    high: tuple[float, float],
    method: str = "arc2",
) -> CircleArcResult:
    """Circle-arc small-sample observed-score equating (compute in Rust;
    Livingston & Kim, 2008). The equating curve is constrained through three
    points: the prespecified end-points ``low = (x1, y1)`` and
    ``high = (x3, y3)`` and an empirically determined ``middle = (x2, y2)``
    (for single-group / equivalent-groups designs, the pair of mean scores;
    for the anchor design see :func:`circle_arc_middle_anchor`). ``method``
    ``"arc1"`` fits a circle arc directly through the three points;
    ``"arc2"`` (the default; the most accurate small-sample method in the
    source's resampling study) decomposes the curve into the line ``L(x)``
    through the end-points plus an arc fitted to the transformed points.
    Scores must lie in ``[x1, x3]``: the source's linear extension below the
    lower end-point is intentionally NOT implemented (reduced scope). When
    the three points are collinear the estimate is the line itself and
    ``collinear`` is ``True``. Raises ``ValueError`` if the fitted circle does
    not carry all three points on a single branch (an end-point on the
    opposite side of the center from the middle point), since the arc is
    then not a function of X.

    References (APA 7th ed.):
        Livingston, S. A., & Kim, S. (2008). *Small-sample equating by the
            circle-arc method* (Research Report No. RR-08-39). ETS.
            https://doi.org/10.1002/j.2333-8504.2008.tb02135.x
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "circle_arc_equate"):
        raise RuntimeError("circle_arc_equate requires the compiled Rust core")
    s = np.asarray(scores)
    if np.iscomplexobj(s):
        raise ValueError("scores must be real-valued")
    try:
        s = s.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError("scores must be numeric") from None
    s = np.ascontiguousarray(s.ravel())
    res = core.circle_arc_equate(
        s,
        _ca_point(low, "low"),
        _ca_point(middle, "middle"),
        _ca_point(high, "high"),
        str(method),
    )
    return CircleArcResult(
        equated=np.asarray(res["equated"], dtype=np.float64),
        xc=float(res["xc"]),
        yc=float(res["yc"]),
        r2=float(res["r2"]),
        collinear=bool(res["collinear"]),
        middle=(float(res["middle"][0]), float(res["middle"][1])),
        method=str(method),
    )


def circle_arc_middle_anchor(
    m_xa: float,
    m_va: float,
    m_yb: float,
    s_yb: float,
    m_vb: float,
    s_vb: float,
) -> tuple[float, float]:
    """Middle point for circle-arc equating under the anchor (NEAT) design
    (compute in Rust; Livingston & Kim, 2008, eq. 9). With ``x2`` chosen as
    the new-form mean ``m_xa``, the chained-linear middle point simplifies to
    ``y2 = m_yb + (s_yb / s_vb) * (m_va - m_vb)`` where ``m``/``s`` are means
    and standard deviations, ``a``/``b`` index the groups taking the new and
    reference forms, and ``v`` the common anchor. Returns ``(x2, y2)``.

    References (APA 7th ed.):
        Livingston, S. A., & Kim, S. (2008). *Small-sample equating by the
            circle-arc method* (Research Report No. RR-08-39). ETS.
            https://doi.org/10.1002/j.2333-8504.2008.tb02135.x
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "circle_arc_middle_anchor"):
        raise RuntimeError("circle_arc_middle_anchor requires the compiled Rust core")
    vals = []
    for name, v in (
        ("m_xa", m_xa), ("m_va", m_va), ("m_yb", m_yb),
        ("s_yb", s_yb), ("m_vb", m_vb), ("s_vb", s_vb),
    ):
        if np.iscomplexobj(np.asarray(v)):
            raise ValueError(f"{name} must be real-valued")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
    x2, y2 = core.circle_arc_middle_anchor(*vals)
    return (float(x2), float(y2))


def nominal_weights_mean_equate(
    x_total: np.ndarray,
    x_anchor: np.ndarray,
    y_total: np.ndarray,
    y_anchor: np.ndarray,
    k_x: int,
    k_y: int,
    k_v: int,
    w1: float = 0.5,
) -> EquateResult:
    """Nominal weights mean equating for the NEAT design (compute in Rust;
    Babcock, Albano, & Raymond, 2012 -- method as restated by Albano, 2016,
    eq. 42, whose derivation and the method authors' R package ``equate``
    were verified; the 2012 article itself was not read). Designed for very
    small samples: the Tucker regression slopes are replaced by the nominal
    weights ``gamma1 = k_x / k_v`` and ``gamma2 = k_y / k_v`` (item counts;
    for the 0..K integer-scored tests in scope these equal the score maxima),
    the synthetic means follow Albano (2016, eqs. 37-38), and the conversion
    is mean equating ``yx(x) = x + (mu_sY - mu_sX)`` (eq. 10, slope exactly
    1). Synthetic variances (eqs. 39-40, population/N-denominator moment
    convention -- NOT the N-1 sample variances the R package reports) are
    returned in ``sigma_x``/``sigma_y`` but do not enter the conversion.
    ``w1`` is the population-1 synthetic weight; when ``k_x == k_y`` the
    intercept is w1-invariant.

    References (APA 7th ed.):
        Babcock, B., Albano, A., & Raymond, M. (2012). Nominal weights mean
            equating: A method for very small samples. *Educational and
            Psychological Measurement, 72*(4), 608-628.
            https://doi.org/10.1177/0013164411428609
        Albano, A. D. (2016). equate: An R package for observed-score linking
            and equating. *Journal of Statistical Software, 74*(8), 1-36.
            https://doi.org/10.18637/jss.v074.i08
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "nominal_weights_mean_equate"):
        raise RuntimeError(
            "nominal_weights_mean_equate requires the compiled Rust core"
        )
    arrs = []
    for name, v in (
        ("x_total", x_total), ("x_anchor", x_anchor),
        ("y_total", y_total), ("y_anchor", y_anchor),
    ):
        a = np.asarray(v)
        if np.iscomplexobj(a):
            raise ValueError(f"{name} must be real-valued")
        try:
            a = a.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric") from None
        arrs.append(np.ascontiguousarray(a.ravel()))
    ks = []
    for name, k in (("k_x", k_x), ("k_y", k_y), ("k_v", k_v)):
        ki = int(k)
        if ki <= 0:
            raise ValueError(f"{name} must be a positive integer")
        ks.append(ki)
    res = core.nominal_weights_mean_equate(*arrs, *ks, w1=float(w1))
    return _build(res, "nominal-weights-mean", "NEAT")

def composite_linking(tables, weights, slopes=None, p=1.0):
    """Composite linking of component conversion tables.

    Weighted average of H component linking functions over a shared x grid
    (Holland & Strawderman, 2011, as cited by Albano, 2016, eq. 31). When
    per-component linear ``slopes`` are supplied, the symmetric
    Holland-Strawderman weight adjustment is applied (Albano, 2016, eq. 32):
    ``W_h = w_h (1 + a_h^p)^(-1/p) / sum(...)``. Without slopes, raw weights
    are normalized (``W_h = w_h / sum(w)``) -- a documented deviation from
    the R ``equate`` package's un-normalized non-symmetric path (identical
    results iff the supplied weights sum to 1).

    Returns a dict with ``composite`` (ndarray), ``adjusted_weights``
    (ndarray) and ``symmetric`` (bool).

    References (APA 7th):
        Albano, A. D. (2016). equate: An R package for observed-score
            linking and equating. Journal of Statistical Software, 74(8),
            1-36. https://doi.org/10.18637/jss.v074.i08  [READ]
        Holland, P. W., & Strawderman, W. E. (2011). How to average equating
            functions, if you must. In A. A. von Davier (Ed.), Statistical
            models for test equating, scaling, and linking (pp. 89-107).
            Springer.  [NOT READ; cited via Albano (2016)]
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "composite_linking"):
        raise RuntimeError("composite_linking requires the compiled Rust core")
    tabs = []
    for i, t in enumerate(tables):
        a = np.asarray(t)
        if np.iscomplexobj(a):
            raise ValueError(f"tables[{i}] must be real-valued")
        try:
            a = a.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"tables[{i}] must be numeric") from None
        tabs.append(np.ascontiguousarray(a.ravel()))
    w = np.asarray(weights)
    if np.iscomplexobj(w):
        raise ValueError("weights must be real-valued")
    try:
        w = w.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError("weights must be numeric") from None
    w = np.ascontiguousarray(w.ravel())
    s = None
    if slopes is not None:
        s = np.asarray(slopes)
        if np.iscomplexobj(s):
            raise ValueError("slopes must be real-valued")
        try:
            s = s.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("slopes must be numeric") from None
        s = np.ascontiguousarray(s.ravel())
    res = core.composite_linking(tabs, w, slopes=s, p=float(p))
    return {
        "composite": np.asarray(res["composite"]),
        "adjusted_weights": np.asarray(res["adjusted_weights"]),
        "symmetric": bool(res["symmetric"]),
    }
