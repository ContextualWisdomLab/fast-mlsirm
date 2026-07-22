"""Warm's (1989) weighted likelihood estimation of ability for unidimensional dichotomous IRT.

The bias-reduced maximum-likelihood ability estimator: it removes the leading ``O(1/n)`` bias of the MLE
and, unlike the MLE (which diverges to ``+/-infinity`` for a perfect or zero score), yields a finite
estimate for every response pattern. The numerical computation runs in Rust."""

from __future__ import annotations

import numpy as np


def score_wle(
    a: np.ndarray,
    b: np.ndarray,
    responses: np.ndarray,
    observed: np.ndarray | None = None,
    c: np.ndarray | None = None,
    d: np.ndarray | None = None,
    theta_bound: float = 20.0,
    tol: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Warm's weighted-likelihood ability estimate for a unidimensional dichotomous test (compute in
    Rust; Warm, 1989).

    Solves the weighted-likelihood estimating equation ``dlnL/dtheta + J(theta)/(2 I(theta)) = 0`` with
    ``J = sum_i P_i' P_i''/(P_i Q_i)`` (the Warm correction, computed directly). ``J = I'(theta)`` for
    the 2PL/Rasch only -- with any ``c > 0`` or ``d < 1`` it is neither ``I'`` nor ``I'/2``, so a
    ``sqrt(I)``-weighted estimator applies the wrong 3PL/4PL correction.
    Here ``P_i = c_i + (d_i - c_i) sigmoid(a_i (theta - b_i))``. The estimate
    removes the leading MLE bias and stays FINITE for the all-correct and all-incorrect patterns, and its
    standard error is ``1/sqrt(I(theta))``.

    ``a`` and ``b`` are the per-item slope (NATURAL scale, not log-alpha) and difficulty; ``c`` (lower
    asymptote, default ``0``) and ``d`` (upper asymptote, default ``1``) give the 3PL/4PL, with
    ``0 <= c_i < d_i <= 1`` (the defaults are the 2PL). ``responses`` is a persons x items ``0/1`` array
    (or a single length-items vector; ``NaN`` = missing, dropped per person), and ``observed`` an
    optional bool mask (defaults to the non-``NaN`` entries). ``theta_bound`` is the hard clamp on the
    root search: when the finite Warm root lies beyond it (very easy/hard items for the pattern) the
    estimate is clamped to the boundary and flagged. Returns per-person NumPy arrays ``theta``, ``se``,
    and ``boundary``. The Rust implementation adapts its bounded search grid to the steepest item and
    raises ``ValueError`` when resolving the global mode would exceed its 65,536-interval work limit;
    this numerical policy is specific to fast-mlsirm rather than Warm's statistical result.

    Reference (APA 7th ed.):
        Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
            *Psychometrika, 54*(3), 427-450. https://doi.org/10.1007/BF02294627
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle"):
        raise RuntimeError("score_wle requires the compiled Rust core")

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n_items = a.shape[0]
    if n_items == 0:
        raise ValueError("need at least one item")
    if b.shape[0] != n_items:
        raise ValueError("a and b must have the same length")
    c = np.zeros(n_items) if c is None else np.asarray(c, dtype=np.float64).reshape(-1)
    d = np.ones(n_items) if d is None else np.asarray(d, dtype=np.float64).reshape(-1)
    if c.shape[0] != n_items or d.shape[0] != n_items:
        raise ValueError("c and d must have the same length as a")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != n_items:
        raise ValueError("responses must be (n_persons, n_items) matching the item parameters")
    n_persons = y.shape[0]
    if observed is None:
        observed = ~np.isnan(y)
    else:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("observed must match responses shape")
    yy = np.where(observed, y, 0.0)
    if not np.all(np.isin(yy[observed], (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 where observed (NaN = missing)")

    res = core.score_wle(
        a,
        b,
        c,
        d,
        yy.reshape(-1),
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        float(theta_bound),
        float(tol),
    )
    return {
        "theta": np.asarray(res["theta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "boundary": np.asarray(res["boundary"], dtype=bool),
    }


def score_wle_poly(
    responses: np.ndarray,
    slope: np.ndarray,
    cat_params: np.ndarray,
    n_cat: int,
    model: str = "grm",
    observed: np.ndarray | None = None,
    theta_bound: float = 20.0,
    tol: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Warm's (1989) weighted-likelihood ability estimates for POLYTOMOUS items (compute in Rust).

    The polytomous counterpart of :func:`score_wle`. Solves
    ``dlnL/dtheta + J(theta)/(2 I(theta)) = 0`` with, per item and category ``k``,
    ``I = sum_k P'_k**2 / P_k`` and ``J = sum_k P'_k P''_k / P_k`` — the exact generalization of the
    dichotomous ``sum_i P' P''/(P Q)``, which is its two-category case. ``J`` is computed DIRECTLY, not
    as a derivative of ``I``.

    Unlike :func:`fast_mlsirm.score_polytomous` (EAP) this applies NO prior, so the estimate is not
    shrunk toward a population mean — the usual requirement when individual scores are reported. It
    stays FINITE for the all-lowest and all-highest response patterns, where the maximum-likelihood
    estimate diverges.

    ``model`` is ``"grm"`` or ``"gpcm"``. PCM is the GPCM path with ``slope`` all ones. RSM is NOT
    supported: its fitted ``(delta, shared tau)`` parameterization is not convertible through any
    exposed API.

    **Verification status.** That the polytomous Warm correction is ``J/(2I)`` with
    ``J = sum_k P' P''/P`` is confirmed from the ``catR`` package's source, not from a primary paper;
    ``catR``'s Jeffreys-prior branch uses a different expression and the two are kept distinct here
    (Magis & Raîche, 2012). Separately, and proved in-repository rather than taken from a source,
    ``J = I'`` holds exactly for both families shipped here — that identity is used only as a test
    oracle, never as an implementation shortcut, and it fails for models with per-boundary slopes or a
    lower asymptote. Because it is exact here, replacing ``J`` by a derivative of ``I`` would be
    behaviour-preserving for these two families and no polytomous test can detect it; the anchors that
    do are in the dichotomous suite. A family added later must re-derive ``J``. Penfield and Bergeron
    (2005) treat the GPCM but their equations were not obtainable and are not the source of anything
    here.

    ``responses`` is (n_persons, n_items) with categories in ``0..n_cat``; NaN marks missing unless
    ``observed`` is given. ``cat_params`` is (n_items, n_cat - 1). Returns ``theta``, ``se`` and
    ``boundary``; a person with no observed items gets NaN with ``boundary`` set.

    References (APA 7th ed.):
        Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized
            adaptive testing with the R package catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
        Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm.
            *Applied Psychological Measurement, 16*(2), 159-176.
            https://doi.org/10.1177/014662169201600206
        Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores.
            *Psychometrika, 34*(S1), 1-97. https://doi.org/10.1007/BF03372160
        Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
            *Psychometrika, 54*(3), 427-450. https://doi.org/10.1007/BF02294627
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle_poly"):
        raise RuntimeError("score_wle_poly requires the compiled Rust core")

    n_cat = int(n_cat)
    if n_cat < 2:
        raise ValueError("n_cat must be >= 2")
    slope = np.asarray(slope, dtype=np.float64).reshape(-1)
    n_items = slope.shape[0]
    if n_items == 0:
        raise ValueError("need at least one item")
    cat = np.asarray(cat_params, dtype=np.float64)
    if cat.shape != (n_items, n_cat - 1):
        raise ValueError("cat_params must be (n_items, n_cat - 1)")
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != n_items:
        raise ValueError("responses must be (n_persons, n_items) matching the item parameters")
    n_persons = y.shape[0]
    if observed is None:
        observed = ~np.isnan(y)
    else:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("observed must match responses shape")
    yy = np.where(observed, y, 0.0)
    seen = yy[observed]
    if seen.size and (not np.all(np.isfinite(seen)) or not np.all(seen == np.floor(seen))
                      or seen.min() < 0 or seen.max() > n_cat - 1):
        raise ValueError("responses must be integers in 0..n_cat-1 where observed (NaN = missing)")

    res = core.score_wle_poly(
        yy.reshape(-1).astype(np.int64),
        int(n_persons),
        int(n_items),
        n_cat,
        slope,
        cat.reshape(-1),
        observed.reshape(-1),
        str(model),
        float(theta_bound),
        float(tol),
    )
    return {
        "theta": np.asarray(res["theta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "boundary": np.asarray(res["boundary"], dtype=bool),
    }
