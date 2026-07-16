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
    ``J = sum_i P_i' P_i''/(P_i Q_i)`` (the Warm correction, computed directly -- it is not ``I'/2``
    except for the 2PL/Rasch), where ``P_i = c_i + (d_i - c_i) sigmoid(a_i (theta - b_i))``. The estimate
    removes the leading MLE bias and stays FINITE for the all-correct and all-incorrect patterns, and its
    standard error is ``1/sqrt(I(theta))``.

    ``a`` and ``b`` are the per-item slope (NATURAL scale, not log-alpha) and difficulty; ``c`` (lower
    asymptote, default ``0``) and ``d`` (upper asymptote, default ``1``) give the 3PL/4PL, with
    ``0 <= c_i < d_i <= 1`` (the defaults are the 2PL). ``responses`` is a persons x items ``0/1`` array
    (or a single length-items vector; ``NaN`` = missing, dropped per person), and ``observed`` an
    optional bool mask (defaults to the non-``NaN`` entries). ``theta_bound`` is the hard clamp on the
    root search: when the finite Warm root lies beyond it (very easy/hard items for the pattern) the
    estimate is clamped to the boundary and flagged. Returns per-person NumPy arrays ``theta``, ``se``,
    and ``boundary``.

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
