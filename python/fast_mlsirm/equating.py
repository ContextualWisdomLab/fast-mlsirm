"""Observed-score equating (Kolen & Brennan, 2014): the raw-score complement to
the IRT scale linking in :mod:`fast_mlsirm.linking`. Equivalent-groups mean /
linear / equipercentile equating and NEAT chained / frequency-estimation
equating, all computed in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _infer_k(scores: np.ndarray, k, name: str) -> int:
    if k is not None:
        return int(k)
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

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores"):
        raise RuntimeError("equate_observed_scores requires the compiled Rust core")
    xs = np.asarray(x_scores, dtype=np.float64).ravel()
    ys = np.asarray(y_scores, dtype=np.float64).ravel()
    kx = _infer_k(xs, k_x, "k_x")
    ky = _infer_k(ys, k_y, "k_y")
    res = core.equate_observed_scores(xs, ys, int(kx), int(ky), method=str(method))
    return _build(res, str(method), "EG")


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
