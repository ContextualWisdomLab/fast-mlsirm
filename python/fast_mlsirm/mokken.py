"""Mokken scale analysis: Loevinger scalability coefficients and the automated
item selection procedure (AISP). All numeric work happens in the Rust core;
this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES


@dataclass
class MokkenResult:
    """Mokken scalability coefficients and (optionally) AISP scale labels.

    ``hij`` is the ``items x items`` matrix of pairwise scalability
    coefficients (NaN diagonal), ``hi`` the per-item coefficients, ``h`` the
    total scale coefficient; ``zij``/``zi``/``z`` are the matching Mokken Z
    statistics for the null hypothesis of inter-item independence.
    ``scale`` holds per-item AISP labels: 0 = unscalable, 1, 2, ... in
    formation order. Sample statistics follow the mokken R package
    (van der Ark, 2007)."""

    hij: np.ndarray
    hi: np.ndarray
    h: float
    zij: np.ndarray
    zi: np.ndarray
    z: float
    scale: np.ndarray


def mokken_analysis(
    responses: np.ndarray,
    lower_bound: float = 0.3,
    alpha: float = 0.05,
) -> MokkenResult:
    """Mokken scale analysis (compute in Rust; Mokken, 1971, as cited in
    van der Ark, 2007).

    Computes the Loevinger scalability coefficients ``Hij``, ``Hi``, ``H``
    with their Mokken Z statistics, and partitions the items into Mokken
    scales with the automated item selection procedure (AISP), following the
    sample statistics and "search normal" algorithm of the mokken R package
    (van der Ark, 2007): ``Hij = S_ij / Smax_ij`` where ``S`` is the sample
    covariance matrix and ``Smax_ij`` the maximum covariance given the two
    items' marginals (sorted-column coupling); ``Hi`` and ``H`` are ratios of
    the corresponding pairwise sums. A Mokken scale at lower bound ``c``
    requires nonnegative inter-item covariances and ``Hi >= c`` (rule of
    thumb ``c = 0.3``; Straat et al., 2013).

    In LLM-as-a-Judge item-quality management, AISP flags evaluation items
    that do not scale with the rest (label 0) and detects multidimensional
    item pools before parametric IRT calibration.

    ``responses`` is a complete ``persons x items`` array of integer scores
    (dichotomous 0/1 or polytomous); missing values are not supported —
    Mokken sample statistics assume complete data (van der Ark, 2007).

    References (APA 7th ed.):
        van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
            Statistical Software, 20*(11), 1-19.
            https://doi.org/10.18637/jss.v020.i11
        Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing
            optimization algorithms for item selection in Mokken scale
            analysis. *Journal of Classification, 30*(1), 75-99.
            https://doi.org/10.1007/s00357-013-9122-y
        Mokken, R. J. (1971). *A theory and procedure of scale analysis*.
            De Gruyter. (as cited in van der Ark, 2007)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mokken_coef_h"):
        raise RuntimeError("mokken_analysis requires the compiled Rust core")

    if not np.isfinite(lower_bound) or not (0.0 <= lower_bound < 1.0):
        raise ValueError("lower_bound must be in [0, 1)")
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")
    if np.any(y != np.floor(y)) or np.any(y < 0):
        raise ValueError("responses must be non-negative integer scores")
    if y.size and int(y.max()) + 1 > MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(
            f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
        )
    x = y.astype(np.int64).reshape(-1)
    res = core.mokken_coef_h(x, int(n_persons), int(n_items))
    scale = core.mokken_aisp(
        x, int(n_persons), int(n_items), float(lower_bound), float(alpha)
    )
    return MokkenResult(
        hij=np.asarray(res["hij"], dtype=np.float64).reshape(n_items, n_items),
        hi=np.asarray(res["hi"], dtype=np.float64),
        h=float(res["h"]),
        zij=np.asarray(res["zij"], dtype=np.float64).reshape(n_items, n_items),
        zi=np.asarray(res["zi"], dtype=np.float64),
        z=float(res["z"]),
        scale=np.asarray(scale, dtype=np.int64),
    )
