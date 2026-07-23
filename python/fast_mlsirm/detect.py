"""Confirmatory DETECT dimensionality analysis (Zhang & Stout, 1999, as
implemented by CRAN sirt's sum-score conditioning path). All numeric work
happens in the Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectResult:
    """Confirmatory DETECT indices for a known item partition.

    ``detect``/``madcov100``/``mcov100`` are on the x100 scale of the sirt
    package; ``assi`` and ``ratio`` are unscaled in ``[-1, 1]``. ``pair_i``,
    ``pair_j``, ``ccov`` give the bias-corrected conditional covariance per
    item pair (``i < j``, row-major). Interpretation conventions quoted in
    the sirt documentation (as cited there from Jang & Roussos, 2007, and
    Zhang, 2007): DETECT < 0.2 suggests essential unidimensionality, >= 1.0
    sizeable multidimensionality relative to the supplied partition."""

    detect: float
    assi: float
    ratio: float
    madcov100: float
    mcov100: float
    n_pairs: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    ccov: np.ndarray


def detect_analysis(
    responses: np.ndarray,
    cluster: np.ndarray,
) -> DetectResult:
    """Confirmatory DETECT analysis of a binary response matrix (compute in
    Rust; Zhang & Stout, 1999, as cited in Robitzsch, 2024).

    Estimates the pairwise conditional covariances of the items given the
    raw total score and the pair rest score (bias-corrected average of the
    two conditionings), then aggregates them against the supplied item
    clustering into the DETECT, ASSI, RATIO, MADCOV100, and MCOV100 indices.
    Formulas were transcribed from the CRAN ``sirt`` R sources (read line by
    line); the original DETECT papers are paywalled and cited only through
    the sirt documentation. This matches sirt's explicit
    ``ccov.np(use_sum_score=TRUE, scale_score=FALSE)`` path; the kernel-
    smoothed default, missing data (sirt pairwise-deletes), exploratory
    cluster search, and polytomous DETECT are not implemented.

    In LLM-as-a-Judge item-quality management this diagnoses whether a
    rubric partition of judge items behaves as distinct dimensions (clearly
    positive DETECT) or as a single dimension (DETECT near zero).

    ``responses`` is a complete ``persons x items`` array with entries
    exactly 0 or 1. ``cluster`` assigns each item an integer label; labels
    are opaque (compared for equality only).

    References (APA 7th ed.):
        Jang, E. E., & Roussos, L. (2007). An investigation into the
            dimensionality of TOEFL using conditional covariance-based
            nonparametric approach. *Journal of Educational Measurement,
            44*(1), 1-21. (as cited in Robitzsch, 2024)
        Robitzsch, A. (2024). *sirt: Supplementary item response theory
            models* (R package). https://CRAN.R-project.org/package=sirt
        Zhang, J., & Stout, W. (1999). The theoretical DETECT index of
            dimensionality and its application to approximate simple
            structure. *Psychometrika, 64*(2), 213-249. (as cited in
            Robitzsch, 2024)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "detect_analysis"):
        raise RuntimeError("detect_analysis requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 2 or n_items < 2:
        raise ValueError("responses needs at least 2 persons and 2 items")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    c = np.asarray(cluster).reshape(-1)
    if c.shape[0] != n_items:
        raise ValueError("cluster must assign one label per item")
    if not np.issubdtype(c.dtype, np.integer):
        cf = np.asarray(cluster, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(cf)) or np.any(cf != np.round(cf)):
            raise ValueError("cluster labels must be integers")
        # Reject labels outside i64 before casting: astype(np.int64) on an
        # out-of-range float silently wraps/saturates, which would collapse
        # distinct labels and change the partition (equality-only contract).
        if np.any(cf < -(2.0**63)) or np.any(cf >= 2.0**63):
            raise ValueError("cluster labels must fit in a 64-bit integer")
        c = cf.astype(np.int64)

    res = core.detect_analysis(
        y.reshape(-1), int(n_persons), int(n_items), [int(v) for v in c]
    )
    return DetectResult(
        detect=float(res["detect"]),
        assi=float(res["assi"]),
        ratio=float(res["ratio"]),
        madcov100=float(res["madcov100"]),
        mcov100=float(res["mcov100"]),
        n_pairs=int(res["n_pairs"]),
        pair_i=np.asarray(res["pair_i"], dtype=np.int64),
        pair_j=np.asarray(res["pair_j"], dtype=np.int64),
        ccov=np.asarray(res["ccov"], dtype=np.float64),
    )
