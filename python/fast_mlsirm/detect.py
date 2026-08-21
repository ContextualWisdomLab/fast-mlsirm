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
    response_array = np.asarray(responses)
    if np.iscomplexobj(response_array):
        raise ValueError("responses must be real-valued")
    if response_array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    y = response_array.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 2 or n_items < 2:
        raise ValueError("responses needs at least 2 persons and 2 items")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("responses must be exactly 0 or 1 (no missing values)")

    c = np.asarray(cluster).reshape(-1)
    if np.iscomplexobj(c):
        raise ValueError("cluster labels must be real integers")
    if c.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("cluster labels must be a numeric array")
    if c.shape[0] != n_items:
        raise ValueError("cluster must assign one label per item")
    I64_MAX = np.iinfo(np.int64).max
    I64_MIN = np.iinfo(np.int64).min
    if not np.issubdtype(c.dtype, np.integer):
        cf = c.astype(np.float64, copy=False)
        if not np.all(np.isfinite(cf)) or np.any(cf != np.round(cf)):
            raise ValueError("cluster labels must be integers")
        # Reject labels outside i64 before casting: astype(np.int64) on an
        # out-of-range float silently wraps/saturates, which would collapse
        # distinct labels and change the partition (equality-only contract).
        if np.any(cf < -(2.0**63)) or np.any(cf >= 2.0**63):
            raise ValueError("cluster labels must fit in a 64-bit integer")
        c = cf.astype(np.int64)
    else:
        # Also enforce i64 bounds for integer dtypes that can exceed i64
        # range. Unsigned types are always >= 0, so only the upper bound can
        # be violated; avoid comparing unsigned arrays to negative values to
        # prevent NumPy mixed-signedness promotion surprises. Signed types
        # narrower than or equal to i64 are always in range.
        if np.issubdtype(c.dtype, np.unsignedinteger):
            if np.any(c > I64_MAX):
                raise ValueError("cluster labels must fit in a 64-bit integer")
        elif np.any(c < I64_MIN) or np.any(c > I64_MAX):
            # Unreachable for a NumPy signed-integer array: no signed dtype is
            # wider than int64, so a value cannot fall outside [I64_MIN, I64_MAX].
            # Kept as a defensive guard.
            raise ValueError("cluster labels must fit in a 64-bit integer")  # pragma: no cover
        c = c.astype(np.int64)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "detect_analysis"):
        raise RuntimeError("detect_analysis requires the compiled Rust core")

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


@dataclass
class DimtestResult:
    """Confirmatory Stout-style DIMTEST statistic.

    ``t = (t_l - t_b) / sqrt(2)`` with ``t_l`` computed on AT1 and the bias
    correction ``t_b`` on AT2; ``p_value`` is the one-sided upper-tail normal
    p-value (reject essential unidimensionality when small).
    ``retained_pt_scores`` are the raw PT total scores of the retained
    (size >= 20) examinee groups, ascending."""

    t: float
    t_l: float
    t_b: float
    p_value: float
    groups_used: int
    n_discarded: int
    retained_pt_scores: np.ndarray


def dimtest(
    responses: np.ndarray,
    at1: np.ndarray,
    at2: np.ndarray,
) -> DimtestResult:
    """Confirmatory Stout-style DIMTEST of essential unidimensionality
    (compute in Rust; Stout, 1987, as described by Nandakumar & Stout, 1993).

    Splits the items into caller-supplied assessment subtests AT1 and AT2
    (equal length >= 4, disjoint) and the complementary partitioning subtest
    PT, groups examinees by raw PT total score (groups smaller than 20 are
    discarded), and within each retained group compares the observed ML
    variance of AT total scores with the variance expected under local
    independence, normalized by Stout's standard-error estimate. The AT1
    statistic ``T_L`` is bias-corrected by the same computation on the
    difficulty-matched AT2 subtest: ``T = (T_L - T_B) / sqrt(2)``, referred
    to the upper tail of the standard normal.

    Formulas were transcribed from Nandakumar and Stout's 1992 ERIC
    technical-report version (ED351383) of the 1993 paper, which describes
    Stout (1987, Sec. 4). NOT read: Stout (1987) original article, Stout et
    al. (2001), Froelich and Habing (2008), and DIM-Pack source code; this is
    the ORIGINAL second-AT bias correction, not DIMTEST 2 (no ATFIND, no
    bootstrap bias correction, no polytomous items, no missing data).

    In LLM-as-a-Judge item-quality management this tests whether a suspect
    subset of judge items (AT1) measures a second dimension distinct from
    the remaining items, with an explicit significance level.

    References (APA 7th ed.):
        Kieftenbeld, V., & Nandakumar, R. (2015). Alternative hypothesis
            testing procedures for DIMTEST. *Applied Psychological
            Measurement, 39*(6), 480-493. (read via PMC5978610)
        Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure
            for assessing latent trait unidimensionality. *Journal of
            Educational Statistics, 18*(1), 41-68.
            https://doi.org/10.2307/1165182 (read as ERIC report ED351383)
        Stout, W. (1987). A nonparametric approach for assessing latent
            trait unidimensionality. *Psychometrika, 52*(4), 589-617. (NOT
            read; as described by Nandakumar & Stout, 1993)
    """
    resp = np.asarray(responses)
    if np.iscomplexobj(resp):
        raise ValueError("responses must be real-valued")
    if resp.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    y = resp.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 1 or n_items < 1:
        raise ValueError("responses must be non-empty")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("responses must be exactly 0 or 1 (no missing values)")

    def _index_set(a: np.ndarray, name: str) -> list[int]:
        """Validate and return an item-index selection as a list of ints."""
        arr = np.asarray(a).reshape(-1)
        if np.iscomplexobj(arr):
            raise ValueError(f"{name} indices must be real integers")
        if arr.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueError(f"{name} indices must be a numeric array")
        af = arr.astype(np.float64)
        if arr.size == 0 or not np.all(np.isfinite(af)) or np.any(af != np.round(af)):
            raise ValueError(f"{name} indices must be non-empty integers")
        if np.any(af < 0) or np.any(af >= n_items):
            raise ValueError(f"{name} indices must be in [0, n_items)")
        return [int(v) for v in af]

    idx1 = _index_set(at1, "at1")
    idx2 = _index_set(at2, "at2")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_dimtest"):
        raise RuntimeError("dimtest requires the compiled Rust core")

    res = core.py_dimtest(y.reshape(-1), int(n_persons), int(n_items), idx1, idx2)
    return DimtestResult(
        t=float(res["t"]),
        t_l=float(res["t_l"]),
        t_b=float(res["t_b"]),
        p_value=float(res["p_value"]),
        groups_used=int(res["groups_used"]),
        n_discarded=int(res["n_discarded"]),
        retained_pt_scores=np.asarray(res["retained_pt_scores"], dtype=np.int64),
    )
