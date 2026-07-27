"""Machine-scoring validation gates for LLM-as-a-Judge calibration.

Implements the operational criteria of Williamson, Xi & Breyer (2012),
"A Framework for Evaluation and Use of Automated Scoring" (EM:IP 31(1)):
quadratic-weighted kappa >= .70, Pearson r >= .70, degradation from the
human-human baseline <= .10, |SMD| <= .15 overall and <= .10 within every
subgroup; exact/adjacent agreement are reported but are explicitly NOT gates.
All computation runs in the Rust core (`mlsirm_core::agreement`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_labels(a, name: str, *, k: int | None = None, n: int | None = None) -> np.ndarray:
    """Validate caller-supplied category labels before the uint32 conversion the
    Rust gate expects: reject non-1-D, wrong-length, non-finite, non-integer,
    negative, or (when ``k`` given) out-of-range values instead of silently
    truncating/wrapping them (which would let malformed labels pass the gate)."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length must match the paired labels")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr.astype(np.float64))):
        raise ValueError(f"{name} must be finite")
    fl = arr.astype(np.float64)
    if np.any(fl < 0) or np.any(fl != np.floor(fl)):
        raise ValueError(f"{name} must be non-negative integers")
    if np.any(fl > np.iinfo(np.uint32).max):
        raise ValueError(f"{name} values must fit in uint32")
    if k is not None and np.any(fl >= k):
        raise ValueError(f"{name} values must be in 0..k-1")
    return arr.astype(np.uint32)


@dataclass
class ValidationVerdict:
    gates: list[dict[str, Any]]
    exact_agreement: float
    adjacent_agreement: float
    passed: bool
    failed_gates: list[str] = field(default_factory=list)


def validate_judge(
    judge: np.ndarray,
    human: np.ndarray,
    k: int = 2,
    human_human: tuple[np.ndarray, np.ndarray] | None = None,
    subgroup: np.ndarray | None = None,
) -> ValidationVerdict:
    """Run the Williamson et al. (2012) conjunctive acceptance gates.

    ``judge``/``human`` are paired labels in ``0..k-1``; ``human_human`` is an
    optional double-scored human baseline (pair of label vectors) for the
    degradation criterion; ``subgroup`` labels each observation for the
    fairness SMD.
    """
    from . import _core  # computation lives in the Rust core

    MAX_JUDGE_CATEGORIES = 1_000
    if int(k) < 2:
        raise ValueError("k (number of categories) must be >= 2")
    if int(k) > MAX_JUDGE_CATEGORIES:
        # k drives a dense k-by-k confusion matrix in the Rust core.
        raise ValueError(f"k (number of categories) must be <= {MAX_JUDGE_CATEGORIES}")
    judge_v = _validate_labels(judge, "judge", k=int(k))
    human_v = _validate_labels(human, "human", k=int(k), n=judge_v.shape[0])
    kwargs: dict[str, Any] = {}
    if human_human is not None:
        kwargs["human_a"] = _validate_labels(
            human_human[0], "human_a", k=int(k), n=judge_v.shape[0]
        )
        kwargs["human_b"] = _validate_labels(
            human_human[1], "human_b", k=int(k), n=kwargs["human_a"].shape[0]
        )
    if subgroup is not None:
        sg = _validate_labels(subgroup, "subgroup", n=judge_v.shape[0])
        # Compact to contiguous ids: the Rust core loops 0..max(subgroup)+1,
        # so a sparse label (e.g. uint32 max) is an O(n_groups) CPU-DoS.
        _uniq, sg_compact = np.unique(sg, return_inverse=True)
        kwargs["subgroup"] = sg_compact.astype(np.uint32)
    res = _core.validate_scoring(
        judge_v,
        human_v,
        int(k),
        **kwargs,
    )
    gates = [dict(g) for g in res["gates"]]
    return ValidationVerdict(
        gates=gates,
        exact_agreement=float(res["exact_agreement"]),
        adjacent_agreement=float(res["adjacent_agreement"]),
        passed=bool(res["pass"]),
        failed_gates=[g["name"] for g in gates if not g["pass"]],
    )


@dataclass
class FleissKappaResult:
    """Result of :func:`fleiss_kappa`. In exact (Conger) mode ``z`` and
    ``p_value`` are NaN and the category arrays are empty, mirroring irr's
    ``kappam.fleiss(exact=TRUE)`` which returns neither."""

    kappa: float
    subjects_used: int
    z: float
    p_value: float
    category_kappa: np.ndarray
    category_z: np.ndarray
    category_p: np.ndarray


def fleiss_kappa(
    ratings: np.ndarray,
    k: int | None = None,
    exact: bool = False,
) -> FleissKappaResult:
    """Fleiss' kappa for nominal agreement among multiple raters, with the
    exact (Conger) chance-agreement variant.

    Reimplements ``kappam.fleiss()`` from CRAN irr 0.85 (R source READ in
    full; algorithm source of truth). Model origins — cited as origins only,
    NOT READ: Fleiss, J. L. (1971). Measuring nominal scale agreement among
    many raters. *Psychological Bulletin, 76*(5), 378-382; Conger, A. J.
    (1980). Integration and generalization of kappas for multiple raters.
    *Psychological Bulletin, 88*(2), 322-328. Computation runs in the Rust
    core (``mlsirm_core::agreement::fleiss_kappa``).

    ``ratings`` is a 2-D ``(n_subjects, n_raters)`` array of integer category
    codes ``0..k-1``. NaN or any negative value marks a missing rating and
    drops the whole subject row (listwise, as in irr). ``k=None`` infers
    ``max(code)+1``; pass ``k`` explicitly to include trailing empty
    categories in the category-wise detail (their kappas are NaN, matching
    R's 0/0).
    """
    from . import _core  # computation lives in the Rust core

    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D (subjects x raters) array")
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype == object:
        for v in arr.flat:
            if v is None or isinstance(v, (bool, np.bool_, str, bytes)):
                raise ValueError("ratings must be numeric, not boolean/str/None")
        arr = arr.astype(np.float64)
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be integer codes, not booleans")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"ratings dtype {arr.dtype} is not numeric")
    ns, nr = arr.shape
    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if np.any(np.isinf(arr)):
            raise ValueError("ratings must not contain infinities")
        if np.any(arr[finite] != np.floor(arr[finite])):
            raise ValueError("ratings must be integer category codes")
        if np.any(np.abs(arr[finite]) > 2.0**53):
            raise ValueError("ratings exceed exact float64 integer range")
        codes = np.where(finite, arr, -1.0).astype(np.int64)
    else:
        if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
            raise ValueError("ratings values must fit in int64")
        codes = arr.astype(np.int64)
    if k is not None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError("k must be an integer")
        k = int(k)
    if k is None:
        if codes.size == 0 or int(codes.max()) < 0:
            raise ValueError("cannot infer k: no observed category codes")
        k = int(codes.max()) + 1
    res = _core.fleiss_kappa(
        np.ascontiguousarray(codes.reshape(-1)), int(ns), int(nr), int(k), bool(exact)
    )
    return FleissKappaResult(
        kappa=float(res["kappa"]),
        subjects_used=int(res["subjects_used"]),
        z=float(res["z"]),
        p_value=float(res["p_value"]),
        category_kappa=np.asarray(res["category_kappa"]),
        category_z=np.asarray(res["category_z"]),
        category_p=np.asarray(res["category_p"]),
    )
