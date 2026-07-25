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
