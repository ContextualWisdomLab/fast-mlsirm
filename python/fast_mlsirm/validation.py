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

    kwargs: dict[str, Any] = {}
    if human_human is not None:
        kwargs["human_a"] = np.asarray(human_human[0], dtype=np.uint32)
        kwargs["human_b"] = np.asarray(human_human[1], dtype=np.uint32)
    if subgroup is not None:
        kwargs["subgroup"] = np.asarray(subgroup, dtype=np.uint32)
    res = _core.validate_scoring(
        np.asarray(judge, dtype=np.uint32),
        np.asarray(human, dtype=np.uint32),
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
