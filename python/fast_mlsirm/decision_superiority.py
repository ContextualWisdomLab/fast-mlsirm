"""Fail-closed transport for Rust-owned Holm--Wald superiority arithmetic.

This is a numerical foundation, not a calibrated-prediction certificate.  A
consumer must separately bind converged, identified covariance and calibration
evidence before using the result to authorize an operational decision.

References
----------
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
*Scandinavian Journal of Statistics, 6*(2), 65--70. In particular, pp. 65--66
define the ordered step-down procedure and its family-wise type-I error control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from . import _core

DECISION_SUPERIORITY_ALGORITHM: Final = "holm-wald-one-sided-v1"
"""Versioned identity of the public numerical decision kernel."""


@dataclass(frozen=True, slots=True)
class WaldSuperiorityComparison:
    """One ordered candidate contrast after Holm family-wise correction."""

    candidate_id: str
    comparator_id: str
    estimate_difference: float
    standard_error: float
    p_value: float
    null_rejected: bool


@dataclass(frozen=True, slots=True)
class WaldSuperiorityResult:
    """Unique superior candidate, or an explicit indeterminate outcome."""

    decision: str
    winner_id: str | None
    familywise_error_rate: float
    algorithm: str
    comparisons: tuple[WaldSuperiorityComparison, ...]


def _candidate_ids(values: object) -> tuple[str, ...]:
    """Return distinct inert candidate identifiers without caller callbacks."""
    if type(values) not in (tuple, list):
        raise TypeError("candidate_ids must be a tuple or list of strings")
    result: list[str] = []
    for value in values:
        if type(value) is not str or not value.strip():
            raise ValueError("candidate_ids must contain non-empty exact strings")
        result.append(value)
    if len(result) < 2:
        raise ValueError("at least two candidate_ids are required")
    if len(set(result)) != len(result):
        raise ValueError("candidate_ids must be distinct")
    return tuple(result)


def _finite_vector(values: object, expected: int) -> np.ndarray:
    """Validate one finite float64 estimate vector for Rust marshalling."""
    if type(values) is not np.ndarray:
        raise TypeError("estimates must be a NumPy array")
    if values.ndim != 1 or values.shape[0] != expected:
        raise ValueError("estimates must contain one value per candidate")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("estimates must be real numeric values")
    result = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("estimates must be finite")
    return result


def _covariance_matrix(values: object, expected: int) -> np.ndarray:
    """Validate covariance storage while leaving matrix arithmetic to Rust."""
    if type(values) is not np.ndarray:
        raise TypeError("covariance must be a NumPy array")
    if values.ndim != 2 or values.shape != (expected, expected):
        raise ValueError("covariance must be a square matrix matching estimates")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("covariance must be real numeric values")
    result = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("covariance must be finite")
    return result


def assess_wald_superiority(
    candidate_ids: tuple[str, ...] | list[str],
    estimates: np.ndarray,
    covariance: np.ndarray,
    *,
    familywise_error_rate: float,
) -> WaldSuperiorityResult:
    """Assess unique candidate superiority with one-sided Holm--Wald tests.

    The supplied covariance must describe the joint estimator distribution.
    Rust computes every contrast variance, standard error, normal-tail
    probability, Holm rejection, and winner decision.  The caller must supply
    ``familywise_error_rate`` from a documented decision policy; there is no
    package default.  Candidate order is not a tie-break.

    This function alone does not prove convergence, identification, covariance
    calibration, interval coverage, or fitness for an operational decision.
    Those gates remain mandatory before a consumer may interpret ``superior``
    as decision-authorizing evidence.
    """
    identifiers = _candidate_ids(candidate_ids)
    estimate_vector = _finite_vector(estimates, len(identifiers))
    covariance_matrix = _covariance_matrix(covariance, len(identifiers))
    if type(familywise_error_rate) is not float:
        raise TypeError("familywise_error_rate must be an exact float")
    if not np.isfinite(familywise_error_rate) or not 0.0 < familywise_error_rate < 1.0:
        raise ValueError(
            "familywise_error_rate must be finite and strictly between zero and one"
        )
    raw = _core.holm_wald_superiority(
        estimate_vector,
        covariance_matrix,
        familywise_error_rate,
    )
    comparisons = tuple(
        WaldSuperiorityComparison(
            candidate_id=identifiers[int(row["candidate_index"])],
            comparator_id=identifiers[int(row["comparator_index"])],
            estimate_difference=float(row["estimate_difference"]),
            standard_error=float(row["standard_error"]),
            p_value=float(row["p_value"]),
            null_rejected=bool(row["null_rejected"]),
        )
        for row in raw["comparisons"]
    )
    winner_index = raw["winner_index"]
    winner_id = None if winner_index is None else identifiers[int(winner_index)]
    return WaldSuperiorityResult(
        decision="indeterminate" if winner_id is None else "superior",
        winner_id=winner_id,
        familywise_error_rate=familywise_error_rate,
        algorithm=DECISION_SUPERIORITY_ALGORITHM,
        comparisons=comparisons,
    )


__all__ = [
    "DECISION_SUPERIORITY_ALGORITHM",
    "WaldSuperiorityComparison",
    "WaldSuperiorityResult",
    "assess_wald_superiority",
]
