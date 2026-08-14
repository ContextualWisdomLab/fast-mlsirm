"""Govern model-selection validation units without computing psychometric statistics.

The model-selection workflow must validate at the unit implied by its intended
generalization claim. This module provides only immutable validation/orchestration
contracts: numerical likelihoods, bootstrap statistics, predictive scores, and
other result-affecting psychometric arithmetic remain Rust-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class GeneralizationUnit(str, Enum):
    """Scientific unit kept intact when evaluating model generalization."""

    PERSON_SYSTEM = "person_system"
    QUERY_TESTLET = "query_testlet"
    RATER_FAMILY = "rater_family"
    DOMAIN_LANGUAGE = "domain_language"
    CLUSTER_CONTEXT = "cluster_context"
    TEMPORAL_PERIOD = "temporal_period"


class ValidationStrategy(str, Enum):
    """Supported high-level resampling or holdout strategies."""

    GROUP_HOLDOUT = "group_holdout"
    TEMPORAL_FORWARD = "temporal_forward"
    BLOCK_BOOTSTRAP = "block_bootstrap"


@dataclass(frozen=True, slots=True)
class ModelValidationPlan:
    """Declare a validation strategy and the scientific unit it preserves.

    Response cells are intentionally not a generalization unit. A caller must
    instead declare the person/system, query/testlet, rater/family,
    domain/language, cluster/context, or temporal period whose observations must
    remain together. ``TEMPORAL_FORWARD`` is restricted to ``TEMPORAL_PERIOD``
    so future observations cannot be represented as an ordinary shuffled group
    holdout under this contract.
    """

    strategy: ValidationStrategy
    unit: GeneralizationUnit

    def __post_init__(self) -> None:
        """Reject untyped or temporally incoherent validation plans."""
        if not isinstance(self.strategy, ValidationStrategy):
            raise TypeError("strategy must be a ValidationStrategy")
        if not isinstance(self.unit, GeneralizationUnit):
            raise TypeError("unit must be a GeneralizationUnit")
        if (
            self.strategy is ValidationStrategy.TEMPORAL_FORWARD
            and self.unit is not GeneralizationUnit.TEMPORAL_PERIOD
        ):
            raise ValueError(
                "temporal_forward validation requires the temporal_period unit"
            )


def validate_group_partition(
    *,
    group_ids: Sequence[str],
    fold_ids: Sequence[str],
) -> None:
    """Reject a holdout/bootstrap partition that leaks a group across folds.

    ``group_ids`` and ``fold_ids`` are parallel observation-level identity
    vectors. Repeated observations from one declared scientific group may occur
    within one fold, but the same group cannot appear in two folds. Scalar
    strings are rejected rather than being misread as sequences of one-character
    identities. The function validates identities only; it performs no scoring,
    estimation, resampling, or model-selection arithmetic.

    Args:
        group_ids: Declared scientific group identity for each observation.
        fold_ids: Fold or resample-block identity for each observation.

    Raises:
        TypeError: If either identity vector is supplied as a scalar string.
        ValueError: If vectors are malformed or one group crosses fold
            boundaries.
    """
    if isinstance(group_ids, (str, bytes)) or isinstance(fold_ids, (str, bytes)):
        raise TypeError(
            "group_ids and fold_ids must be identity sequences, not strings"
        )
    if len(group_ids) != len(fold_ids):
        raise ValueError("group_ids and fold_ids must have equal length")
    if len(group_ids) == 0:
        raise ValueError("group_ids must not be empty")
    if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
        raise ValueError("group_ids entries must be non-empty strings")
    if any(not isinstance(fold_id, str) or not fold_id for fold_id in fold_ids):
        raise ValueError("fold_ids entries must be non-empty strings")

    group_fold: dict[str, str] = {}
    for group_id, fold_id in zip(group_ids, fold_ids, strict=True):
        prior_fold = group_fold.get(group_id)
        if prior_fold is not None and prior_fold != fold_id:
            raise ValueError(
                f"generalization group {group_id!r} appears in multiple folds"
            )
        group_fold[group_id] = fold_id


__all__ = [
    "GeneralizationUnit",
    "ModelValidationPlan",
    "ValidationStrategy",
    "validate_group_partition",
]
