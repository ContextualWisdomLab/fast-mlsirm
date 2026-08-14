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
    """Reject a degenerate or leakage-prone grouped validation partition.

    ``group_ids`` and ``fold_ids`` are parallel observation-level identity
    vectors. Repeated observations from one declared scientific group may occur
    within one fold, but the same group cannot appear in two folds. A usable
    holdout/bootstrap partition must contain at least two scientific groups and
    at least two folds; otherwise no out-of-group validation contrast exists.
    Scalar strings, blank identities, and identities with surrounding
    whitespace are rejected rather than normalized because silent trimming can
    merge caller-declared identities while accepting raw padded values can split
    one scientific identity into artificial groups/folds. The function validates
    identities only; it performs no scoring, estimation, resampling, or
    model-selection arithmetic.

    Args:
        group_ids: Declared scientific group identity for each observation.
        fold_ids: Fold or resample-block identity for each observation.

    Raises:
        TypeError: If either identity vector is supplied as a scalar string.
        ValueError: If vectors are malformed, degenerate, contain ambiguous
            padded identities, or one group crosses fold boundaries.
    """
    if isinstance(group_ids, (str, bytes)) or isinstance(fold_ids, (str, bytes)):
        raise TypeError(
            "group_ids and fold_ids must be identity sequences, not strings"
        )
    if len(group_ids) != len(fold_ids):
        raise ValueError("group_ids and fold_ids must have equal length")
    if len(group_ids) == 0:
        raise ValueError("group_ids must not be empty")
    if any(
        not isinstance(group_id, str) or not group_id.strip()
        for group_id in group_ids
    ):
        raise ValueError("group_ids entries must be non-empty strings")
    if any(group_id != group_id.strip() for group_id in group_ids):
        raise ValueError("group_ids entries must not contain surrounding whitespace")
    if any(
        not isinstance(fold_id, str) or not fold_id.strip()
        for fold_id in fold_ids
    ):
        raise ValueError("fold_ids entries must be non-empty strings")
    if any(fold_id != fold_id.strip() for fold_id in fold_ids):
        raise ValueError("fold_ids entries must not contain surrounding whitespace")
    if len(set(group_ids)) < 2:
        raise ValueError(
            "group partition requires at least two generalization groups"
        )
    if len(set(fold_ids)) < 2:
        raise ValueError("group partition requires at least two folds")

    group_fold: dict[str, str] = {}
    for group_id, fold_id in zip(group_ids, fold_ids, strict=True):
        prior_fold = group_fold.get(group_id)
        if prior_fold is not None and prior_fold != fold_id:
            raise ValueError(
                f"generalization group {group_id!r} appears in multiple folds"
            )
        group_fold[group_id] = fold_id


def validate_temporal_forward_window(
    *,
    training_periods: Sequence[int],
    validation_periods: Sequence[int],
) -> None:
    """Reject temporal validation windows with look-ahead or period overlap.

    Periods are caller-defined integer ordinals. The contract deliberately does
    not infer calendar semantics: it only requires every training period to
    strictly precede every validation period. This keeps chronology validation
    deterministic and separate from scoring, estimation, or model-selection
    arithmetic.

    Args:
        training_periods: Integer period ordinals used for fitting.
        validation_periods: Integer period ordinals used for validation.

    Raises:
        TypeError: If either period vector is supplied as a scalar string.
        ValueError: If vectors are empty, contain non-integers, or overlap in
            temporal order.
    """
    if isinstance(training_periods, (str, bytes)) or isinstance(
        validation_periods, (str, bytes)
    ):
        raise TypeError(
            "training_periods and validation_periods must be integer sequences"
        )
    if len(training_periods) == 0:
        raise ValueError("training_periods must not be empty")
    if len(validation_periods) == 0:
        raise ValueError("validation_periods must not be empty")
    if any(type(period) is not int for period in training_periods):
        raise ValueError("training_periods entries must be integers")
    if any(type(period) is not int for period in validation_periods):
        raise ValueError("validation_periods entries must be integers")
    if max(training_periods) >= min(validation_periods):
        raise ValueError("training periods must strictly precede validation periods")


__all__ = [
    "GeneralizationUnit",
    "ModelValidationPlan",
    "ValidationStrategy",
    "validate_group_partition",
    "validate_temporal_forward_window",
]
