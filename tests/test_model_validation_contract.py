from __future__ import annotations

import pytest

from fast_mlsirm.model_validation import (
    GeneralizationUnit,
    ModelValidationPlan,
    ValidationStrategy,
    validate_group_partition,
)


def test_group_holdout_requires_a_declared_noncell_generalization_unit() -> None:
    plan = ModelValidationPlan(
        strategy=ValidationStrategy.GROUP_HOLDOUT,
        unit=GeneralizationUnit.QUERY_TESTLET,
    )

    assert plan.strategy is ValidationStrategy.GROUP_HOLDOUT
    assert plan.unit is GeneralizationUnit.QUERY_TESTLET
    assert "response_cell" not in {unit.value for unit in GeneralizationUnit}


def test_temporal_forward_requires_temporal_period_unit() -> None:
    ModelValidationPlan(
        strategy=ValidationStrategy.TEMPORAL_FORWARD,
        unit=GeneralizationUnit.TEMPORAL_PERIOD,
    )

    with pytest.raises(
        ValueError,
        match="temporal_forward validation requires the temporal_period unit",
    ):
        ModelValidationPlan(
            strategy=ValidationStrategy.TEMPORAL_FORWARD,
            unit=GeneralizationUnit.PERSON_SYSTEM,
        )


def test_plan_rejects_truthy_substitutes_for_closed_enums() -> None:
    with pytest.raises(TypeError, match="strategy must be a ValidationStrategy"):
        ModelValidationPlan(  # type: ignore[arg-type]
            strategy="group_holdout",
            unit=GeneralizationUnit.CLUSTER_CONTEXT,
        )

    with pytest.raises(TypeError, match="unit must be a GeneralizationUnit"):
        ModelValidationPlan(  # type: ignore[arg-type]
            strategy=ValidationStrategy.GROUP_HOLDOUT,
            unit="cluster_context",
        )


def test_group_partition_rejects_leakage_across_folds() -> None:
    with pytest.raises(
        ValueError,
        match="generalization group 'query-a' appears in multiple folds",
    ):
        validate_group_partition(
            group_ids=("query-a", "query-b", "query-a"),
            fold_ids=("fold-1", "fold-1", "fold-2"),
        )


def test_group_partition_accepts_repeated_groups_within_one_fold() -> None:
    validate_group_partition(
        group_ids=("query-a", "query-a", "query-b", "query-c", "query-c"),
        fold_ids=("fold-1", "fold-1", "fold-2", "fold-3", "fold-3"),
    )


def test_group_partition_rejects_malformed_identity_vectors() -> None:
    with pytest.raises(ValueError, match="group_ids and fold_ids must have equal length"):
        validate_group_partition(
            group_ids=("query-a",),
            fold_ids=("fold-1", "fold-2"),
        )

    with pytest.raises(ValueError, match="group_ids must not be empty"):
        validate_group_partition(group_ids=(), fold_ids=())

    with pytest.raises(ValueError, match="group_ids entries must be non-empty strings"):
        validate_group_partition(group_ids=("",), fold_ids=("fold-1",))

    with pytest.raises(ValueError, match="fold_ids entries must be non-empty strings"):
        validate_group_partition(group_ids=("query-a",), fold_ids=("",))


def test_group_partition_is_order_invariant() -> None:
    groups = ("query-a", "query-b", "query-c", "query-a")
    folds = ("fold-1", "fold-2", "fold-2", "fold-1")
    validate_group_partition(group_ids=groups, fold_ids=folds)

    order = (3, 2, 1, 0)
    validate_group_partition(
        group_ids=tuple(groups[index] for index in order),
        fold_ids=tuple(folds[index] for index in order),
    )
