"""Architecture and resource fitness for dynamic evaluation contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fast_mlsirm.measurement as measurement
from fast_mlsirm.measurement import dynamic_evaluation
from fast_mlsirm.measurement.dynamic_evaluation import (
    DynamicEvaluationContractError,
    DynamicItemOrigin,
    EvaluationItemRole,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    build_dynamic_evaluation_item,
    build_evaluation_item_set_snapshot,
)


class _ExplodingOverBudgetList(list[str]):
    """Prove allocation admission happens before caller-controlled iteration."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Fail if production code iterates an already over-budget collection."""
        raise AssertionError("over-budget references must be rejected before iteration")


def _item():  # type: ignore[no-untyped-def]
    """Build one valid item for integrity and public-contract tests."""
    return build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_alpha",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_alpha",
        content_sha256="a" * 64,
        origin=DynamicItemOrigin.GENERATED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_refs=("criterion_accuracy",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=RegenerationStatus.INPUTS_RECORDED,
    )


def test_reference_budget_is_checked_before_iterating_caller_input() -> None:
    """An oversized collection fails by length without traversing its contents."""
    criteria = _ExplodingOverBudgetList(
        f"criterion_{index}" for index in range(129)
    )

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_alpha",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_alpha",
            content_sha256="a" * 64,
            origin=DynamicItemOrigin.GENERATED,
            role=EvaluationItemRole.CANDIDATE,
            reference_semantics=ReferenceSemantics.RUBRIC,
            reference_status=ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_refs=criteria,
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            regeneration_status=RegenerationStatus.INPUTS_RECORDED,
        )

    assert caught.value.code == "invalid_reference_count"


def test_item_snapshot_rejects_post_construction_mutation() -> None:
    """A frozen item cannot be changed through object-level mutation and reused."""
    item = _item()
    object.__setattr__(item, "role", EvaluationItemRole.ANCHOR)

    with pytest.raises(DynamicEvaluationContractError) as caught:
        item.to_dict()
    assert caught.value.code == "item_snapshot_integrity_mismatch"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(item,),
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_snapshot_integrity_mismatch"

    malformed = _item()
    object.__setattr__(malformed, "role", object())
    with pytest.raises(DynamicEvaluationContractError) as caught:
        malformed.to_dict()
    assert caught.value.code == "item_snapshot_integrity_mismatch"


def test_run_snapshot_rejects_post_construction_mutation() -> None:
    """A frozen run cannot acquire a later linking claim through object mutation."""
    run = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(run, "linking_status", LinkingStatus.LINKED)

    with pytest.raises(DynamicEvaluationContractError) as caught:
        run.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _ = run.anchor_item_refs
    assert caught.value.code == "run_snapshot_integrity_mismatch"

    malformed = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_2",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(malformed, "linking_status", object())
    with pytest.raises(DynamicEvaluationContractError) as caught:
        malformed.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"


def test_snapshot_fingerprints_are_deterministic_and_exposed() -> None:
    """Equivalent admitted snapshots publish the same deterministic identities."""
    first = _item()
    second = _item()
    assert first.contract_id == "fast_mlsirm_dynamic_evaluation_item/v1"
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.to_dict()["snapshot_sha256"] == first.snapshot_sha256

    first_run = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(first,),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    second_run = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=[second],
        linking_status="unavailable",
    )
    assert first_run.snapshot_sha256 == second_run.snapshot_sha256
    assert first_run.to_dict()["snapshot_sha256"] == first_run.snapshot_sha256


def test_dynamic_evaluation_module_has_complete_docstrings() -> None:
    """Every class and function in the new public module has a real docstring."""
    source_path = Path(dynamic_evaluation.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    missing = [
        f"{node.name}@{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is None
    ]
    assert missing == []


def test_dynamic_evaluation_contract_is_exported_from_measurement_package() -> None:
    """Consumers can discover the versioned contract through its owner package."""
    assert (
        measurement.DYNAMIC_EVALUATION_ITEM_CONTRACT_ID
        == "fast_mlsirm_dynamic_evaluation_item/v1"
    )
    assert measurement.DynamicEvaluationItemSnapshot is (
        dynamic_evaluation.DynamicEvaluationItemSnapshot
    )
    assert measurement.EvaluationItemSetSnapshot is (
        dynamic_evaluation.EvaluationItemSetSnapshot
    )
