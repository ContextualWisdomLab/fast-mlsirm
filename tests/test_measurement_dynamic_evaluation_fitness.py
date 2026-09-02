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
    build_evaluation_category_definition,
    build_evaluation_criterion_definition,
    build_evaluation_criterion_set_snapshot,
    build_evaluation_item_set_snapshot,
)


class _ExplodingOverBudgetList(list[str]):
    """Prove allocation admission happens before caller-controlled iteration."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Fail if production code iterates an already over-budget collection."""
        raise AssertionError("over-budget references must be rejected before iteration")


def _criterion_set():  # type: ignore[no-untyped-def]
    """Build one explicit criterion set for integrity tests."""
    categories = (
        build_evaluation_category_definition(
            category_ref="category_not_satisfied",
            definition_ref="definition_category_not_satisfied_1",
            definition_sha256="6" * 64,
            order_index=0,
        ),
        build_evaluation_category_definition(
            category_ref="category_satisfied",
            definition_ref="definition_category_satisfied_1",
            definition_sha256="7" * 64,
            order_index=1,
        ),
    )
    criterion = build_evaluation_criterion_definition(
        criterion_ref="criterion_accuracy",
        criterion_revision_ref="criterion_accuracy_revision_1",
        definition_ref="definition_criterion_accuracy_1",
        definition_sha256="c" * 64,
        admissible_evidence_rule_ref="admissible_evidence_accuracy_1",
        admissible_evidence_rule_sha256="1" * 64,
        exclusion_rule_ref="exclusion_accuracy_1",
        exclusion_rule_sha256="2" * 64,
        response_semantics_ref="response_semantics_accuracy_1",
        response_semantics_sha256="3" * 64,
        abstention_rule_ref="abstention_accuracy_1",
        abstention_rule_sha256="4" * 64,
        not_observable_rule_ref="not_observable_accuracy_1",
        not_observable_rule_sha256="5" * 64,
        category_definitions=categories,
    )
    return build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_accuracy_1",
        construct_ref="construct_accuracy_1",
        population_scope_ref="population_scope_synthetic_1",
        language_scope_ref="language_scope_synthetic_1",
        domain_scope_ref="domain_scope_contract_test_1",
        criteria=(criterion,),
    )


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
        criterion_set_snapshot=_criterion_set(),
        criterion_refs=("criterion_accuracy",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=RegenerationStatus.INPUTS_RECORDED,
    )


def test_reference_budget_is_checked_before_iterating_caller_input() -> None:
    """An oversized collection fails by length without traversing its contents."""
    criteria = _ExplodingOverBudgetList(f"criterion_{index}" for index in range(129))

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
            criterion_set_snapshot=_criterion_set(),
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
            criterion_set_snapshot=_criterion_set(),
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
        criterion_set_snapshot=_criterion_set(),
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
        criterion_set_snapshot=_criterion_set(),
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
        criterion_set_snapshot=_criterion_set(),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    second_run = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=[second],
        criterion_set_snapshot=_criterion_set(),
        linking_status="unavailable",
    )
    assert first_run.snapshot_sha256 == second_run.snapshot_sha256
    assert first_run.to_dict()["snapshot_sha256"] == first_run.snapshot_sha256


def test_dynamic_evaluation_module_has_complete_docstrings() -> None:
    """Every class and function in the new public module has a real docstring."""
    package_path = Path(dynamic_evaluation.__file__).parent
    missing = []
    for source_path in sorted(package_path.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        missing.extend(
            f"{source_path.name}:{node.name}@{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and ast.get_docstring(node) is None
        )
    assert missing == []


def test_dynamic_evaluation_contract_is_exported_from_measurement_package() -> None:
    """Consumers can discover the versioned contract through its owner package."""
    assert measurement.DYNAMIC_EVALUATION_ITEM_CONTRACT_ID == (
        "fast_mlsirm_dynamic_evaluation_item/v1"
    )
    assert measurement.DynamicEvaluationItemSnapshot is (
        dynamic_evaluation.DynamicEvaluationItemSnapshot
    )
    assert measurement.EvaluationCriterionSetSnapshot is (
        dynamic_evaluation.EvaluationCriterionSetSnapshot
    )
    assert measurement.EvaluationItemSetSnapshot is (
        dynamic_evaluation.EvaluationItemSetSnapshot
    )
