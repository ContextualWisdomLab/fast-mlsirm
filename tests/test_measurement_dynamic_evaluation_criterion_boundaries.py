"""Integrity and resource boundaries for dynamic criterion snapshots."""

from __future__ import annotations

import pytest

from fast_mlsirm.measurement.dynamic_evaluation import (
    DynamicEvaluationContractError,
    DynamicItemOrigin,
    EvaluationCriterionDefinition,
    EvaluationCriterionSetSnapshot,
    EvaluationItemRole,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    build_dynamic_evaluation_item,
    build_evaluation_criterion_definition,
    build_evaluation_criterion_set_snapshot,
    build_evaluation_item_set_snapshot,
)


def _criterion(
    *,
    criterion_ref: str = "criterion_evidence_support",
    criterion_revision_ref: str = "criterion_evidence_support_revision_1",
    marker: str = "c",
) -> EvaluationCriterionDefinition:
    """Build one explicit criterion through the public admission boundary."""
    return build_evaluation_criterion_definition(
        criterion_ref=criterion_ref,
        criterion_revision_ref=criterion_revision_ref,
        definition_ref=f"definition_{criterion_ref}",
        definition_sha256=marker * 64,
        admissible_evidence_rule_ref=f"admissible_{criterion_ref}",
        exclusion_rule_ref=f"exclusion_{criterion_ref}",
        response_semantics_ref=f"semantics_{criterion_ref}",
        abstention_rule_ref=f"abstention_{criterion_ref}",
        not_observable_rule_ref=f"not_observable_{criterion_ref}",
    )


def _criterion_set() -> EvaluationCriterionSetSnapshot:
    """Build one criterion set for run-integrity tests."""
    return build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_model_response_evaluation_1",
        construct_ref="construct_evidence_grounding_1",
        population_scope_ref="population_scope_enterprise_responses_1",
        language_scope_ref="language_scope_multilingual_1",
        domain_scope_ref="domain_scope_enterprise_qa_1",
        criteria=(_criterion(),),
    )


def _item():  # type: ignore[no-untyped-def]
    """Build one item bound to the fixture criterion identity."""
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
        criterion_refs=("criterion_evidence_support",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=RegenerationStatus.INPUTS_RECORDED,
    )


def _criterion_set_kwargs() -> dict[str, object]:
    """Return reusable criterion-set arguments for boundary mutations."""
    return {
        "criterion_set_snapshot_ref": "criterion_set_snapshot_1",
        "criterion_set_revision_ref": "criterion_set_revision_1",
        "blueprint_revision_ref": "evaluation_blueprint_revision_1",
        "rubric_revision_ref": "rubric_revision_1",
        "intended_use_ref": "intended_use_model_response_evaluation_1",
        "construct_ref": "construct_evidence_grounding_1",
        "population_scope_ref": "population_scope_enterprise_responses_1",
        "language_scope_ref": "language_scope_multilingual_1",
        "domain_scope_ref": "domain_scope_enterprise_qa_1",
    }


def test_criterion_factories_are_sealed_and_detect_mutation() -> None:
    """Criterion meaning cannot bypass admission or change after admission."""
    with pytest.raises(ValueError, match="build_evaluation_criterion_definition"):
        EvaluationCriterionDefinition(  # type: ignore[call-arg]
            criterion_ref="criterion_evidence_support",
            criterion_revision_ref="criterion_revision_1",
            definition_ref="definition_1",
            definition_sha256="c" * 64,
            admissible_evidence_rule_ref="admissible_1",
            exclusion_rule_ref="exclusion_1",
            response_semantics_ref="semantics_1",
            abstention_rule_ref="abstention_1",
            not_observable_rule_ref="not_observable_1",
        )

    admitted = _criterion()
    assert admitted.snapshot_sha256 == admitted.to_dict()["snapshot_sha256"]
    object.__setattr__(admitted, "definition_ref", "changed_definition")
    with pytest.raises(DynamicEvaluationContractError) as caught:
        admitted.to_dict()
    assert caught.value.code == "criterion_definition_integrity_mismatch"

    malformed = _criterion(marker="d")
    object.__setattr__(malformed, "definition_sha256", object())
    with pytest.raises(DynamicEvaluationContractError) as caught:
        malformed.to_dict()
    assert caught.value.code == "criterion_definition_integrity_mismatch"

    with pytest.raises(ValueError, match="build_evaluation_criterion_set_snapshot"):
        EvaluationCriterionSetSnapshot(  # type: ignore[call-arg]
            **_criterion_set_kwargs(),
            criteria=(_criterion(),),
        )


def test_criterion_set_rejects_budget_type_and_duplicate_identity_defects() -> None:
    """Criterion-set resource and identity defects fail before run admission."""
    too_many = tuple(
        _criterion(
            criterion_ref=f"criterion_{index}",
            criterion_revision_ref=f"criterion_revision_{index}",
            marker=f"{index % 16:x}",
        )
        for index in range(129)
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=too_many,
        )
    assert caught.value.code == "criterion_set_budget_exceeded"

    with pytest.raises(TypeError, match="exact EvaluationCriterionDefinition"):
        build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(_criterion(), object()),  # type: ignore[arg-type]
        )

    duplicate_ref = _criterion(
        criterion_revision_ref="criterion_other_revision",
        marker="e",
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(_criterion(), duplicate_ref),
        )
    assert caught.value.code == "duplicate_criterion_definition"

    first = _criterion()
    duplicate_revision = _criterion(
        criterion_ref="criterion_safety",
        criterion_revision_ref=first.criterion_revision_ref,
        marker="f",
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(first, duplicate_revision),
        )
    assert caught.value.code == "duplicate_criterion_revision"


def test_nested_criterion_and_foreign_value_invalidate_the_set() -> None:
    """Nested mutation or foreign collection values invalidate the set digest."""
    criterion_set = _criterion_set()
    assert criterion_set.criterion_refs == ("criterion_evidence_support",)
    assert criterion_set.snapshot_sha256 == (
        criterion_set.to_dict()["snapshot_sha256"]
    )

    nested = criterion_set.criteria[0]
    object.__setattr__(nested, "definition_ref", "changed_definition")
    with pytest.raises(DynamicEvaluationContractError) as caught:
        criterion_set.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"

    foreign = _criterion_set()
    object.__setattr__(foreign, "criteria", (object(),))
    with pytest.raises(DynamicEvaluationContractError) as caught:
        foreign.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"


def test_run_properties_preserve_binding_and_detect_substitution() -> None:
    """Run accessors expose the exact criteria and reject substituted objects."""
    criterion_set = _criterion_set()
    snapshot = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_properties",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=criterion_set,
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    assert snapshot.criterion_set_snapshot_ref == (
        criterion_set.criterion_set_snapshot_ref
    )
    assert snapshot.criterion_set_sha256 == criterion_set.snapshot_sha256
    assert snapshot.criterion_refs == ("criterion_evidence_support",)
    assert snapshot.snapshot_sha256 == snapshot.to_dict()["snapshot_sha256"]

    wrong_set = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_wrong_set",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=_criterion_set(),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(wrong_set, "criterion_set_snapshot", object())
    with pytest.raises(DynamicEvaluationContractError) as caught:
        wrong_set.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"

    wrong_items = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_wrong_items",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=_criterion_set(),
        linking_status=LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(wrong_items, "items", (object(),))
    with pytest.raises(DynamicEvaluationContractError) as caught:
        wrong_items.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"

    with pytest.raises(TypeError, match="exact EvaluationCriterionSetSnapshot"):
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_wrong_set_type",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            criterion_set_snapshot=object(),  # type: ignore[arg-type]
            linking_status=LinkingStatus.UNAVAILABLE,
        )
