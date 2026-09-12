"""Canonical membership semantics for dynamic-evaluation evidence references."""

from __future__ import annotations

from fast_mlsirm.measurement.dynamic_evaluation import (
    DynamicItemOrigin,
    EvaluationItemRole,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    build_dynamic_evaluation_item,
    build_evaluation_category_definition,
    build_evaluation_criterion_definition,
    build_evaluation_criterion_set_snapshot,
)


def _criterion_set():  # type: ignore[no-untyped-def]
    """Build one immutable criterion set shared by evidence-order regressions."""
    category = build_evaluation_category_definition(
        category_ref="category_satisfied",
        definition_ref="definition_category_satisfied_1",
        definition_sha256="a" * 64,
    )
    criterion = build_evaluation_criterion_definition(
        criterion_ref="criterion_accuracy",
        criterion_revision_ref="criterion_accuracy_revision_1",
        definition_ref="definition_criterion_accuracy_1",
        definition_sha256="b" * 64,
        admissible_evidence_rule_ref="admissible_evidence_accuracy_1",
        admissible_evidence_rule_sha256="c" * 64,
        exclusion_rule_ref="exclusion_accuracy_1",
        exclusion_rule_sha256="d" * 64,
        response_semantics_ref="response_semantics_accuracy_1",
        response_semantics_sha256="e" * 64,
        abstention_rule_ref="abstention_accuracy_1",
        abstention_rule_sha256="f" * 64,
        not_observable_rule_ref="not_observable_accuracy_1",
        not_observable_rule_sha256="1" * 64,
        category_definitions=(category,),
    )
    return build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_accuracy_1",
        construct_ref="construct_accuracy_1",
        population_scope_ref="population_scope_1",
        language_scope_ref="language_scope_1",
        domain_scope_ref="domain_scope_1",
        criteria=(criterion,),
    )


def _validated_item(
    *,
    provenance_refs: tuple[str, ...],
    validation_evidence_refs: tuple[str, ...],
):  # type: ignore[no-untyped-def]
    """Build one otherwise-identical validated item from evidence memberships."""
    return build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_alpha",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_evaluation_item_alpha",
        content_sha256="2" * 64,
        origin=DynamicItemOrigin.AUTHORED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=ReferenceStatus.VALIDATED,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=_criterion_set(),
        criterion_refs=("criterion_accuracy",),
        provenance_refs=provenance_refs,
        generation_invocation_ref=None,
        regeneration_status=RegenerationStatus.UNAVAILABLE,
        validation_evidence_refs=validation_evidence_refs,
    )


def test_provenance_reference_membership_is_order_invariant() -> None:
    """Equivalent provenance membership has one item identity regardless of order."""
    reverse_order = _validated_item(
        provenance_refs=("source_beta", "source_alpha"),
        validation_evidence_refs=("validation_alpha", "validation_beta"),
    )
    canonical_order = _validated_item(
        provenance_refs=("source_alpha", "source_beta"),
        validation_evidence_refs=("validation_alpha", "validation_beta"),
    )

    assert reverse_order.snapshot_sha256 == canonical_order.snapshot_sha256
    assert reverse_order.provenance_refs == ("source_alpha", "source_beta")


def test_validation_evidence_membership_is_order_invariant() -> None:
    """Equivalent validation evidence has one item identity regardless of order."""
    reverse_order = _validated_item(
        provenance_refs=("source_alpha", "source_beta"),
        validation_evidence_refs=("validation_beta", "validation_alpha"),
    )
    canonical_order = _validated_item(
        provenance_refs=("source_alpha", "source_beta"),
        validation_evidence_refs=("validation_alpha", "validation_beta"),
    )

    assert reverse_order.snapshot_sha256 == canonical_order.snapshot_sha256
    assert reverse_order.validation_evidence_refs == (
        "validation_alpha",
        "validation_beta",
    )
