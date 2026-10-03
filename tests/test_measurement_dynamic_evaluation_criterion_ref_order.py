"""Canonical criterion-membership semantics for dynamic evaluation items."""

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


def _criterion(criterion_ref: str, marker: str):  # type: ignore[no-untyped-def]
    """Build one content-addressed synthetic criterion for identity testing."""
    category = build_evaluation_category_definition(
        category_ref=f"{criterion_ref}_satisfied",
        definition_ref=f"definition_{criterion_ref}_satisfied_1",
        definition_sha256=marker * 64,
    )
    return build_evaluation_criterion_definition(
        criterion_ref=criterion_ref,
        criterion_revision_ref=f"{criterion_ref}_revision_1",
        definition_ref=f"definition_{criterion_ref}_1",
        definition_sha256=marker * 64,
        admissible_evidence_rule_ref=f"admissible_evidence_{criterion_ref}_1",
        admissible_evidence_rule_sha256="1" * 64,
        exclusion_rule_ref=f"exclusion_{criterion_ref}_1",
        exclusion_rule_sha256="2" * 64,
        response_semantics_ref=f"response_semantics_{criterion_ref}_1",
        response_semantics_sha256="3" * 64,
        abstention_rule_ref=f"abstention_{criterion_ref}_1",
        abstention_rule_sha256="4" * 64,
        not_observable_rule_ref=f"not_observable_{criterion_ref}_1",
        not_observable_rule_sha256="5" * 64,
        category_definitions=(category,),
    )


def _criterion_set():  # type: ignore[no-untyped-def]
    """Build the same immutable two-criterion set for both item admissions."""
    return build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_quality_1",
        construct_ref="construct_quality_1",
        population_scope_ref="population_scope_1",
        language_scope_ref="language_scope_1",
        domain_scope_ref="domain_scope_1",
        criteria=(
            _criterion("criterion_accuracy", "a"),
            _criterion("criterion_evidence", "b"),
        ),
    )


def _item(criterion_refs: tuple[str, ...]):  # type: ignore[no-untyped-def]
    """Build one otherwise-identical item with caller-controlled criterion order."""
    return build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_alpha",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_alpha",
        content_sha256="c" * 64,
        origin=DynamicItemOrigin.AUTHORED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=_criterion_set(),
        criterion_refs=criterion_refs,
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref=None,
        regeneration_status=RegenerationStatus.UNAVAILABLE,
    )


def test_item_criterion_membership_is_order_invariant() -> None:
    """Criterion membership has one item identity regardless of caller ordering."""
    reverse_order = _item(("criterion_evidence", "criterion_accuracy"))
    canonical_order = _item(("criterion_accuracy", "criterion_evidence"))

    assert reverse_order.snapshot_sha256 == canonical_order.snapshot_sha256
    assert reverse_order.criterion_refs == (
        "criterion_accuracy",
        "criterion_evidence",
    )
