"""Canonical set semantics for dynamic-evaluation run snapshots."""

from __future__ import annotations

from fast_mlsirm.measurement.dynamic_evaluation import (
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


def _criterion_set():  # type: ignore[no-untyped-def]
    """Build one immutable criterion set shared by both orderings."""
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


def _item(item_instance_ref: str):  # type: ignore[no-untyped-def]
    """Build one valid item whose identity alone differs across the set."""
    return build_dynamic_evaluation_item(
        item_instance_ref=item_instance_ref,
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref=f"content_{item_instance_ref}",
        content_sha256=("2" if item_instance_ref.endswith("alpha") else "3") * 64,
        origin=DynamicItemOrigin.AUTHORED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=_criterion_set(),
        criterion_refs=("criterion_accuracy",),
        provenance_refs=(f"source_{item_instance_ref}",),
        generation_invocation_ref=None,
        regeneration_status=RegenerationStatus.UNAVAILABLE,
    )


def test_item_set_snapshot_is_order_invariant() -> None:
    """A mathematical item set has one digest regardless of caller ordering."""
    criterion_set = _criterion_set()
    alpha = _item("evaluation_item_alpha")
    beta = _item("evaluation_item_beta")

    reverse_order = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        criterion_set_snapshot=criterion_set,
        items=(beta, alpha),
        linking_status=LinkingStatus.WITHIN_RUN_ONLY,
    )
    canonical_order = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        criterion_set_snapshot=criterion_set,
        items=(alpha, beta),
        linking_status=LinkingStatus.WITHIN_RUN_ONLY,
    )

    assert reverse_order.snapshot_sha256 == canonical_order.snapshot_sha256
    assert tuple(item.item_instance_ref for item in reverse_order.items) == (
        "evaluation_item_alpha",
        "evaluation_item_beta",
    )
