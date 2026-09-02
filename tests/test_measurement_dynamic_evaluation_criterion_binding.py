"""Criterion-set admission contracts for dynamic evaluation runs."""

from __future__ import annotations

import pytest

from fast_mlsirm.measurement.dynamic_evaluation import (
    DynamicEvaluationContractError,
    DynamicItemOrigin,
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


def _criterion_set():  # type: ignore[no-untyped-def]
    """Build an immutable criterion set with explicit evidence semantics."""
    criterion = build_evaluation_criterion_definition(
        criterion_ref="criterion_evidence_support",
        criterion_revision_ref="criterion_evidence_support_revision_1",
        definition_ref="criterion_definition_artifact_1",
        definition_sha256="c" * 64,
        admissible_evidence_rule_ref="evidence_rule_supported_claim_1",
        exclusion_rule_ref="exclusion_rule_unsupported_claim_1",
        response_semantics_ref="response_semantics_supported_not_supported_1",
        abstention_rule_ref="abstention_rule_insufficient_evidence_1",
        not_observable_rule_ref="not_observable_rule_missing_source_1",
    )
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
        criteria=(criterion,),
    )


def _item(*, criterion_refs=("criterion_evidence_support",)):  # type: ignore[no-untyped-def]
    """Build one candidate item whose criteria must resolve in the run set."""
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
        criterion_refs=criterion_refs,
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=RegenerationStatus.INPUTS_RECORDED,
    )


def test_run_snapshot_requires_an_explicit_nonempty_criterion_set() -> None:
    """A concrete item set is not evaluable until its criteria are frozen."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            criterion_set_snapshot=None,
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_required"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_criterion_set_snapshot(
            criterion_set_snapshot_ref="criterion_set_snapshot_empty",
            criterion_set_revision_ref="criterion_set_revision_empty",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            rubric_revision_ref="rubric_revision_1",
            intended_use_ref="intended_use_model_response_evaluation_1",
            construct_ref="construct_evidence_grounding_1",
            population_scope_ref="population_scope_enterprise_responses_1",
            language_scope_ref="language_scope_multilingual_1",
            domain_scope_ref="domain_scope_enterprise_qa_1",
            criteria=(),
        )
    assert caught.value.code == "invalid_criterion_set"


def test_item_criteria_and_rubric_must_resolve_in_the_frozen_criterion_set() -> None:
    """An evaluator cannot invent criteria or silently switch rubric revisions."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_unknown_criterion",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(criterion_refs=("criterion_unregistered",)),),
            criterion_set_snapshot=_criterion_set(),
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_criterion_not_registered"

    foreign_rubric = build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_foreign_rubric",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_foreign_rubric",
        content_sha256="b" * 64,
        origin=DynamicItemOrigin.AUTHORED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_2",
        criterion_refs=("criterion_evidence_support",),
        provenance_refs=("authoring_record_1",),
        generation_invocation_ref=None,
        regeneration_status=RegenerationStatus.UNAVAILABLE,
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_foreign_rubric",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(foreign_rubric,),
            criterion_set_snapshot=_criterion_set(),
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_rubric_mismatch"


def test_run_snapshot_publishes_exact_criterion_identity_and_digest() -> None:
    """Every evaluation artifact identifies the criteria actually administered."""
    criterion_set = _criterion_set()
    snapshot = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=criterion_set,
        linking_status=LinkingStatus.WITHIN_RUN_ONLY,
    )

    payload = snapshot.to_dict()
    assert payload["criterion_set_snapshot_ref"] == criterion_set.criterion_set_snapshot_ref
    assert payload["criterion_set_sha256"] == criterion_set.snapshot_sha256
    assert payload["criterion_refs"] == ["criterion_evidence_support"]
    assert payload["criterion_set"]["criteria"][0]["admissible_evidence_rule_ref"]
    assert payload["criterion_set"]["criteria"][0]["abstention_rule_ref"]


def test_criterion_set_is_factory_sealed_and_mutation_detected() -> None:
    """Criterion meaning cannot change after a run binds to its digest."""
    criterion_set = _criterion_set()
    object.__setattr__(criterion_set, "rubric_revision_ref", "rubric_revision_2")

    with pytest.raises(DynamicEvaluationContractError) as caught:
        criterion_set.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_mutated_criteria",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            criterion_set_snapshot=criterion_set,
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_integrity_mismatch"
