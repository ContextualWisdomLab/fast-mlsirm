"""Contracts for dynamic evaluation items without a required fixed item set."""

from __future__ import annotations

import pytest

from fast_mlsirm.measurement.dynamic_evaluation import (
    DYNAMIC_EVALUATION_ITEM_CONTRACT_ID,
    DynamicEvaluationContractError,
    DynamicEvaluationItemSnapshot,
    DynamicItemOrigin,
    EvaluationItemRole,
    EvaluationItemSetSnapshot,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    build_dynamic_evaluation_item,
    build_evaluation_item_set_snapshot,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _item(
    *,
    item_instance_ref: str = "evaluation_item_alpha",
    role: EvaluationItemRole = EvaluationItemRole.CANDIDATE,
    reference_status: ReferenceStatus = ReferenceStatus.PROVISIONAL,
    adjudication_ref: str | None = None,
    validation_evidence_refs: tuple[str, ...] = (),
    regeneration_status: RegenerationStatus = RegenerationStatus.INPUTS_RECORDED,
    regeneration_evidence_ref: str | None = None,
) -> DynamicEvaluationItemSnapshot:
    """Build one generated item through the public admission boundary."""
    return build_dynamic_evaluation_item(
        item_instance_ref=item_instance_ref,
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref=f"content_{item_instance_ref}",
        content_sha256=_SHA_A if item_instance_ref.endswith("alpha") else _SHA_B,
        origin=DynamicItemOrigin.GENERATED,
        role=role,
        reference_semantics=ReferenceSemantics.RUBRIC,
        reference_status=reference_status,
        rubric_revision_ref="rubric_revision_1",
        criterion_refs=("criterion_accuracy", "criterion_evidence"),
        provenance_refs=("source_snapshot_1", "generation_invocation_1"),
        generation_invocation_ref="generation_invocation_1",
        seed_ref="seed_recorded_not_deterministic",
        regeneration_status=regeneration_status,
        regeneration_evidence_ref=regeneration_evidence_ref,
        adjudication_ref=adjudication_ref,
        validation_evidence_refs=validation_evidence_refs,
    )


def test_zero_anchor_snapshot_is_valid_but_cross_version_linking_is_unavailable() -> None:
    """Cold-start evaluation may run without anchors but cannot claim linked scores."""
    snapshot = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        linking_status=LinkingStatus.UNAVAILABLE,
    )

    assert snapshot.contract_id == DYNAMIC_EVALUATION_ITEM_CONTRACT_ID
    assert snapshot.anchor_item_refs == ()
    assert snapshot.linking_status is LinkingStatus.UNAVAILABLE
    assert snapshot.linking_evidence_ref is None

    within_run = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_2",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        linking_status=LinkingStatus.WITHIN_RUN_ONLY,
    )
    assert within_run.linking_status is LinkingStatus.WITHIN_RUN_ONLY

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_3",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            linking_status=LinkingStatus.LINKED,
            linking_evidence_ref="linking_evidence_1",
        )
    assert caught.value.code == "linked_snapshot_requires_anchor"


def test_adjudicated_is_not_validated_or_anchor_authority() -> None:
    """Adjudication provenance does not silently promote an item to a validated anchor."""
    item = _item(
        reference_status=ReferenceStatus.ADJUDICATED,
        adjudication_ref="adjudication_resolution_1",
    )
    assert item.reference_status is ReferenceStatus.ADJUDICATED
    assert item.role is EvaluationItemRole.CANDIDATE

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(
            role=EvaluationItemRole.ANCHOR,
            reference_status=ReferenceStatus.ADJUDICATED,
            adjudication_ref="adjudication_resolution_1",
        )
    assert caught.value.code == "anchor_requires_validated_reference"

    anchor = _item(
        role=EvaluationItemRole.ANCHOR,
        reference_status=ReferenceStatus.VALIDATED,
        validation_evidence_refs=("validation_evidence_1",),
    )
    assert anchor.role is EvaluationItemRole.ANCHOR
    assert anchor.reference_status is ReferenceStatus.VALIDATED


def test_reference_status_controls_adjudication_and_validation_evidence() -> None:
    """Each reference status carries only the evidence appropriate to that state."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(reference_status=ReferenceStatus.ADJUDICATED)
    assert caught.value.code == "adjudicated_reference_requires_resolution"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(
            reference_status=ReferenceStatus.PROVISIONAL,
            adjudication_ref="adjudication_resolution_1",
        )
    assert caught.value.code == "unexpected_adjudication_resolution"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(reference_status=ReferenceStatus.VALIDATED)
    assert caught.value.code == "validated_reference_requires_evidence"

    invalidated = _item(
        reference_status=ReferenceStatus.INVALIDATED,
        validation_evidence_refs=("invalidation_evidence_1",),
    )
    assert invalidated.reference_status is ReferenceStatus.INVALIDATED


def test_seed_and_recorded_inputs_do_not_claim_deterministic_regeneration() -> None:
    """A seed is provenance only until reproducibility has separate validation evidence."""
    item = _item(regeneration_status=RegenerationStatus.INPUTS_RECORDED)
    assert item.seed_ref == "seed_recorded_not_deterministic"
    assert item.regeneration_status is RegenerationStatus.INPUTS_RECORDED
    assert item.regeneration_evidence_ref is None

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(regeneration_status=RegenerationStatus.VERIFIED)
    assert caught.value.code == "verified_regeneration_requires_evidence"

    verified = _item(
        regeneration_status=RegenerationStatus.VERIFIED,
        regeneration_evidence_ref="regeneration_validation_1",
    )
    assert verified.regeneration_status is RegenerationStatus.VERIFIED

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _item(
            regeneration_status=RegenerationStatus.UNAVAILABLE,
            regeneration_evidence_ref="regeneration_validation_1",
        )
    assert caught.value.code == "unexpected_regeneration_evidence"


def test_generated_items_require_generation_identity_and_exact_content_digest() -> None:
    """Generated content is frozen by exact invocation and content identities."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_alpha",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_alpha",
            content_sha256="not-a-digest",
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
    assert caught.value.code == "invalid_sha256"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_alpha",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_alpha",
            content_sha256=_SHA_A,
            origin=DynamicItemOrigin.GENERATED,
            role=EvaluationItemRole.CANDIDATE,
            reference_semantics=ReferenceSemantics.RUBRIC,
            reference_status=ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_refs=("criterion_accuracy",),
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref=None,
            regeneration_status=RegenerationStatus.INPUTS_RECORDED,
        )
    assert caught.value.code == "generated_item_requires_invocation"


def test_item_set_is_immutable_unique_and_blueprint_consistent() -> None:
    """A run freezes one concrete unique item set under one blueprint revision."""
    first = _item()
    second = _item(item_instance_ref="evaluation_item_beta")
    snapshot = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=[first, second],
        linking_status=LinkingStatus.WITHIN_RUN_ONLY,
    )
    assert snapshot.items == (first, second)
    assert snapshot.to_dict()["items"][0]["content_sha256"] == _SHA_A

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_duplicate",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(first, first),
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "duplicate_item_instance"

    foreign = build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_foreign",
        blueprint_revision_ref="evaluation_blueprint_revision_2",
        content_ref="content_foreign",
        content_sha256="c" * 64,
        origin=DynamicItemOrigin.AUTHORED,
        role=EvaluationItemRole.CANDIDATE,
        reference_semantics=ReferenceSemantics.EXACT,
        reference_status=ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_2",
        criterion_refs=("criterion_accuracy",),
        provenance_refs=("authoring_record_1",),
        generation_invocation_ref=None,
        regeneration_status=RegenerationStatus.UNAVAILABLE,
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_foreign",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(first, foreign),
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_blueprint_mismatch"


def test_public_aggregates_are_factory_sealed() -> None:
    """Callers cannot instantiate admitted domain values without replaying invariants."""
    with pytest.raises(ValueError, match="build_dynamic_evaluation_item"):
        DynamicEvaluationItemSnapshot(  # type: ignore[call-arg]
            item_instance_ref="evaluation_item_alpha",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_alpha",
            content_sha256=_SHA_A,
            origin=DynamicItemOrigin.GENERATED,
            role=EvaluationItemRole.CANDIDATE,
            reference_semantics=ReferenceSemantics.RUBRIC,
            reference_status=ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_refs=("criterion_accuracy",),
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            seed_ref=None,
            regeneration_status=RegenerationStatus.INPUTS_RECORDED,
            regeneration_evidence_ref=None,
            adjudication_ref=None,
            validation_evidence_refs=(),
        )

    with pytest.raises(ValueError, match="build_evaluation_item_set_snapshot"):
        EvaluationItemSetSnapshot(  # type: ignore[call-arg]
            run_snapshot_ref="evaluation_run_snapshot_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            linking_status=LinkingStatus.UNAVAILABLE,
            linking_evidence_ref=None,
        )
