"""Fail-closed boundaries for dynamic evaluation item snapshots."""

from __future__ import annotations

from typing import Any

import pytest

from fast_mlsirm.measurement.dynamic_evaluation import (
    DynamicEvaluationContractError,
    DynamicEvaluationItemSnapshot,
    DynamicItemOrigin,
    EvaluationItemRole,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    build_dynamic_evaluation_item,
    build_evaluation_item_set_snapshot,
)


def _build(**overrides: Any) -> DynamicEvaluationItemSnapshot:
    """Build a valid generated candidate, replacing only requested fields."""
    payload: dict[str, Any] = {
        "item_instance_ref": "evaluation_item_alpha",
        "blueprint_revision_ref": "evaluation_blueprint_revision_1",
        "content_ref": "content_alpha",
        "content_sha256": "a" * 64,
        "origin": DynamicItemOrigin.GENERATED,
        "role": EvaluationItemRole.CANDIDATE,
        "reference_semantics": ReferenceSemantics.RUBRIC,
        "reference_status": ReferenceStatus.PROVISIONAL,
        "rubric_revision_ref": "rubric_revision_1",
        "criterion_refs": ("criterion_accuracy",),
        "provenance_refs": ("source_snapshot_1",),
        "generation_invocation_ref": "generation_invocation_1",
        "seed_ref": "seed_recorded_not_deterministic",
        "regeneration_status": RegenerationStatus.INPUTS_RECORDED,
        "regeneration_evidence_ref": None,
        "adjudication_ref": None,
        "validation_evidence_refs": (),
    }
    payload.update(overrides)
    return build_dynamic_evaluation_item(**payload)


def test_string_enum_values_are_admitted_but_unknown_or_foreign_values_fail() -> None:
    """Transport strings may be parsed only through the closed enum vocabulary."""
    item = _build(
        origin="generated",
        role="candidate",
        reference_semantics="rubric",
        reference_status="provisional",
        regeneration_status="inputs_recorded",
    )
    assert item.origin is DynamicItemOrigin.GENERATED

    for field in (
        "origin",
        "role",
        "reference_semantics",
        "reference_status",
        "regeneration_status",
    ):
        with pytest.raises(TypeError):
            _build(**{field: object()})

    with pytest.raises(DynamicEvaluationContractError) as caught:
        _build(origin="unknown_origin")
    assert caught.value.code == "invalid_enum_value"


@pytest.mark.parametrize(
    "invalid_ref",
    (
        "",
        " item_ref",
        "item_ref ",
        "\ufeffitem_ref",
        "item_ref\ufeff",
        "line\nbreak",
        "\ud800",
        "x" * 257,
    ),
)
def test_references_are_exact_bounded_unicode_scalars(invalid_ref: str) -> None:
    """Opaque identities reject aliases, controls, surrogates, and excess length."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        _build(item_instance_ref=invalid_ref)
    assert caught.value.code == "invalid_reference"

    with pytest.raises(TypeError, match="item_instance_ref must be a string"):
        _build(item_instance_ref=object())


@pytest.mark.parametrize(
    ("criteria", "expected_code"),
    (
        ((), "invalid_reference_count"),
        (("criterion_accuracy", "criterion_accuracy"), "duplicate_reference"),
        (tuple(f"criterion_{index}" for index in range(129)), "invalid_reference_count"),
    ),
)
def test_reference_collections_are_typed_bounded_and_unique(
    criteria: tuple[str, ...], expected_code: str
) -> None:
    """Criterion collections cannot be empty, duplicate, or exceed their budget."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        _build(criterion_refs=criteria)
    assert caught.value.code == expected_code

    with pytest.raises(TypeError, match="criterion_refs must be a tuple or list"):
        _build(criterion_refs="criterion_accuracy")


def test_generation_identity_matches_the_item_origin() -> None:
    """Generated origins require invocation provenance and authored origins reject it."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        _build(generation_invocation_ref=None)
    assert caught.value.code == "generated_item_requires_invocation"

    for origin in (DynamicItemOrigin.AUTHORED, DynamicItemOrigin.PRODUCTION_SAMPLE):
        with pytest.raises(DynamicEvaluationContractError) as caught:
            _build(origin=origin)
        assert caught.value.code == "unexpected_generation_invocation"


def test_reference_decision_evidence_is_state_specific() -> None:
    """Adjudication, validation, and invalidation remain separate evidence states."""
    with pytest.raises(DynamicEvaluationContractError) as caught:
        _build(validation_evidence_refs=("validation_evidence_1",))
    assert caught.value.code == "unexpected_validation_evidence"

    validated = _build(
        reference_status=ReferenceStatus.VALIDATED,
        adjudication_ref="adjudication_resolution_1",
        validation_evidence_refs=("validation_evidence_1",),
    )
    assert validated.adjudication_ref == "adjudication_resolution_1"


def test_digest_carrier_and_item_set_resource_limits_fail_closed() -> None:
    """Malformed digests and oversized or untyped item sets emit no snapshot."""
    with pytest.raises(TypeError, match="content_sha256 must be a string"):
        _build(content_sha256=object())

    for invalid_items in ((), [], "not-an-item-set"):
        with pytest.raises(DynamicEvaluationContractError) as caught:
            build_evaluation_item_set_snapshot(
                run_snapshot_ref="evaluation_run_snapshot_empty",
                blueprint_revision_ref="evaluation_blueprint_revision_1",
                items=invalid_items,  # type: ignore[arg-type]
                linking_status=LinkingStatus.UNAVAILABLE,
            )
        assert caught.value.code == "invalid_item_set"

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_too_large",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=[_build()] * 10_001,
            linking_status=LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_set_budget_exceeded"

    with pytest.raises(TypeError, match="exact DynamicEvaluationItemSnapshot"):
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_wrong_type",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_build(), object()),  # type: ignore[arg-type]
            linking_status=LinkingStatus.UNAVAILABLE,
        )


def test_linked_status_requires_validated_anchor_and_linking_evidence() -> None:
    """Cross-version comparability needs both an anchor and independent evidence."""
    anchor = _build(
        role=EvaluationItemRole.ANCHOR,
        reference_status=ReferenceStatus.VALIDATED,
        validation_evidence_refs=("validation_evidence_1",),
    )
    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_anchor_no_evidence",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(anchor,),
            linking_status=LinkingStatus.LINKED,
        )
    assert caught.value.code == "linked_snapshot_requires_evidence"

    linked = build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_linked",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(anchor,),
        linking_status="linked",
        linking_evidence_ref="linking_evidence_1",
    )
    assert linked.anchor_item_refs == ("evaluation_item_alpha",)

    with pytest.raises(DynamicEvaluationContractError) as caught:
        build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_unlinked_evidence",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_build(),),
            linking_status=LinkingStatus.WITHIN_RUN_ONLY,
            linking_evidence_ref="linking_evidence_1",
        )
    assert caught.value.code == "unexpected_linking_evidence"
