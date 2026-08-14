"""Boundary tests for the evidence-gated rubric item-bank lifecycle."""

from __future__ import annotations

from dataclasses import replace
import pytest

from fast_mlsirm.rubric import (
    CandidateLifecycleState,
    PilotAdmissionError,
    ResponseFormat,
    audit_generated_item_candidate,
)
from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleRecord,
    ItemBankLifecycleState,
    PolicyCriticality,
    build_item_bank_pilot_record,
    transition_item_bank_record,
)
import fast_mlsirm.rubric.item_bank as item_bank
import fast_mlsirm.rubric.audit as audit_module

from test_rubric_item_bank_lifecycle import (
    _AUDIT_FIXTURES,
    _calibrated_record,
    _evidence,
    _pilot_record,
)


def _raw_record(**overrides: object) -> ItemBankLifecycleRecord:
    """Build a factory-token record for direct invariant coverage."""
    pilot = _pilot_record()
    values: dict[str, object] = {
        "item_id": pilot.item_id,
        "item_version": "1.0.0",
        "candidate_fingerprint": pilot.candidate_fingerprint,
        "pilot_record_fingerprint": pilot.pilot_record_fingerprint,
        "audit_report_fingerprint": pilot.audit_report_fingerprint,
        "blueprint_id": pilot.blueprint_id,
        "rubric_id": pilot.rubric_id,
        "rubric_version": pilot.rubric_version,
        "lifecycle_state": ItemBankLifecycleState.PILOTING,
        "policy_criticality": PolicyCriticality.ORDINARY,
        "approved_use_ids": (),
        "evidence_references": (),
        "previous_record_fingerprint": None,
        "transition_reason_id": "pilot_admission",
        "_creation_token": item_bank._RECORD_CREATION_TOKEN,
    }
    values.update(overrides)
    return ItemBankLifecycleRecord(**values)


def test_reference_and_normalization_failures_are_stable() -> None:
    """Invalid evidence identities and collection shapes never bypass the gate."""
    with pytest.raises(ValueError, match="redacted JSON-style"):
        item_bank._lifecycle_path("$.not valid")
    with pytest.raises(ValueError, match="64 lower hexadecimal"):
        item_bank._fingerprint("not-a-digest", "digest")
    with pytest.raises(ValueError, match="one of"):
        ItemBankEvidenceReference("unknown", "evidence_id", "a" * 64)

    with pytest.raises(ItemBankLifecycleError) as lifecycle_error:
        item_bank._normalize_evidence_references(object(), error_type=ItemBankLifecycleError)
    assert lifecycle_error.value.code == "invalid_evidence_references"
    with pytest.raises(ValueError, match="must be a collection"):
        item_bank._normalize_evidence_references(object())
    with pytest.raises(ValueError, match="ItemBankEvidenceReference"):
        item_bank._normalize_evidence_references((object(),))

    reference = _evidence(ItemBankEvidenceKind.CALIBRATION, "same")
    with pytest.raises(ItemBankLifecycleError) as duplicate_error:
        item_bank._normalize_evidence_references(
            (reference, reference), error_type=ItemBankLifecycleError
        )
    assert duplicate_error.value.code == "duplicate_evidence_reference"
    with pytest.raises(ValueError, match="must not contain duplicates"):
        item_bank._normalize_evidence_references((reference, reference))

    conflicting = (
        reference,
        _evidence(ItemBankEvidenceKind.ITEM_FIT, "same", fingerprint_character="b"),
    )
    with pytest.raises(ItemBankLifecycleError) as conflict_error:
        item_bank._normalize_evidence_references(
            conflicting, error_type=ItemBankLifecycleError
        )
    assert conflict_error.value.code == "conflicting_evidence_identity"
    with pytest.raises(ValueError, match="multiple fingerprints"):
        item_bank._normalize_evidence_references(conflicting)


def test_record_invariants_cover_piloting_and_post_pilot_states() -> None:
    """Factory-token records still enforce every lifecycle-state invariant."""
    fingerprint = "f" * 64
    for overrides, message in (
        ({"previous_record_fingerprint": fingerprint}, "previous record"),
        (
            {"evidence_references": (_evidence(ItemBankEvidenceKind.CALIBRATION, "evidence"),)},
            "lifecycle evidence",
        ),
        ({"approved_use_ids": ("production_use",)}, "approved uses"),
        ({"transition_reason_id": "wrong_reason"}, "pilot_admission"),
    ):
        with pytest.raises(ValueError, match=message):
            _raw_record(**overrides)
    with pytest.raises(ValueError, match="previous record fingerprint"):
        _raw_record(
            lifecycle_state=ItemBankLifecycleState.CALIBRATED,
            transition_reason_id="calibration_completed",
        )
    with pytest.raises(ValueError, match="approved use identifiers"):
        _raw_record(
            lifecycle_state=ItemBankLifecycleState.APPROVED,
            previous_record_fingerprint=fingerprint,
            transition_reason_id="governance_approval",
        )
    core_pilot = _pilot_record()._core_record()
    with pytest.raises(ValueError, match="lifecycle_state='pilot'"):
        replace(core_pilot, lifecycle_state=CandidateLifecycleState.DRAFT)


def test_public_transition_boundaries_cover_type_state_and_scope_errors() -> None:
    """Transitions reject forged records, invalid targets, and immutable-scope edits."""
    pilot = build_item_bank_pilot_record(_pilot_record(), item_version="1.0.0")
    with pytest.raises(ItemBankLifecycleError, match="current record"):
        transition_item_bank_record(
            object(), ItemBankLifecycleState.CALIBRATED, evidence_references=(), transition_reason_id="bad_transition"
        )
    with pytest.raises(TypeError, match="exact PilotCandidateRecord"):
        build_item_bank_pilot_record(object(), item_version="1.0.0")
    invalid_state = _pilot_record()
    object.__setattr__(invalid_state, "lifecycle_state", CandidateLifecycleState.AUDITED)
    with pytest.raises(ItemBankLifecycleError, match="verified pilot"):
        build_item_bank_pilot_record(invalid_state, item_version="1.0.0")
    with pytest.raises(ItemBankLifecycleError, match="not supported"):
        transition_item_bank_record(
            pilot, "unknown_state", evidence_references=(), transition_reason_id="bad_transition"
        )

    calibrated = _calibrated_record()
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(_evidence(ItemBankEvidenceKind.APPROVAL, "approval", fingerprint_character="e"),),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_use",),
    )
    with pytest.raises(ItemBankLifecycleError, match="approved use identifiers"):
        transition_item_bank_record(
            approved,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=(),
            transition_reason_id="release_activation",
            approved_use_ids=("not a valid identifier",),
        )
    with pytest.raises(ItemBankLifecycleError, match="change the approved use"):
        transition_item_bank_record(
            approved,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=(),
            transition_reason_id="release_activation",
            approved_use_ids=("different_use",),
        )
    with pytest.raises(ItemBankLifecycleError, match="transition reason"):
        transition_item_bank_record(
            calibrated,
            ItemBankLifecycleState.APPROVED,
            evidence_references=(_evidence(ItemBankEvidenceKind.APPROVAL, "approval_again", fingerprint_character="e"),),
            transition_reason_id="not a reason",
            approved_use_ids=("production_use",),
        )


def test_pilot_admission_rejects_untyped_and_mismatched_audit_artifacts() -> None:
    """Pilot admission binds exact candidate and audit-report identities."""
    candidate_factory = _AUDIT_FIXTURES["_candidate"]
    pilot_kwargs = _AUDIT_FIXTURES["_pilot_kwargs"]()
    candidate = candidate_factory()
    report = audit_generated_item_candidate(candidate)
    with pytest.raises(TypeError, match="GeneratedItemCandidate"):
        audit_module.build_pilot_candidate_record(object(), report, **pilot_kwargs)
    with pytest.raises(TypeError, match="CandidateAuditReport"):
        audit_module.build_pilot_candidate_record(candidate, object(), **pilot_kwargs)
    object.__setattr__(report, "candidate_fingerprint", "f" * 64)
    with pytest.raises(PilotAdmissionError, match="exact candidate"):
        audit_module.build_pilot_candidate_record(candidate, report, **pilot_kwargs)


def test_core_option_audit_accepts_ordinary_selected_response_options() -> None:
    """The aggregate-option rule must also exercise its ordinary-option branch."""
    request = _AUDIT_FIXTURES["_request"](ResponseFormat.SELECTED_RESPONSE)
    candidate = _AUDIT_FIXTURES["_candidate"](request)

    assert audit_module._option_findings(candidate) == []
