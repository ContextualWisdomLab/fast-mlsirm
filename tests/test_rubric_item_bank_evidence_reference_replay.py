"""Replay contracts for governed item-bank evidence references."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleState,
    build_item_bank_pilot_record,
    transition_item_bank_record,
)

_LIFECYCLE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _reference() -> ItemBankEvidenceReference:
    return ItemBankEvidenceReference(
        evidence_kind=ItemBankEvidenceKind.CALIBRATION,
        evidence_id="calibration_evidence",
        evidence_fingerprint="a" * 64,
    )


def test_valid_evidence_reference_serialization_is_stable() -> None:
    reference = _reference()

    assert reference.to_dict() == {
        "evidence_kind": "calibration",
        "evidence_id": "calibration_evidence",
        "evidence_fingerprint": "a" * 64,
    }


def test_evidence_reference_replays_mutated_fingerprint_before_serialization() -> None:
    reference = _reference()
    object.__setattr__(reference, "evidence_fingerprint", "not-a-fingerprint")

    with pytest.raises(ValueError, match="evidence_fingerprint"):
        reference.to_dict()


def test_evidence_reference_rejects_callback_bearing_mutation_without_dispatch() -> None:
    callbacks = {"str": 0}

    class HostileIdentifier(str):
        def __str__(self) -> str:
            callbacks["str"] += 1
            raise AssertionError("caller string conversion must not execute")

    reference = _reference()
    object.__setattr__(reference, "evidence_id", HostileIdentifier("hostile_evidence"))

    with pytest.raises(ValueError, match="evidence reference"):
        reference.to_dict()

    assert callbacks == {"str": 0}


def test_transition_replays_mutated_evidence_reference_before_successor_authority() -> None:
    pilot = build_item_bank_pilot_record(
        _LIFECYCLE_FIXTURES["_pilot_record"](),
        item_version="1.0.0",
    )
    evidence = list(_LIFECYCLE_FIXTURES["_calibration_evidence"]())
    object.__setattr__(evidence[0], "evidence_fingerprint", "not-a-fingerprint")

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=evidence,
            transition_reason_id="calibration_completed",
        )

    assert caught.value.code == "invalid_evidence_reference"
    assert caught.value.path == "$.evidence_references[0]"


def test_transition_rejects_callback_bearing_evidence_mutation_without_dispatch() -> None:
    callbacks = {"str": 0}

    class HostileIdentifier(str):
        def __str__(self) -> str:
            callbacks["str"] += 1
            raise AssertionError("caller string conversion must not execute")

    pilot = build_item_bank_pilot_record(
        _LIFECYCLE_FIXTURES["_pilot_record"](),
        item_version="1.0.0",
    )
    evidence = list(_LIFECYCLE_FIXTURES["_calibration_evidence"]())
    object.__setattr__(
        evidence[0],
        "evidence_id",
        HostileIdentifier("hostile_evidence"),
    )

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=evidence,
            transition_reason_id="calibration_completed",
        )

    assert caught.value.code == "invalid_evidence_reference"
    assert caught.value.path == "$.evidence_references[0]"
    assert callbacks == {"str": 0}
