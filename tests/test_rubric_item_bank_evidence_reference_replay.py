"""Replay contracts for standalone governed item-bank evidence references."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
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
