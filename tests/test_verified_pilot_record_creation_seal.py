"""Creation-seal regressions for public pilot-admission records."""

from __future__ import annotations

import gc
from pathlib import Path
import runpy
import weakref

import pytest

from fast_mlsirm.rubric import (
    audit_generated_item_candidate,
    build_pilot_candidate_record,
)
from fast_mlsirm.rubric import verified_pilot as pilot_safety

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)
_candidate = _FIXTURES["_candidate"]
_pilot_kwargs = _FIXTURES["_pilot_kwargs"]
_screening_result = _FIXTURES["_screening_result"]


def _verified_pilot_record():
    """Return one public pilot record minted through the governed admission path."""
    candidate = _candidate()
    report = audit_generated_item_candidate(candidate)
    return build_pilot_candidate_record(
        candidate,
        report,
        screening_result=_screening_result(candidate, report),
        **_pilot_kwargs(),
    )


def test_public_pilot_record_rejects_post_construction_screening_rebinding() -> None:
    """A valid replacement fingerprint cannot mint new pilot authority after creation."""
    record = _verified_pilot_record()
    original_fingerprint = record.pilot_record_fingerprint

    object.__setattr__(record, "screening_result_fingerprint", "0" * 64)

    with pytest.raises(ValueError, match="factory seal"):
        record.to_dict()
    with pytest.raises(ValueError, match="factory seal"):
        _ = record.pilot_record_fingerprint
    assert original_fingerprint != "0" * 64


def test_pilot_creation_seal_rejects_reused_object_identity_entry() -> None:
    """A registry entry for another live object cannot authorize this pilot record."""
    record = _verified_pilot_record()
    other = _verified_pilot_record()
    record_key = id(record)
    original_entry = pilot_safety._CREATION_SEALS[record_key]
    pilot_safety._CREATION_SEALS[record_key] = (
        weakref.ref(other),
        original_entry[1],
    )
    try:
        with pytest.raises(ValueError, match="factory seal"):
            record.to_dict()
    finally:
        pilot_safety._CREATION_SEALS[record_key] = original_entry


def test_pilot_creation_seal_registry_releases_discarded_records() -> None:
    """The package-owned creation seal does not retain discarded pilot records."""
    record = _verified_pilot_record()
    record_key = id(record)
    reference = weakref.ref(record)

    assert record_key in pilot_safety._CREATION_SEALS
    del record
    gc.collect()

    assert reference() is None
    assert record_key not in pilot_safety._CREATION_SEALS


def test_pilot_replay_rejects_callback_bearing_rebinding_before_comparison() -> None:
    """A hostile scalar subclass cannot execute equality during seal replay."""
    record = _verified_pilot_record()
    callbacks = 0

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller equality must not run")

    object.__setattr__(
        record,
        "screening_result_fingerprint",
        HostileString("0" * 64),
    )

    with pytest.raises(ValueError, match="factory seal"):
        record.to_dict()

    assert callbacks == 0


def test_valid_pilot_creation_seal_preserves_identity_and_serialization() -> None:
    """Replay hardening leaves valid public pilot identity and payload unchanged."""
    record = _verified_pilot_record()

    payload = record.to_dict()

    assert payload["pilot_record_fingerprint"] == record.pilot_record_fingerprint
    assert payload["pilot_record_id"] == record.pilot_record_id
