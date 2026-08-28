"""Creation-seal regression for public pilot-admission records."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import (
    audit_generated_item_candidate,
    build_pilot_candidate_record,
)

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
