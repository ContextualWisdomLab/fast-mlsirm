"""Factory-sealing contracts for public pilot-admission records."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import (
    PilotCandidateRecord,
    audit_generated_item_candidate,
    build_pilot_candidate_record,
)
from fast_mlsirm.rubric.audit import PilotCandidateRecord as CorePilotCandidateRecord
from fast_mlsirm.rubric.verified_pilot import _from_verified_core

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)
_candidate = _FIXTURES["_candidate"]
_pilot_kwargs = _FIXTURES["_pilot_kwargs"]
_PUBLIC_FIELDS = (
    "pilot_study_id",
    "query_testlet_id",
    "generator_family_id",
    "judge_policy_id",
    "occasion_id",
    "item_id",
    "candidate_fingerprint",
    "audit_report_fingerprint",
    "screening_result_fingerprint",
    "audit_policy_id",
    "audit_policy_version",
    "blueprint_id",
    "rubric_id",
    "rubric_version",
    "lifecycle_state",
    "schema_version",
)


def _verified_record() -> PilotCandidateRecord:
    """Build one replay-verified pilot record through the public policy."""
    candidate = _candidate()
    report = audit_generated_item_candidate(candidate)
    return build_pilot_candidate_record(
        candidate,
        report,
        screening_result=_FIXTURES["_screening_result"](candidate, report),
        **_pilot_kwargs(),
    )


def test_public_pilot_record_requires_replay_verified_factory():
    """Valid-looking public values cannot directly mint an admission record."""
    verified = _verified_record()
    public_values = {name: getattr(verified, name) for name in _PUBLIC_FIELDS}

    assert verified.pilot_record_fingerprint == verified.to_dict()[
        "pilot_record_fingerprint"
    ]
    with pytest.raises(ValueError, match="build_pilot_candidate_record"):
        PilotCandidateRecord(**public_values)


def test_screening_bound_pilot_contract_versions_and_rejects_legacy_schema():
    """Mandatory screening has a distinct record schema and admission-policy version."""
    verified = _verified_record()
    assert verified.schema_version == "2.0"
    assert verified.audit_policy_version == "2.0.0"

    legacy_values = {name: getattr(verified, name) for name in _PUBLIC_FIELDS}
    legacy_values["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="pilot schema_version must be '2.0'"):
        CorePilotCandidateRecord(**legacy_values)


def test_verified_core_wrapper_rejects_wrong_runtime_type():
    """The private adapter fails closed before dereferencing caller objects."""
    with pytest.raises(TypeError, match="core PilotCandidateRecord"):
        _from_verified_core(object())
