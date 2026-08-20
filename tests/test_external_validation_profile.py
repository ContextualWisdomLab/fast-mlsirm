"""Contract tests for preregistered external-validity evidence profiles."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm.external_validation import (
    EvidenceClass,
    EvidenceStatus,
    ExternalValidationProfile,
    ValidationEvidence,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


class _HostileText(str):
    """String subclass whose text callbacks must not run during admission."""

    callbacks = 0

    def strip(self, *args: object, **kwargs: object) -> str:
        """Fail if package validation dispatches caller-owned text behavior."""
        type(self).callbacks += 1
        raise AssertionError("caller text callback executed")


def _evidence(
    evidence_id: str,
    evidence_class: EvidenceClass,
    *,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    available_time: str = "2026-07-01T00:00:00Z",
) -> ValidationEvidence:
    """Build one source-free evidence reference for a profile fixture."""
    return ValidationEvidence(
        evidence_id=evidence_id,
        evidence_class=evidence_class,
        status=status,
        available_time=available_time,
        artifact_sha256=_SHA_D,
        limitation="aggregate-only evidence; no raw participant content",
    )


def _profile(*evidence: ValidationEvidence) -> ExternalValidationProfile:
    """Build a bounded provider-neutral validation profile fixture."""
    return ExternalValidationProfile(
        validation_profile_id="external_validity_v1",
        construct="reasoning quality",
        score_interpretation="higher scores indicate stronger rubric-aligned reasoning",
        population="adult knowledge workers",
        setting="independent held-out evaluation",
        decision_use="research validation only",
        assessment_fingerprint=_SHA_A,
        rubric_fingerprint=_SHA_B,
        item_bank_fingerprint=_SHA_C,
        model_fingerprint=_SHA_D,
        development_dataset_ids=("development_set_v1",),
        internal_validation_dataset_ids=("internal_validation_v1",),
        external_validation_dataset_ids=("external_validation_v1",),
        sites=("site_alpha", "site_beta"),
        languages=("en", "ko"),
        preregistration_reference="protocol_2026_001",
        preregistered_at="2026-06-01T00:00:00Z",
        analysis_cutoff="2026-08-01T00:00:00Z",
        data_license="synthetic_and_open",
        purpose_classification="research_validation",
        evidence=evidence,
    )


def test_manifest_is_deterministic_and_keeps_evidence_classes_distinct() -> None:
    """Evidence order cannot change identity and classes remain separately visible."""
    technical = _evidence("technical_conformance", EvidenceClass.TECHNICAL)
    transport = _evidence(
        "transportability_holdout",
        EvidenceClass.TRANSPORTABILITY,
        status=EvidenceStatus.INDETERMINATE,
    )

    first = _profile(technical, transport)
    second = _profile(transport, technical)

    assert first.profile_fingerprint == second.profile_fingerprint
    manifest = first.to_manifest()
    assert manifest["profile_fingerprint"] == first.profile_fingerprint
    assert [row["evidence_class"] for row in manifest["evidence"]] == [
        "technical",
        "transportability",
    ]
    assert [row["status"] for row in manifest["evidence"]] == [
        "passed",
        "indeterminate",
    ]


def test_future_evidence_cannot_cross_the_analysis_cutoff() -> None:
    """Evidence unavailable at the preregistered cutoff fails closed."""
    future = _evidence(
        "future_outcome",
        EvidenceClass.DECISION_UTILITY,
        available_time="2026-08-02T00:00:00Z",
    )

    with pytest.raises(ValueError, match="available_time must not exceed analysis_cutoff"):
        _profile(future)


def test_preregistration_must_precede_the_analysis_cutoff() -> None:
    """A protocol timestamp after analysis cutoff cannot masquerade as preregistration."""
    profile = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))

    with pytest.raises(ValueError, match="preregistered_at must not exceed analysis_cutoff"):
        replace(profile, preregistered_at="2026-08-02T00:00:00Z")


def test_duplicate_evidence_ids_are_rejected() -> None:
    """One evidence identity cannot silently represent multiple evidence classes."""
    first = _evidence("shared_evidence", EvidenceClass.TECHNICAL)
    second = _evidence("shared_evidence", EvidenceClass.FAIRNESS)

    with pytest.raises(ValueError, match="evidence_id values must be unique"):
        _profile(first, second)


def test_status_vocabulary_preserves_failure_and_nonexecution_states() -> None:
    """The reusable contract must never collapse non-success states into pass."""
    assert {member.value for member in EvidenceStatus} == {
        "passed",
        "failed",
        "indeterminate",
        "not_executed",
        "not_applicable",
    }
    assert {member.value for member in EvidenceClass} == {
        "technical",
        "construct",
        "transportability",
        "fairness",
        "decision_utility",
    }


def test_text_subclasses_fail_before_text_callbacks() -> None:
    """Profile identity admission rejects hostile text without invoking ``strip``."""
    _HostileText.callbacks = 0
    technical = _evidence("technical_conformance", EvidenceClass.TECHNICAL)

    with pytest.raises(ValueError, match="validation_profile_id must be a string"):
        replace(_profile(technical), validation_profile_id=_HostileText("profile_v1"))

    assert _HostileText.callbacks == 0
