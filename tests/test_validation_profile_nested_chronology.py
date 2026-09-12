"""Chronology ownership regressions for preregistered validation profiles."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fast_mlsirm.validation_profile import (
    ValidationEvidenceClass,
    ValidationEvidenceReference,
    ValidationEvidenceStatus,
    ValidationProfile,
)


_FINGERPRINTS = tuple(character * 64 for character in "abcde")


def test_profile_constructor_replays_nested_time_without_post_validation_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested mutation after evidence replay cannot execute chronology protocols."""
    callbacks = 0
    reference = ValidationEvidenceReference(
        evidence_id="technical_evidence",
        artifact_fingerprint=_FINGERPRINTS[0],
        evidence_class=ValidationEvidenceClass.TECHNICAL,
        status=ValidationEvidenceStatus.PASSED,
        available_time=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    original = ValidationEvidenceReference._validated_content

    class HostileChronology:
        def __gt__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("post-validation chronology callback must not execute")

    def validate_then_mutate(self: ValidationEvidenceReference) -> dict[str, object]:
        payload = original(self)
        object.__setattr__(self, "available_time", HostileChronology())
        return payload

    monkeypatch.setattr(
        ValidationEvidenceReference,
        "_validated_content",
        validate_then_mutate,
    )

    with pytest.raises(ValueError, match="available_time must be an exact datetime"):
        ValidationProfile(
            validation_profile_id="enterprise_validation_profile",
            protocol_fingerprint=_FINGERPRINTS[0],
            assessment_fingerprint=_FINGERPRINTS[1],
            rubric_fingerprint=_FINGERPRINTS[2],
            item_bank_fingerprint=_FINGERPRINTS[3],
            model_fingerprint=_FINGERPRINTS[4],
            intended_construct="decision_quality",
            score_interpretation="relative_decision_quality",
            population="held_out_enterprise_systems",
            setting="external_validation_sites",
            decision_use="human_review_prioritization",
            protocol_registered_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            analysis_cutoff=datetime(2026, 8, 21, tzinfo=timezone.utc),
            evidence_references=(reference,),
        )

    assert callbacks == 0
