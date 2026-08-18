"""Regression tests for callback-free rubric text admission."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.rubric.audit import AuditSeverity, CandidateAuditFinding


def _hostile_text(value: str) -> tuple[str, type[str]]:
    """Return a string subclass whose text callbacks must never execute."""

    class HostileText(str):
        calls = 0

        def strip(self, *args: object, **kwargs: object) -> str:
            type(self).calls += 1
            raise AssertionError("caller strip callback executed")

        def __hash__(self) -> int:
            type(self).calls += 1
            raise AssertionError("caller hash callback executed")

    return HostileText(value), HostileText


def _levels() -> tuple[RubricLevel, RubricLevel]:
    """Return a minimal valid ordinal rubric scale."""
    return (
        RubricLevel(0, "not_met", "Requirement is not met.", ("missing evidence",)),
        RubricLevel(1, "fully_met", "Requirement is fully met.", ("complete evidence",)),
    )


def test_rubric_level_rejects_string_subclass_without_callback() -> None:
    """Scalar level text rejects caller subclasses before invoking text methods."""
    hostile, hostile_type = _hostile_text("not_met")

    with pytest.raises(ValueError, match="label must be a string"):
        RubricLevel(0, hostile, "Requirement is not met.", ("missing evidence",))

    assert hostile_type.calls == 0


def test_rubric_specification_rejects_identifier_subclass_without_callback() -> None:
    """Identifier text rejects caller subclasses before normalization callbacks."""
    hostile, hostile_type = _hostile_text("evidence_rubric")

    with pytest.raises(ValueError, match="rubric_id must be a string"):
        RubricSpecification(
            rubric_id=hostile,
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=("Cite supporting evidence.",),
        )

    assert hostile_type.calls == 0


def test_rubric_specification_rejects_enum_text_subclass_without_callback() -> None:
    """Enum text rejects caller subclasses before value lookup callbacks."""
    hostile, hostile_type = _hostile_text("ordinal_rating")

    with pytest.raises(ValueError, match="response_format must be one of"):
        RubricSpecification(
            rubric_id="evidence_rubric",
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=hostile,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=("Cite supporting evidence.",),
        )

    assert hostile_type.calls == 0


def test_audit_finding_rejects_enum_text_subclass_without_callback() -> None:
    """Audit enum text rejects caller subclasses before Enum value lookup."""
    hostile, hostile_type = _hostile_text("blocking")

    with pytest.raises(ValueError, match="severity must be one of"):
        CandidateAuditFinding(
            finding_code="unsafe_content",
            severity=hostile,
            path="$.candidate",
            message="Candidate requires review.",
        )

    assert hostile_type.calls == 0


def test_audit_finding_accepts_builtin_enum_value() -> None:
    """Audit findings retain exact built-in enum-string compatibility."""
    finding = CandidateAuditFinding(
        finding_code="unsafe_content",
        severity="blocking",
        path="$.candidate",
        message="Candidate requires review.",
    )

    assert finding.severity is AuditSeverity.BLOCKING


def test_rubric_specification_rejects_collection_text_subclass_without_callback() -> None:
    """Nested rubric text rejects caller subclasses before tuple normalization callbacks."""
    hostile, hostile_type = _hostile_text("Cite supporting evidence.")

    with pytest.raises(ValueError, match=r"evidence_requirements\[0\] must be a string"):
        RubricSpecification(
            rubric_id="evidence_rubric",
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=(hostile,),
        )

    assert hostile_type.calls == 0


def test_builtin_rubric_text_still_normalizes() -> None:
    """Exact built-in strings retain whitespace normalization semantics."""
    level = RubricLevel(
        0,
        "  not_met  ",
        "  Requirement is not met.  ",
        ("  missing evidence  ",),
    )

    assert level.label == "not_met"
    assert level.descriptor == "Requirement is not met."
    assert level.observable_indicators == ("missing evidence",)
