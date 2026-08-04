"""Contract tests for enterprise-issue source and evidence adapters."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
from fast_mlsirm.scoring import AssessmentSpecError, EvidenceRole
from fast_mlsirm.scoring.enterprise_issue import (
    CounterevidenceRecord,
    EnterpriseSourceKind,
    EnterpriseSourceRecord,
    EvidenceAssertionKind,
    EvidenceSpanRecord,
    build_counterevidence_record,
    build_enterprise_source_record,
    build_evidence_span_record,
)


class InvalidIndex:
    """Raise a bounded conversion failure when used as an integer."""

    def __index__(self) -> int:
        """Reject integer conversion."""
        raise TypeError("private callback detail")


def source(**overrides: Any) -> EnterpriseSourceRecord:
    """Return one deterministic source adapter."""
    values: dict[str, Any] = {
        "source_id": "customer_report",
        "source_kind": EnterpriseSourceKind.REPORT_RECORD,
        "source_content_fingerprint": "1" * 64,
        "source_revision_fingerprint": "2" * 64,
        "source_character_count": 1_200,
        "metadata": {"collection_stage": "pilot"},
    }
    values.update(overrides)
    return build_enterprise_source_record(**values)


def span(
    source_record: EnterpriseSourceRecord | None = None,
    **overrides: Any,
) -> EvidenceSpanRecord:
    """Return one deterministic evidence-span adapter."""
    values: dict[str, Any] = {
        "span_id": "customer_span",
        "source_record": source_record or source(),
        "assertion_kind": EvidenceAssertionKind.DIRECT_FACT,
        "content_fingerprint": "3" * 64,
        "start_offset": 20,
        "end_offset": 60,
        "metadata": {"extractor_family": "manual_review"},
    }
    values.update(overrides)
    return build_evidence_span_record(**values)


def counterevidence(
    evidence_span: EvidenceSpanRecord | None = None,
    **overrides: Any,
) -> CounterevidenceRecord:
    """Return one deterministic counterevidence binding."""
    values: dict[str, Any] = {
        "counterevidence_id": "issue_counterevidence",
        "issue_statement_fingerprint": "4" * 64,
        "evidence_span": evidence_span
        or span(assertion_kind=EvidenceAssertionKind.COUNTEREVIDENCE),
        "metadata": {"review_stage": "evidence_screening"},
    }
    values.update(overrides)
    return build_counterevidence_record(**values)


def assert_error(code: str, callback) -> AssessmentSpecError:
    """Assert one stable enterprise/scoring contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def test_public_surface_is_explicit_and_documented() -> None:
    """The enterprise namespace exports only the reviewed provenance surface."""
    expected = {
        "CounterevidenceRecord",
        "EnterpriseSourceKind",
        "EnterpriseSourceRecord",
        "EvidenceAssertionKind",
        "EvidenceSpanRecord",
        "MAX_ENTERPRISE_SOURCE_CHARACTERS",
        "build_counterevidence_record",
        "build_enterprise_source_record",
        "build_evidence_span_record",
    }
    assert set(enterprise.__all__) == expected
    assert all(
        getattr(enterprise, name).__doc__
        for name in expected
        if name[0].isupper()
    )
    assert all(
        getattr(enterprise, name).__doc__
        for name in expected
        if name.startswith("build_")
    )


def test_source_is_content_addressed_and_deeply_immutable() -> None:
    """Equivalent metadata order yields one immutable source identity."""
    metadata = {"nested_value": {"second_key": 2, "first_key": [1, 2]}}
    first = source(metadata=metadata)
    second = source(metadata={"nested_value": {"first_key": [1, 2], "second_key": 2}})
    original = first.source_record_fingerprint
    metadata["nested_value"]["first_key"].append(3)

    assert (
        first.source_record_fingerprint
        == second.source_record_fingerprint
        == original
    )
    assert isinstance(first.metadata, MappingProxyType)
    assert first.source_record_handle == f"enterprise_source_{original[:32]}"
    assert first.to_dict()["source_record_fingerprint"] == original
    assert first.to_dict()["source_kind"] == "report_record"


def test_source_rejects_unverified_or_malformed_inputs() -> None:
    """Source construction fails closed on direct, invalid, or sensitive inputs."""
    assert_error(
        "unverified_enterprise_source",
        lambda: EnterpriseSourceRecord(
            source_id="customer_report",
            source_kind=EnterpriseSourceKind.REPORT_RECORD,
            source_content_fingerprint="1" * 64,
            source_revision_fingerprint="2" * 64,
            source_character_count=1_200,
            metadata={},
        ),
    )
    assert_error("invalid_source_id", lambda: source(source_id="report"))
    assert_error("invalid_source_kind", lambda: source(source_kind="sentiment_record"))
    assert_error(
        "invalid_source_content_fingerprint",
        lambda: source(source_content_fingerprint="not_a_digest"),
    )
    assert_error(
        "invalid_source_revision_fingerprint",
        lambda: source(source_revision_fingerprint="not_a_digest"),
    )
    assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=True),
    )
    callback_error = assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=InvalidIndex()),
    )
    assert "private callback detail" not in str(callback_error)
    assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=-1),
    )
    assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=100_000_001),
    )
    assert_error(
        "sensitive_metadata_field",
        lambda: source(metadata={"source_text": "must not persist"}),
    )


@pytest.mark.parametrize(
    ("assertion_kind", "expected_role"),
    (
        (EvidenceAssertionKind.DIRECT_FACT, EvidenceRole.SUPPORTING),
        (EvidenceAssertionKind.SUPPORTED_INFERENCE, EvidenceRole.SUPPORTING),
        (EvidenceAssertionKind.COUNTEREVIDENCE, EvidenceRole.COUNTER),
        (EvidenceAssertionKind.UNRESOLVED_AMBIGUITY, EvidenceRole.CONTEXT),
        (
            EvidenceAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
            EvidenceRole.CONTEXT,
        ),
    ),
)
def test_evidence_span_projects_explicit_epistemic_roles(
    assertion_kind: EvidenceAssertionKind,
    expected_role: EvidenceRole,
) -> None:
    """Every assertion family maps to a visible shared evidence role."""
    record = span(assertion_kind=assertion_kind)
    reference = record.shared_evidence_reference

    assert reference.evidence_role is expected_role
    assert reference.source_id == record.source_id
    assert reference.span_id == record.span_id
    assert reference.content_fingerprint == record.content_fingerprint
    assert record.to_dict()["shared_evidence_reference"] == reference.to_dict()


def test_evidence_span_is_content_addressed_and_source_bound() -> None:
    """Evidence identity binds exact source revision, offsets, and assertion kind."""
    source_record = source()
    first = span(source_record, metadata={"second_key": 2, "first_key": 1})
    second = span(source_record, metadata={"first_key": 1, "second_key": 2})
    original = first.evidence_span_fingerprint

    assert (
        first.evidence_span_fingerprint
        == second.evidence_span_fingerprint
        == original
    )
    assert first.source_record_fingerprint == source_record.source_record_fingerprint
    assert first.evidence_span_handle == f"enterprise_evidence_{original[:32]}"
    assert first.to_dict()["evidence_span_fingerprint"] == original


def test_evidence_span_rejects_unverified_mismatched_or_invalid_inputs() -> None:
    """Span construction fails closed on provenance, role, and offset failures."""
    source_record = source()
    assert_error(
        "unverified_evidence_span",
        lambda: EvidenceSpanRecord(
            span_id="customer_span",
            source_id=source_record.source_id,
            source_record_fingerprint=source_record.source_record_fingerprint,
            assertion_kind=EvidenceAssertionKind.DIRECT_FACT,
            content_fingerprint="3" * 64,
            start_offset=20,
            end_offset=60,
            metadata={},
        ),
    )
    assert_error(
        "invalid_source_record",
        lambda: span(source_record="not_a_source_record"),
    )
    assert_error("invalid_span_id", lambda: span(span_id="span"))
    assert_error(
        "invalid_assertion_kind",
        lambda: span(assertion_kind="positive_sentiment"),
    )
    assert_error(
        "invalid_content_fingerprint",
        lambda: span(content_fingerprint="not_a_digest"),
    )
    assert_error("invalid_start_offset", lambda: span(start_offset=True))
    assert_error("invalid_start_offset", lambda: span(start_offset=InvalidIndex()))
    assert_error("invalid_start_offset", lambda: span(start_offset=-1))
    assert_error("invalid_end_offset", lambda: span(end_offset=100_000_001))
    assert_error(
        "invalid_evidence_offsets",
        lambda: span(start_offset=60, end_offset=60),
    )
    assert_error(
        "invalid_evidence_offsets",
        lambda: span(start_offset=61, end_offset=60),
    )
    assert_error(
        "sensitive_metadata_field",
        lambda: span(metadata={"response_text": "must not persist"}),
    )


def test_counterevidence_is_content_addressed_and_issue_bound() -> None:
    """Counterevidence retains the exact claim, span, and source identities."""
    counter_span = span(assertion_kind="counterevidence")
    first = counterevidence(counter_span, metadata={"second_key": 2, "first_key": 1})
    second = counterevidence(counter_span, metadata={"first_key": 1, "second_key": 2})
    original = first.counterevidence_fingerprint

    assert (
        first.counterevidence_fingerprint
        == second.counterevidence_fingerprint
        == original
    )
    assert first.evidence_span_id == counter_span.span_id
    assert first.evidence_span_fingerprint == counter_span.evidence_span_fingerprint
    assert first.source_record_fingerprint == counter_span.source_record_fingerprint
    assert first.counterevidence_handle == f"counterevidence_record_{original[:32]}"
    assert first.to_dict()["counterevidence_fingerprint"] == original


def test_counterevidence_rejects_unverified_or_incompatible_inputs() -> None:
    """Counterevidence cannot be fabricated from supporting or untyped spans."""
    counter_span = span(assertion_kind=EvidenceAssertionKind.COUNTEREVIDENCE)
    assert_error(
        "unverified_counterevidence_record",
        lambda: CounterevidenceRecord(
            counterevidence_id="issue_counterevidence",
            issue_statement_fingerprint="4" * 64,
            evidence_span_id=counter_span.span_id,
            evidence_span_fingerprint=counter_span.evidence_span_fingerprint,
            source_record_fingerprint=counter_span.source_record_fingerprint,
            metadata={},
        ),
    )
    assert_error(
        "invalid_evidence_span",
        lambda: counterevidence(evidence_span="not_an_evidence_span"),
    )
    assert_error(
        "invalid_counterevidence_span",
        lambda: counterevidence(span(assertion_kind=EvidenceAssertionKind.DIRECT_FACT)),
    )
    assert_error(
        "invalid_counterevidence_id",
        lambda: counterevidence(counter_span, counterevidence_id="counterevidence"),
    )
    assert_error(
        "invalid_issue_statement_fingerprint",
        lambda: counterevidence(
            counter_span,
            issue_statement_fingerprint="not_a_digest",
        ),
    )
    assert_error(
        "sensitive_metadata_field",
        lambda: counterevidence(counter_span, metadata={"essay_text": "forbidden"}),
    )


def test_serialized_contracts_never_retain_source_text() -> None:
    """Canonical exports contain identities and offsets but no caller content."""
    source_record = source(metadata={"source_channel": "customer_portal"})
    evidence_record = span(
        source_record,
        assertion_kind=EvidenceAssertionKind.COUNTEREVIDENCE,
        metadata={"extraction_method": "manual_review"},
    )
    counter_record = counterevidence(evidence_record)
    serialized = json.dumps(
        {
            "source": source_record.to_dict(),
            "evidence": evidence_record.to_dict(),
            "counterevidence": counter_record.to_dict(),
        },
        sort_keys=True,
    )

    assert "source_text" not in serialized
    assert "response_text" not in serialized
    assert "essay_text" not in serialized
    assert "positive_sentiment" not in serialized
    assert source_record.source_content_fingerprint in serialized
    assert evidence_record.content_fingerprint in serialized
