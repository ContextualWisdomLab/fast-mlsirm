"""Behavior contracts for evidence-grounded semantic issue proposals."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceRole
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    EnterpriseAssertionKind,
    EnterpriseSemanticIssueProvider,
    EnterpriseSourceRecord,
    MAX_SEMANTIC_ASSERTIONS_PER_ISSUE,
    MAX_SEMANTIC_ISSUE_PROPOSALS,
    MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS,
    OfflineSemanticIssueFixtureProvider,
    StakeholderPerspective,
    extract_enterprise_semantic_issues,
)

PROVIDER_FP = hashlib.sha256(b"offline-semantic-provider-v1").hexdigest()
ISSUE_STATEMENT = "Repeated enterprise delivery delays threaten renewal confidence."


def _source(
    source_id: str,
    text: str,
    *,
    source_family_id: str,
) -> EnterpriseSourceRecord:
    """Return one exact source record for transient source text."""
    return EnterpriseSourceRecord(
        source_id=source_id,
        source_family_id=source_family_id,
        source_content_fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_character_count=len(text),
        metadata={"source_channel": source_family_id},
    )


def _span(text: str, value: str) -> tuple[int, int]:
    """Return exact Python Unicode-code-point offsets for one substring."""
    start = text.index(value)
    return start, start + len(value)


def _assertion(
    *,
    source_id: str,
    source_text: str,
    span_text: str,
    assertion_kind: str,
    stakeholder_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return one primitive semantic assertion proposal."""
    start, end = _span(source_text, span_text)
    return {
        "source_id": source_id,
        "start_offset": start,
        "end_offset": end,
        "assertion_kind": assertion_kind,
        "stakeholder_id": stakeholder_id,
        "metadata": {} if metadata is None else metadata,
    }


def _fixture() -> tuple[
    tuple[EnterpriseSourceRecord, ...],
    dict[str, str],
    tuple[dict[str, object], ...],
]:
    """Return realistic report, lead-note, and complaint proposal fixtures."""
    report_text = (
        "Operations recorded three delayed enterprise deliveries in July. "
        "The remediation team reports that the backlog was cleared."
    )
    lead_text = (
        "The renewal sponsor may defer expansion if delivery reliability does not improve. "
        "The exact commercial impact remains unresolved."
    )
    complaint_text = (
        "The customer wrote that another late delivery would be unacceptable to the account team."
    )
    sources = (
        _source(
            "operations_report",
            report_text,
            source_family_id="operations_reporting",
        ),
        _source("sales_lead_note", lead_text, source_family_id="sales_pipeline"),
        _source(
            "customer_complaint",
            complaint_text,
            source_family_id="customer_feedback",
        ),
    )
    texts = {
        "operations_report": report_text,
        "sales_lead_note": lead_text,
        "customer_complaint": complaint_text,
    }
    assertions = (
        _assertion(
            source_id="operations_report",
            source_text=report_text,
            span_text="three delayed enterprise deliveries",
            assertion_kind="direct_fact",
            metadata={"assertion_scope": "reported_incidents"},
        ),
        _assertion(
            source_id="sales_lead_note",
            source_text=lead_text,
            span_text="may defer expansion",
            assertion_kind="supported_inference",
            metadata={"assertion_scope": "renewal_risk"},
        ),
        _assertion(
            source_id="sales_lead_note",
            source_text=lead_text,
            span_text="exact commercial impact remains unresolved",
            assertion_kind="unresolved_ambiguity",
        ),
        _assertion(
            source_id="operations_report",
            source_text=report_text,
            span_text="backlog was cleared",
            assertion_kind="counterevidence",
        ),
        _assertion(
            source_id="customer_complaint",
            source_text=complaint_text,
            span_text="another late delivery would be unacceptable",
            assertion_kind="stakeholder_value_judgment",
            stakeholder_id="account_team",
        ),
    )
    proposals = (
        {
            "issue_id": "delivery_reliability_risk",
            "issue_family_id": "service_delivery_risk",
            "issue_statement": ISSUE_STATEMENT,
            "assertions": assertions,
            "metadata": {"proposal_scope": "renewal_review"},
        },
    )
    return sources, texts, proposals


def _extract(
    *,
    proposals: tuple[dict[str, object], ...] | None = None,
    sources: tuple[EnterpriseSourceRecord, ...] | None = None,
    texts: dict[str, str] | None = None,
):
    """Run the stable boundary with deterministic offline fixtures."""
    fixture_sources, fixture_texts, fixture_proposals = _fixture()
    provider = OfflineSemanticIssueFixtureProvider(
        provider_revision_fingerprint=PROVIDER_FP,
        proposals=fixture_proposals if proposals is None else proposals,
    )
    return extract_enterprise_semantic_issues(
        fixture_sources if sources is None else sources,
        source_text_by_id=fixture_texts if texts is None else texts,
        provider=provider,
    )


def test_mixed_semantic_assertions_compile_into_existing_contracts() -> None:
    """All epistemic roles compile without creating a parallel issue schema."""
    sources, texts, _proposals = _fixture()
    issues, perspectives = _extract()

    assert len(issues) == 1
    assert len(perspectives) == 1
    issue = issues[0]
    perspective = perspectives[0]
    assert type(issue) is AtomicIssueRecord
    assert type(perspective) is StakeholderPerspective
    assert issue.issue_id == "delivery_reliability_risk"
    assert issue.issue_family_id == "service_delivery_risk"
    assert issue.issue_content_fingerprint == hashlib.sha256(
        ISSUE_STATEMENT.encode("utf-8")
    ).hexdigest()
    assert {span.assertion_kind for span in issue.evidence_spans} == {
        EnterpriseAssertionKind.DIRECT_FACT,
        EnterpriseAssertionKind.SUPPORTED_INFERENCE,
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
    }
    assert len(issue.counterevidence_records) == 1
    assert (
        issue.counterevidence_records[0].evidence_span.assertion_kind
        is EnterpriseAssertionKind.COUNTEREVIDENCE
    )
    assert perspective.stakeholder_id == "account_team"
    assert (
        perspective.value_judgment_span.assertion_kind
        is EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT
    )
    assert set(issue.source_record_fingerprints).issubset(
        {source.source_record_fingerprint for source in sources}
    )
    assert issue.metadata["semantic_provider_revision_fingerprint"] == PROVIDER_FP
    assert all(
        reference.evidence_role
        in {EvidenceRole.SUPPORTING, EvidenceRole.COUNTER, EvidenceRole.CONTEXT}
        for reference in issue.evidence_references()
    )

    serialized = repr(
        {
            "issues": [value.to_dict() for value in issues],
            "perspectives": [value.to_dict() for value in perspectives],
        }
    )
    for secret_value in (
        ISSUE_STATEMENT,
        *texts.values(),
        "another late delivery would be unacceptable",
    ):
        assert secret_value not in serialized


def test_exact_spans_and_fingerprints_are_package_derived() -> None:
    """Every accepted span replays exact code-point offsets and UTF-8 content."""
    _sources, texts, _proposals = _fixture()
    issues, perspectives = _extract()
    spans = list(issues[0].evidence_spans)
    spans.extend(
        record.evidence_span for record in issues[0].counterevidence_records
    )
    spans.extend(value.value_judgment_span for value in perspectives)

    for span in spans:
        source_text = texts[span.source_id]
        exact = source_text[span.start_offset : span.end_offset]
        assert span.span_content_fingerprint == hashlib.sha256(
            exact.encode("utf-8")
        ).hexdigest()
        assert span.span_id.startswith("semantic_span_")
        assert span.metadata["offset_unit"] == "python_unicode_code_point"


def test_output_is_invariant_to_proposal_assertion_source_and_mapping_order() -> None:
    """Equivalent primitive inputs produce identical canonical output."""
    sources, texts, proposals = _fixture()
    proposal = proposals[0]
    reversed_proposal = {
        "metadata": dict(reversed(tuple(proposal["metadata"].items()))),
        "assertions": tuple(reversed(proposal["assertions"])),
        "issue_statement": proposal["issue_statement"],
        "issue_family_id": proposal["issue_family_id"],
        "issue_id": proposal["issue_id"],
    }
    first_issues, first_perspectives = _extract()
    second_issues, second_perspectives = _extract(
        proposals=(reversed_proposal,),
        sources=tuple(reversed(sources)),
        texts=dict(reversed(tuple(texts.items()))),
    )

    assert tuple(value.atomic_issue_fingerprint for value in first_issues) == tuple(
        value.atomic_issue_fingerprint for value in second_issues
    )
    assert tuple(value.perspective_fingerprint for value in first_perspectives) == tuple(
        value.perspective_fingerprint for value in second_perspectives
    )


def test_source_revision_changes_are_visible_without_changing_issue_identity() -> None:
    """Reordered source prose changes provenance but not the transient issue digest."""
    sources, texts, proposals = _fixture()
    original_issues, _ = _extract()
    report = texts["operations_report"]
    reordered = "Context updated. " + report
    changed_sources = tuple(
        _source(
            source.source_id,
            reordered if source.source_id == "operations_report" else texts[source.source_id],
            source_family_id=source.source_family_id,
        )
        for source in sources
    )
    changed_texts = {**texts, "operations_report": reordered}
    changed_assertions = []
    for assertion in proposals[0]["assertions"]:
        copied = dict(assertion)
        if copied["source_id"] == "operations_report":
            copied["start_offset"] += len("Context updated. ")
            copied["end_offset"] += len("Context updated. ")
        changed_assertions.append(copied)
    changed_proposal = {**proposals[0], "assertions": tuple(changed_assertions)}
    changed_issues, _ = _extract(
        proposals=(changed_proposal,),
        sources=changed_sources,
        texts=changed_texts,
    )

    assert changed_issues[0].issue_content_fingerprint == original_issues[0].issue_content_fingerprint
    assert changed_issues[0].source_record_fingerprints != original_issues[0].source_record_fingerprints
    assert changed_issues[0].atomic_issue_fingerprint != original_issues[0].atomic_issue_fingerprint


def test_sentiment_outside_selected_spans_cannot_change_issue_content_identity() -> None:
    """Unselected sentiment wording never becomes the persisted issue construct."""
    sources, texts, proposals = _fixture()
    original_issues, _ = _extract()
    complaint = texts["customer_complaint"]
    changed_complaint = complaint + " The author sounded extremely angry and negative."
    changed_sources = tuple(
        _source(
            source.source_id,
            changed_complaint if source.source_id == "customer_complaint" else texts[source.source_id],
            source_family_id=source.source_family_id,
        )
        for source in sources
    )
    changed_texts = {**texts, "customer_complaint": changed_complaint}
    changed_issues, _ = _extract(
        proposals=proposals,
        sources=changed_sources,
        texts=changed_texts,
    )

    assert changed_issues[0].issue_content_fingerprint == original_issues[0].issue_content_fingerprint


def test_offline_provider_is_runtime_checkable_immutable_and_repeatable() -> None:
    """The fixture adapter is an ordinary provider and returns fresh primitives."""
    _sources, _texts, proposals = _fixture()
    provider = OfflineSemanticIssueFixtureProvider(
        provider_revision_fingerprint=PROVIDER_FP,
        proposals=proposals,
    )

    assert isinstance(provider, EnterpriseSemanticIssueProvider)
    first = provider.propose(sources=())
    second = provider.propose(sources=())
    assert first == second
    assert first is not second
    with pytest.raises(FrozenInstanceError):
        provider.provider_revision_fingerprint = "f" * 64  # type: ignore[misc]


def test_public_limits_are_positive_and_descriptive() -> None:
    """Resource bounds remain explicit public contracts."""
    assert MAX_SEMANTIC_ISSUE_PROPOSALS >= 1
    assert MAX_SEMANTIC_ASSERTIONS_PER_ISSUE >= 5
    assert MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS >= len(ISSUE_STATEMENT)
