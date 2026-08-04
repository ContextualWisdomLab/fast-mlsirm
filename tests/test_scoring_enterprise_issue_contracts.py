"""Contract tests for enterprise issue adapters over shared scoring boundaries."""

from __future__ import annotations

from pathlib import Path
import runpy
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    FixtureOutcome,
    ObservationStatus,
    StaticFixtureEngine,
)
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CandidateIntervention,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseIssueScoringRequest,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    StakeholderPerspective,
    build_atomic_issue_record,
    build_candidate_intervention,
    build_counterevidence_record,
    build_enterprise_issue_scoring_request,
    build_enterprise_source_record,
    build_evidence_span_record,
    build_stakeholder_perspective,
    score_enterprise_issue_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
human_engine = _FIXTURES["human_engine"]
rubric = _FIXTURES["rubric"]


def source(**overrides: Any) -> EnterpriseSourceRecord:
    """Return one deterministic source record."""
    values: dict[str, Any] = {
        "source_id": "customer_complaint",
        "source_type_id": "complaint_record",
        "source_content_fingerprint": "1" * 64,
        "source_character_count": 500,
        "source_unit_count": 20,
        "subject_identifier_fingerprint": "2" * 64,
        "metadata": {"collection_channel": "support_portal"},
    }
    values.update(overrides)
    return build_enterprise_source_record(**values)


def evidence_span(
    source_value: EnterpriseSourceRecord | None = None,
    **overrides: Any,
) -> EvidenceSpanRecord:
    """Return one deterministic direct-fact evidence span."""
    values: dict[str, Any] = {
        "source": source_value or source(),
        "span_id": "material_fact_span",
        "content_fingerprint": "3" * 64,
        "assertion_kind": EnterpriseAssertionKind.DIRECT_FACT,
        "start_offset": 10,
        "end_offset": 40,
        "metadata": {"extractor_id": "manual_review"},
    }
    values.update(overrides)
    return build_evidence_span_record(**values)


def counter_span(source_value: EnterpriseSourceRecord | None = None) -> EvidenceSpanRecord:
    """Return one deterministic counterevidence span."""
    return evidence_span(
        source_value,
        span_id="counter_fact_span",
        content_fingerprint="4" * 64,
        assertion_kind=EnterpriseAssertionKind.COUNTEREVIDENCE,
        start_offset=50,
        end_offset=80,
    )


def counterevidence(
    source_value: EnterpriseSourceRecord | None = None,
    **overrides: Any,
) -> CounterevidenceRecord:
    """Return one deterministic counterevidence record."""
    values: dict[str, Any] = {
        "counterevidence_id": "contradictory_evidence",
        "target_claim_id": "recurrence_claim",
        "evidence_span": counter_span(source_value),
        "metadata": {"review_state": "verified"},
    }
    values.update(overrides)
    return build_counterevidence_record(**values)


def issue(**overrides: Any) -> AtomicIssueRecord:
    """Return one deterministic atomic issue."""
    source_record = overrides.pop("source_record", source())
    values: dict[str, Any] = {
        "issue_id": "billing_delay_issue",
        "issue_family_id": "service_reliability",
        "domain_id": "customer_operations",
        "issue_content_fingerprint": "5" * 64,
        "issue_character_count": 120,
        "issue_unit_count": 8,
        "evidence_spans": (evidence_span(source_record),),
        "counterevidence_records": (counterevidence(source_record),),
        "metadata": {"workflow_stage": "evidence_review"},
    }
    values.update(overrides)
    return build_atomic_issue_record(**values)


def perspective(
    issue_value: AtomicIssueRecord | None = None,
    **overrides: Any,
) -> StakeholderPerspective:
    """Return one deterministic stakeholder perspective."""
    values: dict[str, Any] = {
        "perspective_id": "customer_success_view",
        "stakeholder_group_id": "customer_success_team",
        "issue": issue_value or issue(),
        "value_judgment_fingerprint": "6" * 64,
        "metadata": {"approval_state": "proposed"},
    }
    values.update(overrides)
    return build_stakeholder_perspective(**values)


def intervention(
    issue_value: AtomicIssueRecord | None = None,
    **overrides: Any,
) -> CandidateIntervention:
    """Return one deterministic intervention option."""
    values: dict[str, Any] = {
        "intervention_id": "billing_workflow_review",
        "intervention_family_id": "process_control",
        "issue": issue_value or issue(),
        "action_content_fingerprint": "7" * 64,
        "decision_policy_id": "human_review_policy",
        "metadata": {"causal_effect_status": "not_estimated"},
    }
    values.update(overrides)
    return build_candidate_intervention(**values)


def enterprise_request(**overrides: Any) -> EnterpriseIssueScoringRequest:
    """Return one deterministic criterion-level enterprise request."""
    issue_record = overrides.pop("issue_record", issue())
    values: dict[str, Any] = {
        "request_id": "enterprise_issue_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "issue": issue_record,
        "occasion_id": "initial_review",
        "task_id": "issue_evidence_review",
        "task_revision_fingerprint": "8" * 64,
        "task_family_id": "evidence_review",
        "criterion_ids": ("claim_support", "source_alignment"),
        "stakeholder_perspectives": (perspective(issue_record),),
        "candidate_interventions": (intervention(issue_record),),
        "metadata": {"deployment_stage": "offline_fixture"},
    }
    values.update(overrides)
    return build_enterprise_issue_scoring_request(**values)


def fixture_engine(request: EnterpriseIssueScoringRequest) -> StaticFixtureEngine:
    """Return one deterministic engine for the enterprise request."""
    evidence_reference = request.evidence_references[0]
    return StaticFixtureEngine(
        descriptor=automated_engine(),
        outcomes=(
            FixtureOutcome(
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
                evidence_references=(evidence_reference,),
            ),
            FixtureOutcome(
                criterion_id="source_alignment",
                status=ObservationStatus.SCORED,
                score_category=1,
                evidence_references=(evidence_reference,),
            ),
        ),
    )


def assert_error(code: str, callback) -> None:
    """Assert one stable enterprise/scoring contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_public_surface_is_explicit_and_documented() -> None:
    """The namespace exports only the reviewed enterprise adapter surface."""
    expected = {
        "AtomicIssueRecord",
        "CandidateIntervention",
        "CounterevidenceRecord",
        "EnterpriseAssertionKind",
        "EnterpriseIssueScoringRequest",
        "EnterpriseSourceRecord",
        "EvidenceSpanRecord",
        "MAX_ENTERPRISE_COUNTEREVIDENCE",
        "MAX_ENTERPRISE_EVIDENCE_SPANS",
        "MAX_ENTERPRISE_INTERVENTIONS",
        "MAX_ENTERPRISE_PERSPECTIVES",
        "MAX_ENTERPRISE_SOURCE_CHARACTERS",
        "MAX_ENTERPRISE_SOURCE_UNITS",
        "StakeholderPerspective",
        "build_atomic_issue_record",
        "build_candidate_intervention",
        "build_counterevidence_record",
        "build_enterprise_issue_scoring_request",
        "build_enterprise_source_record",
        "build_evidence_span_record",
        "build_stakeholder_perspective",
        "score_enterprise_issue_request",
    }
    assert set(enterprise.__all__) == expected
    assert all(
        getattr(enterprise, name).__doc__
        for name in expected
        if name[0].isupper()
    )


def test_source_is_content_addressed_immutable_and_source_text_free() -> None:
    """Equivalent source metadata yields stable identity without raw text."""
    metadata = {"nested_value": {"second_key": 2, "first_key": [1, 2]}}
    first = source(metadata=metadata)
    second = source(
        metadata={"nested_value": {"first_key": [1, 2], "second_key": 2}}
    )
    original = first.source_fingerprint
    metadata["nested_value"]["first_key"].append(3)

    assert first.source_fingerprint == second.source_fingerprint == original
    assert first.source_handle == f"enterprise_source_{original[:32]}"
    assert isinstance(first.metadata, MappingProxyType)
    assert first.to_dict()["subject_identifier_fingerprint"] == "2" * 64
    assert source(subject_identifier_fingerprint=None).subject_identifier_fingerprint is None
    assert_error(
        "sensitive_metadata_field",
        lambda: source(metadata={"source_text": "private content"}),
    )


def test_source_rejects_direct_and_invalid_counts() -> None:
    """Source construction fails closed on direct and malformed values."""
    assert_error(
        "unverified_enterprise_source",
        lambda: EnterpriseSourceRecord(
            source_id="customer_complaint",
            source_type_id="complaint_record",
            source_content_fingerprint="1" * 64,
            source_character_count=10,
            source_unit_count=1,
            subject_identifier_fingerprint=None,
            metadata={},
        ),
    )
    assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=True),
    )
    assert_error(
        "invalid_source_unit_count",
        lambda: source(source_unit_count="twenty"),
    )
    assert_error(
        "invalid_source_character_count",
        lambda: source(source_character_count=100_000_001),
    )


def test_evidence_assertion_kinds_map_to_shared_roles() -> None:
    """Every epistemic assertion kind maps to an explicit shared evidence role."""
    source_record = source()
    expectations = {
        EnterpriseAssertionKind.DIRECT_FACT: EvidenceRole.SUPPORTING,
        EnterpriseAssertionKind.SUPPORTED_INFERENCE: EvidenceRole.SUPPORTING,
        EnterpriseAssertionKind.COUNTEREVIDENCE: EvidenceRole.COUNTER,
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY: EvidenceRole.CONTEXT,
        EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT: EvidenceRole.CONTEXT,
    }
    for index, (kind, role) in enumerate(expectations.items()):
        record = evidence_span(
            source_record,
            span_id=f"evidence_span_{index}",
            content_fingerprint=f"{index + 3:x}" * 64,
            assertion_kind=kind,
            start_offset=index,
            end_offset=index + 1,
        )
        assert record.evidence_reference.evidence_role is role
        assert record.source_fingerprint == source_record.source_fingerprint
        assert record.evidence_handle.startswith("enterprise_evidence_")
        assert record.to_dict()["assertion_kind"] == kind.value


def test_evidence_rejects_invalid_sources_ranges_and_direct_values() -> None:
    """Evidence spans reject invalid source, kind, range, and shared-role claims."""
    source_record = source(source_character_count=20)
    assert_error(
        "invalid_enterprise_source",
        lambda: build_evidence_span_record(
            source=object(),
            span_id="fact_span",
            content_fingerprint="3" * 64,
            assertion_kind="direct_fact",
            start_offset=0,
            end_offset=1,
        ),
    )
    assert_error(
        "invalid_assertion_kind",
        lambda: evidence_span(source_record, assertion_kind="unsupported_kind"),
    )
    assert_error(
        "invalid_evidence_offsets",
        lambda: evidence_span(source_record, start_offset=10, end_offset=10),
    )
    assert_error(
        "invalid_end_offset",
        lambda: evidence_span(source_record, start_offset=10, end_offset=21),
    )
    assert_error(
        "unverified_evidence_span",
        lambda: EvidenceSpanRecord(
            evidence_reference=EvidenceReference(
                source_id=source_record.source_id,
                span_id="fact_span",
                content_fingerprint="3" * 64,
            ),
            source_fingerprint=source_record.source_fingerprint,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=0,
            end_offset=1,
            metadata={},
        ),
    )
    assert_error(
        "invalid_evidence_reference",
        lambda: EvidenceSpanRecord(
            evidence_reference=object(),
            source_fingerprint=source_record.source_fingerprint,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=0,
            end_offset=1,
            metadata={},
            _evidence_token=enterprise.contracts._EVIDENCE_TOKEN,
        ),
    )
    assert_error(
        "evidence_role_mismatch",
        lambda: EvidenceSpanRecord(
            evidence_reference=EvidenceReference(
                source_id=source_record.source_id,
                span_id="fact_span",
                content_fingerprint="3" * 64,
                evidence_role=EvidenceRole.COUNTER,
            ),
            source_fingerprint=source_record.source_fingerprint,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=0,
            end_offset=1,
            metadata={},
            _evidence_token=enterprise.contracts._EVIDENCE_TOKEN,
        ),
    )


def test_counterevidence_requires_counter_kind_and_is_content_addressed() -> None:
    """Counterevidence remains distinct from supporting and contextual spans."""
    record = counterevidence()
    assert record.evidence_span.assertion_kind is EnterpriseAssertionKind.COUNTEREVIDENCE
    assert record.counterevidence_handle.startswith("counterevidence_record_")
    assert record.to_dict()["target_claim_id"] == "recurrence_claim"
    assert_error(
        "invalid_counterevidence_kind",
        lambda: counterevidence(evidence_span=evidence_span()),
    )
    assert_error(
        "invalid_evidence_span",
        lambda: CounterevidenceRecord(
            counterevidence_id="contradictory_evidence",
            target_claim_id="recurrence_claim",
            evidence_span=object(),
            metadata={},
            _counterevidence_token=enterprise.contracts._COUNTEREVIDENCE_TOKEN,
        ),
    )
    assert_error(
        "unverified_counterevidence_record",
        lambda: CounterevidenceRecord(
            counterevidence_id="contradictory_evidence",
            target_claim_id="recurrence_claim",
            evidence_span=counter_span(),
            metadata={},
        ),
    )


def test_atomic_issue_preserves_evidence_separation_and_references() -> None:
    """Atomic issue identity retains supporting, contextual, and counter evidence."""
    source_record = source()
    supporting = evidence_span(source_record)
    ambiguity = evidence_span(
        source_record,
        span_id="ambiguity_span",
        content_fingerprint="9" * 64,
        assertion_kind=EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
        start_offset=90,
        end_offset=110,
    )
    counter = counterevidence(source_record)
    record = issue(
        evidence_spans=(ambiguity, supporting),
        counterevidence_records=(counter,),
    )

    assert record.issue_handle.startswith("atomic_issue_")
    assert [value.assertion_kind for value in record.evidence_spans] == [
        EnterpriseAssertionKind.DIRECT_FACT,
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
    ]
    assert len(record.evidence_references) == 3
    assert {value.evidence_role for value in record.evidence_references} == {
        EvidenceRole.SUPPORTING,
        EvidenceRole.COUNTER,
        EvidenceRole.CONTEXT,
    }
    assert record.to_dict()["issue_fingerprint"] == record.issue_fingerprint


def test_atomic_issue_rejects_empty_duplicate_misplaced_and_direct_values() -> None:
    """Atomic issues fail closed on missing or structurally invalid evidence."""
    supporting = evidence_span()
    counter = counterevidence()
    assert_error(
        "missing_issue_evidence",
        lambda: issue(evidence_spans=(), counterevidence_records=()),
    )
    assert_error(
        "duplicate_evidence_spans",
        lambda: issue(evidence_spans=(supporting, supporting)),
    )
    assert_error(
        "invalid_evidence_spans",
        lambda: issue(evidence_spans=(object(),)),
    )
    assert_error(
        "duplicate_counterevidence_records",
        lambda: issue(counterevidence_records=(counter, counter)),
    )
    assert_error(
        "invalid_counterevidence_records",
        lambda: issue(counterevidence_records=(object(),)),
    )
    assert_error(
        "misplaced_counterevidence_span",
        lambda: issue(evidence_spans=(counter_span(),)),
    )
    assert_error(
        "unverified_atomic_issue",
        lambda: AtomicIssueRecord(
            issue_id="billing_delay_issue",
            issue_family_id="service_reliability",
            domain_id="customer_operations",
            issue_content_fingerprint="5" * 64,
            issue_character_count=120,
            issue_unit_count=8,
            evidence_spans=(supporting,),
            counterevidence_records=(),
            metadata={},
        ),
    )


def test_perspectives_and_interventions_remain_issue_bound_and_noncausal() -> None:
    """Stakeholder judgments and intervention options retain separate identities."""
    issue_record = issue()
    perspective_record = perspective(issue_record)
    intervention_record = intervention(issue_record)

    assert perspective_record.issue_fingerprint == issue_record.issue_fingerprint
    assert perspective_record.perspective_handle.startswith("stakeholder_perspective_")
    assert intervention_record.issue_fingerprint == issue_record.issue_fingerprint
    assert intervention_record.intervention_handle.startswith("candidate_intervention_")
    assert intervention_record.to_dict()["decision_policy_id"] == "human_review_policy"
    assert intervention(
        issue_record,
        decision_policy_id=None,
    ).decision_policy_id is None
    assert_error(
        "invalid_atomic_issue",
        lambda: build_stakeholder_perspective(
            perspective_id="customer_success_view",
            stakeholder_group_id="customer_success_team",
            issue=object(),
            value_judgment_fingerprint="6" * 64,
        ),
    )
    assert_error(
        "invalid_atomic_issue",
        lambda: build_candidate_intervention(
            intervention_id="billing_workflow_review",
            intervention_family_id="process_control",
            issue=object(),
            action_content_fingerprint="7" * 64,
        ),
    )
    assert_error(
        "unverified_stakeholder_perspective",
        lambda: StakeholderPerspective(
            perspective_id="customer_success_view",
            stakeholder_group_id="customer_success_team",
            issue_fingerprint=issue_record.issue_fingerprint,
            value_judgment_fingerprint="6" * 64,
            metadata={},
        ),
    )
    assert_error(
        "unverified_candidate_intervention",
        lambda: CandidateIntervention(
            intervention_id="billing_workflow_review",
            intervention_family_id="process_control",
            issue_fingerprint=issue_record.issue_fingerprint,
            action_content_fingerprint="7" * 64,
            decision_policy_id=None,
            metadata={},
        ),
    )


def test_request_compiles_exact_provenance_into_shared_contract() -> None:
    """Enterprise adapters compile into one authoritative criterion request."""
    request = enterprise_request()
    shared = request.scoring_request

    assert shared.granularity.value == "criterion_level"
    assert shared.respondent_id == "billing_delay_issue"
    assert shared.response_id == "billing_delay_issue_record"
    assert shared.response_content_fingerprint == "5" * 64
    assert shared.task_revision_fingerprint == "8" * 64
    assert request.issue_fingerprint == issue().issue_fingerprint
    assert len(request.evidence_references) == 2
    assert len(request.perspective_fingerprints) == 1
    assert len(request.intervention_fingerprints) == 1
    assert request.request_handle.startswith("enterprise_issue_request_")
    metadata = shared.to_dict()["metadata"]
    assert metadata["enterprise_issue_fingerprint"] == request.issue_fingerprint
    assert metadata["enterprise_adapter_metadata"] == {
        "deployment_stage": "offline_fixture"
    }


def test_request_rejects_invalid_duplicates_and_cross_issue_replay() -> None:
    """Enterprise request compilation rejects invalid values and replayed records."""
    issue_record = issue()
    other_issue = issue(issue_content_fingerprint="a" * 64)
    perspective_record = perspective(issue_record)
    intervention_record = intervention(issue_record)
    assert_error(
        "invalid_atomic_issue",
        lambda: enterprise_request(issue_record=object()),
    )
    assert_error(
        "invalid_stakeholder_perspectives",
        lambda: enterprise_request(
            issue_record=issue_record,
            stakeholder_perspectives=(object(),),
        ),
    )
    assert_error(
        "duplicate_stakeholder_perspectives",
        lambda: enterprise_request(
            issue_record=issue_record,
            stakeholder_perspectives=(perspective_record, perspective_record),
        ),
    )
    assert_error(
        "invalid_candidate_interventions",
        lambda: enterprise_request(
            issue_record=issue_record,
            candidate_interventions=(object(),),
        ),
    )
    assert_error(
        "duplicate_candidate_interventions",
        lambda: enterprise_request(
            issue_record=issue_record,
            candidate_interventions=(intervention_record, intervention_record),
        ),
    )
    assert_error(
        "perspective_issue_mismatch",
        lambda: enterprise_request(
            issue_record=other_issue,
            stakeholder_perspectives=(perspective_record,),
            candidate_interventions=(),
        ),
    )
    assert_error(
        "intervention_issue_mismatch",
        lambda: enterprise_request(
            issue_record=other_issue,
            stakeholder_perspectives=(),
            candidate_interventions=(intervention_record,),
        ),
    )
    assert_error(
        "unverified_enterprise_issue_request",
        lambda: EnterpriseIssueScoringRequest(
            scoring_request=enterprise_request().scoring_request,
            issue_fingerprint=issue_record.issue_fingerprint,
            evidence_references=issue_record.evidence_references,
            perspective_fingerprints=(),
            intervention_fingerprints=(),
        ),
    )


def test_shared_engine_execution_preserves_authoritative_result_boundary() -> None:
    """Enterprise requests execute without defining a parallel result schema."""
    request = enterprise_request()
    engine = fixture_engine(request)
    result = score_enterprise_issue_request(engine, request)

    assert result.request_fingerprint == request.scoring_request.request_fingerprint
    assert result.engine_fingerprint == engine.descriptor.engine_fingerprint
    assert {value.criterion_id for value in result.observations} == {
        "claim_support",
        "source_alignment",
    }


def test_engine_execution_rejects_invalid_protocol_results_and_replay() -> None:
    """Engine execution fails closed on invalid protocol and provenance values."""
    request = enterprise_request()
    valid_engine = fixture_engine(request)
    valid_result = valid_engine.score(request.scoring_request)
    other_request = enterprise_request(
        issue_record=issue(issue_content_fingerprint="a" * 64),
        request_id="other_issue_request",
    )
    other_result = fixture_engine(other_request).score(other_request.scoring_request)

    class ResultEngine:
        def __init__(self, descriptor, result):
            self.descriptor = descriptor
            self._result = result

        def score(self, scoring_request):
            return self._result

    assert_error(
        "invalid_enterprise_issue_request",
        lambda: score_enterprise_issue_request(valid_engine, object()),
    )
    assert_error(
        "invalid_scoring_engine",
        lambda: score_enterprise_issue_request(object(), request),
    )
    assert_error(
        "invalid_scoring_result",
        lambda: score_enterprise_issue_request(
            ResultEngine(valid_engine.descriptor, object()),
            request,
        ),
    )
    assert_error(
        "invalid_engine_descriptor",
        lambda: score_enterprise_issue_request(
            ResultEngine(object(), valid_result),
            request,
        ),
    )
    assert_error(
        "enterprise_result_request_mismatch",
        lambda: score_enterprise_issue_request(
            ResultEngine(valid_engine.descriptor, other_result),
            request,
        ),
    )
    assert_error(
        "enterprise_result_engine_mismatch",
        lambda: score_enterprise_issue_request(
            ResultEngine(human_engine(), valid_result),
            request,
        ),
    )
