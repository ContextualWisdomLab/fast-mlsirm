"""Review-pinned enterprise calibration replay and privacy tests."""

from __future__ import annotations

import pytest

from enterprise_issue_calibration_fixtures import (
    _digest,
    _engine,
    _execution,
    _issue,
    _managed_observation_metadata,
    _request,
    _result,
    _result_with_replacement,
)
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    ObservationStatus,
    build_score_observation,
    build_scoring_facets_calibration_bundle,
)
from fast_mlsirm.scoring.enterprise_issue import (
    build_enterprise_issue_facets_rating_records,
    enterprise_issue_evidence_references,
)


def _assert_evidence_gate(callback) -> None:
    """Assert the exact observation-evidence replay gate, not a later gate."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == "enterprise_calibration_provenance_mismatch"
    assert captured.value.path == "$.result.observations[0].evidence_references"


def test_undeclared_evidence_is_rejected_by_the_evidence_gate() -> None:
    """Undeclared evidence fails before observation-metadata replay."""
    issue, request, _, engine = _execution(
        issue_label="review_unknown",
        task_label="review_unknown",
        engine_label="alpha",
        scores=(1, 2),
    )
    unknown = EvidenceReference(
        source_id="unknown_source",
        span_id="unknown_span",
        content_fingerprint=_digest("unknown-evidence"),
        evidence_role=EvidenceRole.SUPPORTING,
    )
    replacement = build_score_observation(
        observation_id="review_unknown_evidence",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(unknown,),
        confidence_metadata=_managed_observation_metadata(issue, (unknown,)),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )

    _assert_evidence_gate(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        )
    )


def test_missing_support_is_rejected_by_the_supporting_evidence_gate() -> None:
    """A scored observation without support fails at its intended gate."""
    issue, request, _, engine = _execution(
        issue_label="review_support",
        task_label="review_support",
        engine_label="alpha",
        scores=(1, 2),
    )
    replacement = build_score_observation(
        observation_id="review_missing_support",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(),
        confidence_metadata=_managed_observation_metadata(issue, ()),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )

    _assert_evidence_gate(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        )
    )


def test_missing_counterevidence_is_rejected_by_the_counter_gate() -> None:
    """Supplied counterevidence cannot be hidden by valid metadata."""
    issue = _issue("review_counter", include_counterevidence=True)
    engine = _engine("alpha")
    request = _request(
        issue,
        task_label="review_counter",
        engine_label="alpha",
    )
    supporting = tuple(
        value
        for value in enterprise_issue_evidence_references(issue)
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    replacement = build_score_observation(
        observation_id="review_missing_counter",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=supporting,
        confidence_metadata=_managed_observation_metadata(issue, supporting),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )

    _assert_evidence_gate(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        )
    )


def test_shared_bundle_excludes_real_upstream_enterprise_provenance() -> None:
    """Request-only source provenance and caller metadata do not enter the bundle."""
    issue = _issue("privacy_sentinel")
    engine = _engine("alpha")
    request = _request(
        issue,
        task_label="privacy_sentinel",
        engine_label="alpha",
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(1, 2),
    )
    records = build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    )
    serialized_request = repr(request.to_dict())
    serialized_bundle = repr(
        build_scoring_facets_calibration_bundle(records).to_dict()
    )
    source_record_fingerprint = issue.source_record_fingerprints[0]

    assert source_record_fingerprint in serialized_request
    assert "offline_fixture" in serialized_request
    assert source_record_fingerprint not in serialized_bundle
    assert "offline_fixture" not in serialized_bundle
