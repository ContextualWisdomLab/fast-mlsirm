"""Governed enterprise issue calibration handoff tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
import fast_mlsirm.scoring.enterprise_issue.calibration as calibration_module
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EngineDescriptor,
    EvidenceReference,
    EvidenceRole,
    ObservationStatus,
    ScoringRequest,
    ScoringResult,
    build_score_observation,
    build_scoring_facets_calibration_bundle,
    build_scoring_request,
    build_scoring_result,
)
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    build_enterprise_issue_facets_rating_records,
    build_enterprise_issue_score_observation,
    build_enterprise_issue_scoring_request,
    enterprise_issue_evidence_references,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]
automated_engine = _FIXTURES["automated_engine"]

CRITERION_IDS = ("claim_support", "source_alignment")


def _digest(value: str) -> str:
    """Return one deterministic SHA-256 fixture fingerprint."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _issue(
    label: str,
    *,
    include_counterevidence: bool = False,
) -> AtomicIssueRecord:
    """Return one source-text-free issue revision for calibration tests."""
    issue_content_fingerprint = _digest(f"issue-content:{label}")
    source = EnterpriseSourceRecord(
        source_id=f"source_{label}",
        source_family_id="enterprise_source",
        source_content_fingerprint=_digest(f"source-content:{label}"),
        source_character_count=240,
        metadata={"source_channel": "offline_fixture"},
    )
    supporting = EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id=f"supporting_{label}",
        span_content_fingerprint=_digest(f"supporting-span:{label}"),
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=10,
        end_offset=30,
        metadata={"fixture_kind": "supporting_evidence"},
    )
    counterevidence_records: tuple[CounterevidenceRecord, ...] = ()
    if include_counterevidence:
        counter_span = EvidenceSpanRecord(
            source_id=source.source_id,
            source_record_fingerprint=source.source_record_fingerprint,
            span_id=f"counter_{label}",
            span_content_fingerprint=_digest(f"counter-span:{label}"),
            assertion_kind=EnterpriseAssertionKind.COUNTEREVIDENCE,
            start_offset=40,
            end_offset=60,
            metadata={"fixture_kind": "counter_evidence"},
        )
        counterevidence_records = (
            CounterevidenceRecord(
                counterevidence_id=f"counter_record_{label}",
                issue_content_fingerprint=issue_content_fingerprint,
                evidence_span=counter_span,
                metadata={"verification_state": "source_verified"},
            ),
        )
    return AtomicIssueRecord(
        issue_id=f"issue_{label}",
        issue_family_id="service_risk",
        issue_content_fingerprint=issue_content_fingerprint,
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(supporting,),
        counterevidence_records=counterevidence_records,
        metadata={"decision_scope": "offline_pilot"},
    )


def _engine(label: str) -> EngineDescriptor:
    """Return one deterministic automated judge identity."""
    return automated_engine(
        engine_id=f"engine_{label}",
        engine_family_id=f"engine_family_{label}",
        model_id=f"model_{label}",
        prompt_template_fingerprint=_digest(f"prompt:{label}"),
    )


def _request(
    issue: AtomicIssueRecord,
    *,
    task_label: str,
    engine_label: str,
) -> ScoringRequest:
    """Return one exact enterprise criterion-level request."""
    return build_enterprise_issue_scoring_request(
        request_id=f"request_{issue.issue_id}_{task_label}_{engine_label}",
        assessment=assessment(),
        rubric=rubric(),
        issue=issue,
        response_id=f"response_{issue.issue_id}_{task_label}",
        task_id=f"task_{task_label}",
        task_revision_fingerprint=_digest(f"task-revision:{task_label}"),
        task_family_id="evidence_review",
        occasion_id="pilot_occasion",
        criterion_ids=CRITERION_IDS,
        response_character_count=160,
        response_unit_count=8,
        metadata={"deployment_stage": "offline_fixture"},
    )


def _enterprise_observations(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    scores: tuple[int, int],
    abstain_first: bool = False,
):
    """Return complete enterprise observations for one request."""
    references = enterprise_issue_evidence_references(issue)
    observations = []
    for index, (criterion_id, score) in enumerate(
        zip(CRITERION_IDS, scores, strict=True)
    ):
        abstained = abstain_first and index == 0
        observations.append(
            build_enterprise_issue_score_observation(
                observation_id=(
                    f"observation_{issue.issue_id}_{request.task_id}_"
                    f"{engine.engine_id}_{criterion_id}"
                ),
                request=request,
                engine=engine,
                criterion_id=criterion_id,
                status=(
                    ObservationStatus.ABSTAINED
                    if abstained
                    else ObservationStatus.SCORED
                ),
                score_category=None if abstained else score,
                reason_code="insufficient_evidence" if abstained else None,
                evidence_references=() if abstained else references,
                confidence_metadata={"review_state": "fixture_complete"},
            )
        )
    return tuple(observations)


def _result(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    scores: tuple[int, int],
    abstain_first: bool = False,
) -> ScoringResult:
    """Return one complete governed enterprise result."""
    return build_scoring_result(
        result_id=f"result_{request.request_id}_{engine.engine_id}",
        request=request,
        engine=engine,
        observations=_enterprise_observations(
            issue=issue,
            request=request,
            engine=engine,
            scores=scores,
            abstain_first=abstain_first,
        ),
        execution_attempt=1,
        diagnostics={"execution_mode": "offline_fixture"},
    )


def _execution(
    *,
    issue_label: str,
    task_label: str,
    engine_label: str,
    scores: tuple[int, int],
):
    """Return one matched issue, request, result, and engine execution."""
    issue = _issue(issue_label)
    engine = _engine(engine_label)
    request = _request(
        issue,
        task_label=task_label,
        engine_label=engine_label,
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=scores,
    )
    return issue, request, result, engine


def _assert_error(code: str, callback) -> None:
    """Assert one stable governed scoring error code."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code


def _rebuild_request(
    request: ScoringRequest,
    **overrides: Any,
) -> ScoringRequest:
    """Rebuild one request through the shared factory for adversarial replay."""
    values: dict[str, Any] = {
        "request_id": request.request_id,
        "assessment": assessment(),
        "rubric": rubric(),
        "granularity": request.granularity,
        "respondent_id": request.respondent_id,
        "response_id": request.response_id,
        "task_id": request.task_id,
        "task_revision_fingerprint": request.task_revision_fingerprint,
        "task_family_id": request.task_family_id,
        "occasion_id": request.occasion_id,
        "criterion_ids": request.criterion_ids,
        "response_content_fingerprint": request.response_content_fingerprint,
        "response_character_count": request.response_character_count,
        "response_unit_count": request.response_unit_count,
        "metadata": request.to_dict()["metadata"],
    }
    values.update(overrides)
    return build_scoring_request(**values)


def _result_with_replacement(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    replacement,
) -> ScoringResult:
    """Replace the claim observation while retaining complete result coverage."""
    observations = list(
        _enterprise_observations(
            issue=issue,
            request=request,
            engine=engine,
            scores=(1, 2),
        )
    )
    observations[0] = replacement
    return build_scoring_result(
        result_id=f"result_replaced_{request.task_id}_{engine.engine_id}",
        request=request,
        engine=engine,
        observations=observations,
        execution_attempt=1,
        diagnostics={},
    )


def test_public_surface_and_shared_bundle_preserve_exact_enterprise_identity() -> None:
    """Enterprise executions become only the existing shared facets contracts."""
    executions = (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )
    records = tuple(
        record
        for issue, request, result, engine in executions
        for record in build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )
    bundle = build_scoring_facets_calibration_bundle(records)

    assert "build_enterprise_issue_facets_rating_records" in enterprise.__all__
    assert build_enterprise_issue_facets_rating_records.__doc__
    assert bundle.criterion_ids == CRITERION_IDS
    assert all(
        design.respondent_ids == ("issue_alpha", "issue_beta")
        for design in bundle.designs
    )
    assert all(
        design.task_ids == ("task_alpha", "task_beta")
        for design in bundle.designs
    )
    assert all(design.connected for design in bundle.designs)
    assert {record.response_content_fingerprint for record in records} == {
        _digest("issue-content:alpha"),
        _digest("issue-content:beta"),
    }
    assert "source text" not in repr(bundle.to_dict()).lower()


def test_execution_order_does_not_change_shared_bundle_identity() -> None:
    """Input execution order cannot become a hidden calibration feature."""
    executions = (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )

    def build(values):
        records = tuple(
            record
            for issue, request, result, engine in values
            for record in build_enterprise_issue_facets_rating_records(
                issue=issue,
                request=request,
                result=result,
                engine=engine,
            )
        )
        return build_scoring_facets_calibration_bundle(records)

    assert build(executions).bundle_fingerprint == build(
        tuple(reversed(executions))
    ).bundle_fingerprint


def test_projection_delegates_to_the_shared_rating_builder(monkeypatch) -> None:
    """The enterprise adapter performs replay and delegates record projection."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    captured: dict[str, Any] = {}
    sentinel = (object(),)

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        calibration_module,
        "build_scoring_facets_rating_records",
        fake_builder,
    )
    assert build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    ) is sentinel
    assert captured == {
        "request": request,
        "result": result,
        "engine": engine,
    }


@pytest.mark.parametrize(
    ("field_name", "code"),
    (
        ("issue", "invalid_atomic_issue"),
        ("request", "invalid_scoring_request"),
        ("result", "invalid_scoring_result"),
        ("engine", "invalid_engine_descriptor"),
    ),
)
def test_exact_public_contract_types_are_required(
    field_name: str,
    code: str,
) -> None:
    """Counterfeit top-level contract values fail before provenance access."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    values = {
        "issue": issue,
        "request": request,
        "result": result,
        "engine": engine,
    }
    values[field_name] = object()
    _assert_error(
        code,
        lambda: build_enterprise_issue_facets_rating_records(**values),
    )


def test_request_must_have_enterprise_provenance() -> None:
    """A generic shared request cannot masquerade as an enterprise request."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    generic_request = _rebuild_request(request, metadata={})
    _assert_error(
        "missing_enterprise_request_provenance",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=generic_request,
            result=result,
            engine=engine,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    ("atomic", "content", "respondent", "response_revision"),
)
def test_request_issue_binding_replays_every_identity(mutation: str) -> None:
    """Issue, content, respondent, and response-revision drift fail closed."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    metadata = request.to_dict()["metadata"]
    overrides: dict[str, Any] = {}
    if mutation == "atomic":
        metadata["enterprise_atomic_issue_fingerprint"] = _digest(
            "other-atomic"
        )
        overrides["metadata"] = metadata
    elif mutation == "content":
        metadata["enterprise_issue_content_fingerprint"] = _digest(
            "other-content"
        )
        overrides["metadata"] = metadata
    elif mutation == "respondent":
        overrides["respondent_id"] = "issue_other"
    else:
        overrides["response_content_fingerprint"] = _digest("other-response")
    changed_request = _rebuild_request(request, **overrides)

    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=changed_request,
            result=result,
            engine=engine,
        ),
    )


def test_undeclared_observation_evidence_fails_replay() -> None:
    """Observation evidence must remain a subset of the exact request packet."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_unknown_evidence",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(
            EvidenceReference(
                source_id="unknown_source",
                span_id="unknown_span",
                content_fingerprint=_digest("unknown-evidence"),
                evidence_role=EvidenceRole.SUPPORTING,
            ),
        ),
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
    )


def test_non_abstained_observation_requires_supporting_evidence() -> None:
    """A generic scored observation cannot bypass the enterprise evidence gate."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_support",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(),
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
    )


def test_declared_counterevidence_must_survive_calibration_replay() -> None:
    """Counterevidence cannot disappear before shared calibration."""
    issue = _issue("counter", include_counterevidence=True)
    engine = _engine("counter")
    request = _request(issue, task_label="counter", engine_label="counter")
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
    assert len(records) == 2

    supporting = tuple(
        value
        for value in enterprise_issue_evidence_references(issue)
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_counter",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=supporting,
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
    )


def test_observation_managed_metadata_must_replay_exactly() -> None:
    """Generic confidence metadata cannot counterfeit enterprise observations."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_metadata",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=enterprise_issue_evidence_references(issue),
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
    )


def test_abstention_remains_a_terminal_missing_rating() -> None:
    """Insufficient evidence remains abstention rather than a fabricated score."""
    issue = _issue("abstention")
    engine = _engine("abstention")
    request = _request(
        issue,
        task_label="abstention",
        engine_label="abstention",
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(0, 2),
        abstain_first=True,
    )
    records = build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    )

    by_criterion = {record.criterion_id: record for record in records}
    assert by_criterion["claim_support"].status is ObservationStatus.ABSTAINED
    assert by_criterion["claim_support"].score_category is None
    assert by_criterion["source_alignment"].score_category == 2
