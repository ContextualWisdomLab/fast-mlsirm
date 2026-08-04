"""Core behavior for shared scoring-engine and observation contracts."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from types import MappingProxyType

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import (
    MAX_EVIDENCE_SPANS,
    OBSERVATION_SCHEMA_VERSION,
    EngineDescriptor,
    EvidenceSourceKind,
    EvidenceSpan,
    ObservationStatus,
    RaterKind,
    ScoreObservation,
    build_score_observation,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_observation_fixtures.py"))
)
argument_rubric = _FIXTURES["argument_rubric"]
assessment_spec = _FIXTURES["assessment_spec"]


def _automated_engine() -> EngineDescriptor:
    """Return one deterministic automated scorer descriptor."""
    return EngineDescriptor(
        engine_id="fixture_engine",
        engine_family_id="fixture_family",
        engine_version="1.2.3",
        rater_kind=RaterKind.AUTOMATED,
        prompt_template_id="analytic_template",
        prompt_template_version="2.0.0",
        metadata={"deployment_region": "offline_fixture"},
    )


def _spans() -> tuple[EvidenceSpan, ...]:
    """Return two evidence spans in deliberately non-canonical order."""
    return (
        EvidenceSpan(
            source_kind=EvidenceSourceKind.RESPONSE,
            source_id="essay_response",
            content_digest="b" * 64,
            start_offset=20,
            end_offset=35,
        ),
        EvidenceSpan(
            source_kind=EvidenceSourceKind.PROMPT,
            source_id="essay_prompt",
            content_digest="a" * 64,
            start_offset=0,
            end_offset=12,
        ),
    )


def _scored_observation(**overrides):
    """Build one valid scored observation and allow focused field overrides."""
    assessment = assessment_spec()
    rubric = argument_rubric()
    values = {
        "assessment": assessment,
        "rubrics": (rubric,),
        "engine": _automated_engine(),
        "response_id": "essay_response",
        "response_fingerprint": "b" * 64,
        "task_id": "essay_prompt",
        "task_fingerprint": "a" * 64,
        "rater_id": "automated_rater",
        "occasion_id": "scoring_occasion",
        "construct_id": "argument_quality",
        "rubric_fingerprint": rubric.fingerprint,
        "status": ObservationStatus.SCORED,
        "score_category": 1,
        "confidence": 0.75,
        "reason_code": None,
        "evidence_spans": _spans(),
        "metadata": {"batch_id": "pilot_batch"},
    }
    values.update(overrides)
    return build_score_observation(**values)


def test_engine_descriptor_is_content_addressed_deterministic_and_immutable():
    """Equivalent descriptors share one identity and cannot retain metadata aliases."""
    source_metadata = {"nested_metadata": {"worker_count": 4}}
    first = EngineDescriptor(
        engine_id="fixture_engine",
        engine_family_id="fixture_family",
        engine_version="1.2.3",
        rater_kind="automated",
        prompt_template_id="analytic_template",
        prompt_template_version="2.0.0",
        metadata=source_metadata,
    )
    second = EngineDescriptor(
        engine_id="fixture_engine",
        engine_family_id="fixture_family",
        engine_version="1.2.3",
        rater_kind=RaterKind.AUTOMATED,
        prompt_template_id="analytic_template",
        prompt_template_version="2.0.0",
        metadata={"nested_metadata": {"worker_count": 4}},
    )

    source_metadata["nested_metadata"]["worker_count"] = 9
    assert first == second
    assert first.rater_kind is RaterKind.AUTOMATED
    assert len(first.engine_fingerprint) == 64
    assert first.engine_handle == f"engine_descriptor_{first.engine_fingerprint[:32]}"
    assert first.metadata["nested_metadata"]["worker_count"] == 4
    assert isinstance(first.metadata, MappingProxyType)
    assert first.to_dict()["engine_fingerprint"] == first.engine_fingerprint


def test_evidence_span_validates_offsets_and_has_stable_content_identity():
    """Evidence retains only source provenance and bounded offsets, never raw text."""
    first = EvidenceSpan(
        source_kind="response",
        source_id="essay_response",
        content_digest="a" * 64,
        start_offset=1,
        end_offset=9,
    )
    second = EvidenceSpan(
        source_kind=EvidenceSourceKind.RESPONSE,
        source_id="essay_response",
        content_digest="a" * 64,
        start_offset=1,
        end_offset=9,
    )

    assert first == second
    assert first.source_kind is EvidenceSourceKind.RESPONSE
    assert len(first.span_fingerprint) == 64
    payload = first.to_dict()
    assert payload["span_fingerprint"] == first.span_fingerprint
    assert "content" not in payload
    assert "text" not in payload


def test_scored_observation_binds_exact_graph_and_normalizes_evidence_order():
    """A score binds exact assessment, rubric, engine, response, task, and evidence."""
    first = _scored_observation()
    second = _scored_observation(evidence_spans=tuple(reversed(_spans())))

    assert first == second
    assert first.schema_version == OBSERVATION_SCHEMA_VERSION
    assert first.status is ObservationStatus.SCORED
    assert first.score_category == 1
    assert first.reason_code is None
    assert first.response_fingerprint == "b" * 64
    assert first.task_fingerprint == "a" * 64
    assert first.assessment_fingerprint == assessment_spec().assessment_fingerprint
    assert first.engine_fingerprint == _automated_engine().engine_fingerprint
    assert first.evidence_spans == tuple(
        sorted(first.evidence_spans, key=lambda span: span.span_fingerprint)
    )
    assert len(first.observation_fingerprint) == 64
    assert first.observation_handle == (
        f"score_observation_{first.observation_fingerprint[:32]}"
    )
    payload = first.to_dict()
    assert payload["observation_fingerprint"] == first.observation_fingerprint
    assert payload["assessment_fingerprint"] == first.assessment_fingerprint
    assert payload["engine"]["engine_fingerprint"] == first.engine_fingerprint
    assert payload["response_fingerprint"] == first.response_fingerprint
    assert payload["task_fingerprint"] == first.task_fingerprint


def test_response_and_task_fingerprints_change_observation_identity():
    """Reusing labels for changed artifacts cannot preserve an observation identity."""
    baseline = _scored_observation()
    changed_response = _scored_observation(response_fingerprint="c" * 64)
    changed_task = _scored_observation(task_fingerprint="d" * 64)

    assert changed_response.response_id == baseline.response_id
    assert changed_task.task_id == baseline.task_id
    assert changed_response.observation_fingerprint != baseline.observation_fingerprint
    assert changed_task.observation_fingerprint != baseline.observation_fingerprint


def test_non_scored_observation_states_preserve_reason_without_score():
    """Abstained, failed, and excluded outputs remain distinct and auditable."""
    for status, reason in (
        (ObservationStatus.ABSTAINED, "insufficient_evidence"),
        (ObservationStatus.FAILED, "engine_failure"),
        (ObservationStatus.EXCLUDED, "policy_exclusion"),
    ):
        observation = _scored_observation(
            status=status,
            score_category=None,
            confidence=None,
            reason_code=reason,
            evidence_spans=(),
        )
        assert observation.status is status
        assert observation.score_category is None
        assert observation.reason_code == reason


def test_observation_direct_construction_is_factory_sealed():
    """Cross-reference validation cannot be bypassed with a public constructor."""
    valid = _scored_observation()
    with pytest.raises(Exception, match="build_score_observation"):
        ScoreObservation(
            assessment_fingerprint=valid.assessment_fingerprint,
            engine=valid.engine,
            response_id=valid.response_id,
            response_fingerprint=valid.response_fingerprint,
            task_id=valid.task_id,
            task_fingerprint=valid.task_fingerprint,
            rater_id=valid.rater_id,
            occasion_id=valid.occasion_id,
            construct_id=valid.construct_id,
            rubric_fingerprint=valid.rubric_fingerprint,
            status=valid.status,
            score_category=valid.score_category,
            confidence=valid.confidence,
            reason_code=valid.reason_code,
            evidence_spans=valid.evidence_spans,
            metadata=valid.metadata,
        )


def test_observation_public_exports_are_explicit_and_documented():
    """The shared namespace exposes the complete observation contract surface."""
    expected_additions = {
        "OBSERVATION_SCHEMA_VERSION",
        "MAX_EVIDENCE_SPANS",
        "EngineDescriptor",
        "EvidenceSourceKind",
        "EvidenceSpan",
        "ObservationStatus",
        "RaterKind",
        "ScoreObservation",
        "build_score_observation",
    }
    assert expected_additions.issubset(set(scoring.__all__))
    for name in expected_additions:
        value = getattr(scoring, name)
        assert value is globals()[name]
        if callable(value):
            assert inspect.getdoc(value)
    assert OBSERVATION_SCHEMA_VERSION == "1.0"
    assert MAX_EVIDENCE_SPANS == 64
