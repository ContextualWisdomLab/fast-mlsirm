"""Fail-closed validation for scoring-engine and observation contracts."""

from __future__ import annotations

import math
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import RubricSpecification
from fast_mlsirm.scoring import (
    AssessmentSpecError,
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


def _engine(*, rater_kind: RaterKind = RaterKind.AUTOMATED, engine_id: str = "fixture_engine"):
    """Return one scorer descriptor for validation tests."""
    return EngineDescriptor(
        engine_id=engine_id,
        engine_family_id=(
            "fixture_family" if rater_kind is RaterKind.AUTOMATED else "human_family"
        ),
        engine_version="1.0.0",
        rater_kind=rater_kind,
        prompt_template_id="analytic_template",
        prompt_template_version="1.0.0",
    )


def _build(**overrides):
    """Build one valid observation and allow focused invalid overrides."""
    rubric = argument_rubric()
    values = {
        "assessment": assessment_spec(),
        "rubrics": (rubric,),
        "engine": _engine(),
        "response_id": "essay_response",
        "task_id": "essay_prompt",
        "rater_id": "automated_rater",
        "occasion_id": "scoring_occasion",
        "construct_id": "argument_quality",
        "rubric_fingerprint": rubric.fingerprint,
        "status": ObservationStatus.SCORED,
        "score_category": 2,
        "confidence": 0.75,
        "reason_code": None,
        "evidence_spans": (),
    }
    values.update(overrides)
    return build_score_observation(**values)


def test_engine_policy_controls_human_and_automated_descriptors():
    """Automated engines are allowlisted and human raters require policy support."""
    with pytest.raises(AssessmentSpecError) as automated_disabled:
        _build(
            assessment=assessment_spec(allow_human=True, allow_automated=False),
            engine=_engine(),
        )
    assert automated_disabled.value.code == "automated_rater_disabled"

    with pytest.raises(AssessmentSpecError) as engine_unknown:
        _build(engine=_engine(engine_id="unknown_engine"))
    assert engine_unknown.value.code == "unknown_engine_id"

    human = _build(engine=_engine(rater_kind=RaterKind.HUMAN, engine_id="human_adapter"))
    assert human.engine.rater_kind is RaterKind.HUMAN

    with pytest.raises(AssessmentSpecError) as human_disabled:
        _build(
            assessment=assessment_spec(allow_human=False, allow_automated=True),
            engine=_engine(rater_kind=RaterKind.HUMAN, engine_id="human_adapter"),
        )
    assert human_disabled.value.code == "human_rater_disabled"


def test_observation_requires_exact_assessment_construct_and_rubric_graph():
    """Unknown or mismatched rubric and construct references fail closed."""
    with pytest.raises(AssessmentSpecError) as construct_error:
        _build(construct_id="unknown_construct")
    assert construct_error.value.code == "unknown_observation_construct"

    with pytest.raises(AssessmentSpecError) as fingerprint_error:
        _build(rubric_fingerprint="f" * 64)
    assert fingerprint_error.value.code == "unknown_observation_rubric"

    other_rubric = RubricSpecification(
        **{
            **argument_rubric().__dict__,
            "rubric_id": "alternate_rubric",
            "construct_id": "alternate_construct",
            "construct_definition": "A different construct.",
        }
    )
    with pytest.raises(AssessmentSpecError) as registry_error:
        _build(rubrics=(argument_rubric(), other_rubric))
    assert registry_error.value.code == "unused_observation_rubric"

    with pytest.raises(AssessmentSpecError) as duplicate_error:
        rubric = argument_rubric()
        _build(rubrics=(rubric, rubric))
    assert duplicate_error.value.code == "duplicate_observation_rubric"


def test_observation_status_score_and_reason_contracts_are_exclusive():
    """Each status has one unambiguous score/reason representation."""
    with pytest.raises(AssessmentSpecError) as missing_score:
        _build(score_category=None)
    assert missing_score.value.code == "missing_score_category"

    with pytest.raises(AssessmentSpecError) as scored_reason:
        _build(reason_code="unexpected_reason")
    assert scored_reason.value.code == "scored_reason_forbidden"

    with pytest.raises(AssessmentSpecError) as invalid_score:
        _build(score_category=1)
    assert invalid_score.value.code == "invalid_score_category"

    with pytest.raises(AssessmentSpecError) as boolean_score:
        _build(score_category=True)
    assert boolean_score.value.code == "invalid_score_category"

    for status in (
        ObservationStatus.ABSTAINED,
        ObservationStatus.FAILED,
        ObservationStatus.EXCLUDED,
    ):
        with pytest.raises(AssessmentSpecError) as score_forbidden:
            _build(status=status, score_category=2, reason_code="review_reason")
        assert score_forbidden.value.code == "non_scored_category_forbidden"

        with pytest.raises(AssessmentSpecError) as reason_missing:
            _build(status=status, score_category=None, reason_code=None)
        assert reason_missing.value.code == "missing_reason_code"


def test_confidence_is_optional_finite_numeric_probability_not_boolean():
    """Confidence has one portable bounded representation without numeric coercion."""
    assert _build(confidence=None).confidence is None
    for value in (True, -0.1, 1.1, math.nan, math.inf, "high"):
        with pytest.raises(AssessmentSpecError) as captured:
            _build(confidence=value)
        assert captured.value.code == "invalid_observation_confidence"


def test_evidence_span_rejects_invalid_identifiers_digests_and_offsets():
    """Evidence provenance cannot contain ambiguous, unbounded, or reversed spans."""
    invalid_cases = (
        {"source_kind": "unknown", "source_id": "essay_response", "content_digest": "a" * 64, "start_offset": 0, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "invalid", "content_digest": "a" * 64, "start_offset": 0, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "essay_response", "content_digest": "bad", "start_offset": 0, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "essay_response", "content_digest": "a" * 64, "start_offset": -1, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "essay_response", "content_digest": "a" * 64, "start_offset": 1, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "essay_response", "content_digest": "a" * 64, "start_offset": True, "end_offset": 1},
        {"source_kind": EvidenceSourceKind.RESPONSE, "source_id": "essay_response", "content_digest": "a" * 64, "start_offset": 0, "end_offset": 1 << 63},
    )
    for values in invalid_cases:
        with pytest.raises(AssessmentSpecError):
            EvidenceSpan(**values)


def test_observation_rejects_duplicate_or_oversized_evidence_spans():
    """Evidence lists are bounded and each exact span appears at most once."""
    span = EvidenceSpan(
        source_kind=EvidenceSourceKind.RESPONSE,
        source_id="essay_response",
        content_digest="a" * 64,
        start_offset=0,
        end_offset=1,
    )
    with pytest.raises(AssessmentSpecError) as duplicate:
        _build(evidence_spans=(span, span))
    assert duplicate.value.code == "duplicate_evidence_span"

    spans = tuple(
        EvidenceSpan(
            source_kind=EvidenceSourceKind.RESPONSE,
            source_id=f"evidence_source_{index}",
            content_digest=f"{index:064x}",
            start_offset=0,
            end_offset=1,
        )
        for index in range(65)
    )
    with pytest.raises(AssessmentSpecError) as oversized:
        _build(evidence_spans=spans)
    assert oversized.value.code == "invalid_evidence_spans"


def test_engine_and_observation_identifiers_versions_enums_and_metadata_fail_closed():
    """Public values use stable descriptive identities and bounded metadata."""
    with pytest.raises(AssessmentSpecError):
        _engine(engine_id="invalid")
    with pytest.raises(AssessmentSpecError):
        EngineDescriptor(
            engine_id="fixture_engine",
            engine_family_id="fixture_family",
            engine_version="not-semver",
            rater_kind=RaterKind.AUTOMATED,
            prompt_template_id="analytic_template",
            prompt_template_version="1.0.0",
        )
    with pytest.raises(AssessmentSpecError):
        EngineDescriptor(
            engine_id="fixture_engine",
            engine_family_id="fixture_family",
            engine_version="1.0.0",
            rater_kind="unknown",
            prompt_template_id="analytic_template",
            prompt_template_version="1.0.0",
        )
    with pytest.raises(AssessmentSpecError):
        _build(status="unknown")
    with pytest.raises(AssessmentSpecError):
        _build(response_id="invalid")
    with pytest.raises(AssessmentSpecError):
        _build(metadata={"response_text": "secret"})


def test_observation_factory_validates_owned_component_types_and_schema_version():
    """Parallel contract implementations and incompatible schema versions are rejected."""
    with pytest.raises(AssessmentSpecError) as assessment_error:
        _build(assessment=object())
    assert assessment_error.value.code == "invalid_observation_assessment"

    with pytest.raises(AssessmentSpecError) as engine_error:
        _build(engine=object())
    assert engine_error.value.code == "invalid_observation_engine"

    valid = _build()
    with pytest.raises(AssessmentSpecError) as direct_error:
        ScoreObservation(
            assessment_fingerprint=valid.assessment_fingerprint,
            engine=valid.engine,
            response_id=valid.response_id,
            task_id=valid.task_id,
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
            schema_version="9.9",
        )
    assert direct_error.value.code == "unverified_score_observation"
