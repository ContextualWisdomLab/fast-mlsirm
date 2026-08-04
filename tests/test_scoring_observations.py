"""Behavioral contracts for lossless scoring observations."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceSpan,
    ObservationLevel,
    ObservationStatus,
    RaterKind,
    ScoreObservation,
    artifact_digest,
    build_score_observation,
    canonical_json,
    validate_observations,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_observation_fixtures.py"))
)
approved_assessment = _FIXTURES["approved_assessment"]
approved_rubrics = _FIXTURES["approved_rubrics"]
argument_rubric = _FIXTURES["argument_rubric"]


def _observation(**overrides):
    values = {
        "assessment": approved_assessment(),
        "rubrics": approved_rubrics(),
        "observation_id": "argument_score_observation",
        "rubric_fingerprint": argument_rubric().fingerprint,
        "response_id": "essay_response_alpha",
        "rater_id": "automated_rater_alpha",
        "rater_kind": RaterKind.AUTOMATED,
        "engine_id": "fixture_engine",
        "construct_id": "argument_quality",
        "observation_level": ObservationLevel.CRITERION_LEVEL,
        "criterion_id": "evidence_alignment",
        "status": ObservationStatus.SCORED,
        "score_category": 1,
        "reason_code": None,
        "confidence": 0.75,
        "evidence_spans": (
            EvidenceSpan(
                source_id="essay_response_alpha",
                start_offset=4,
                end_offset=12,
                evidence_label="supporting_span",
            ),
        ),
        "occasion_id": "scoring_occasion_alpha",
        "scorer_family": "deterministic_fixture",
        "scorer_version": "1.0.0",
        "prompt_template_version": "1.0.0",
        "metadata": {"review_mode": "offline_fixture"},
    }
    values.update(overrides)
    return build_score_observation(**values)


def test_automated_scored_observation_binds_exact_provenance() -> None:
    """A scored observation retains exact assessment, rubric, and engine identity."""
    observation = _observation()

    assert observation.status is ObservationStatus.SCORED
    assert observation.score_category == 1
    assert observation.assessment_fingerprint == (
        approved_assessment().assessment_fingerprint
    )
    assert observation.rubric_fingerprint == argument_rubric().fingerprint
    assert observation.engine_id == "fixture_engine"
    assert len(observation.observation_fingerprint) == 64
    assert observation.observation_handle == (
        f"score_observation_{observation.observation_fingerprint[:32]}"
    )
    assert artifact_digest(observation) == observation.observation_fingerprint
    assert observation.to_dict()["observation_fingerprint"] == (
        observation.observation_fingerprint
    )
    payload = canonical_json(observation)
    assert "response_text" not in payload
    assert "source_content" not in payload


@pytest.mark.parametrize(
    ("status", "reason_code"),
    [
        (ObservationStatus.MISSING, None),
        (ObservationStatus.ABSTAINED, "insufficient_evidence"),
        (ObservationStatus.FAILED, "engine_execution_failed"),
        (ObservationStatus.EXCLUDED, "response_out_of_scope"),
    ],
)
def test_non_scored_states_remain_distinct(
    status: ObservationStatus,
    reason_code: str | None,
) -> None:
    """Missing, abstained, failed, and excluded states never become scores."""
    observation = _observation(
        observation_id=f"{status.value}_score_observation",
        status=status,
        score_category=None,
        reason_code=reason_code,
    )

    assert observation.status is status
    assert observation.score_category is None
    assert observation.reason_code == reason_code
    assert f'"status":"{status.value}"' in canonical_json(observation)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"score_category": None}, "missing_score_category"),
        ({"reason_code": "unexpected_reason"}, "unexpected_reason_code"),
        (
            {
                "status": ObservationStatus.ABSTAINED,
                "score_category": 1,
                "reason_code": "insufficient_evidence",
            },
            "unexpected_score_category",
        ),
        (
            {
                "status": ObservationStatus.ABSTAINED,
                "score_category": None,
                "reason_code": None,
            },
            "missing_reason_code",
        ),
        (
            {
                "status": ObservationStatus.MISSING,
                "score_category": None,
                "reason_code": "unassigned_cell",
            },
            "unexpected_reason_code",
        ),
        ({"score_category": 9}, "unknown_score_category"),
    ],
)
def test_status_and_category_contracts_fail_closed(overrides, code: str) -> None:
    """Invalid score/status combinations return stable domain errors."""
    with pytest.raises(AssessmentSpecError) as captured:
        _observation(**overrides)

    assert captured.value.code == code


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"criterion_id": None}, "missing_criterion_id"),
        (
            {
                "observation_level": ObservationLevel.HOLISTIC,
                "criterion_id": "unexpected_criterion",
            },
            "unexpected_criterion_id",
        ),
        (
            {
                "rater_kind": RaterKind.HUMAN,
                "rater_id": "human_rater_alpha",
                "engine_id": "fixture_engine",
            },
            "unexpected_engine_id",
        ),
        ({"engine_id": None}, "missing_engine_id"),
        ({"engine_id": "unknown_engine"}, "unknown_engine_id"),
    ],
)
def test_level_and_rater_engine_contracts_fail_closed(overrides, code: str) -> None:
    """Observation level and rater provenance cannot be interchanged."""
    with pytest.raises(AssessmentSpecError) as captured:
        _observation(**overrides)

    assert captured.value.code == code


def test_human_holistic_observation_has_no_engine_or_criterion() -> None:
    """Human holistic observations remain distinct and engine-free."""
    observation = _observation(
        observation_id="holistic_human_observation",
        rater_id="human_rater_alpha",
        rater_kind=RaterKind.HUMAN,
        engine_id=None,
        observation_level=ObservationLevel.HOLISTIC,
        criterion_id=None,
        score_category=2,
    )

    assert observation.observation_level is ObservationLevel.HOLISTIC
    assert observation.criterion_id is None
    assert observation.engine_id is None


def test_evidence_order_and_metadata_order_do_not_change_identity() -> None:
    """Input ordering cannot change the canonical observation identity."""
    first_span = EvidenceSpan(
        source_id="essay_response_alpha",
        start_offset=20,
        end_offset=24,
        evidence_label="secondary_span",
    )
    second_span = EvidenceSpan(
        source_id="essay_response_alpha",
        start_offset=2,
        end_offset=8,
        evidence_label="primary_span",
    )
    first = _observation(
        evidence_spans=(first_span, second_span),
        metadata={"second_value": 2, "first_value": -0.0},
    )
    second = _observation(
        evidence_spans=(second_span, first_span),
        metadata={"first_value": 0.0, "second_value": 2},
    )

    assert first == second
    assert first.evidence_spans == (second_span, first_span)
    assert first.observation_fingerprint == second.observation_fingerprint
    assert canonical_json(first) == canonical_json(second)
    assert '"first_value":0.0' in canonical_json(first)


def test_validate_observations_preserves_order_and_context() -> None:
    """Batch validation preserves caller order while verifying every binding."""
    first = _observation(observation_id="first_score_observation")
    second = _observation(
        observation_id="second_score_observation",
        status=ObservationStatus.ABSTAINED,
        score_category=None,
        reason_code="insufficient_evidence",
    )

    result = validate_observations(
        (second, first),
        assessment=approved_assessment(),
        rubrics=approved_rubrics(),
    )

    assert result == (second, first)


def test_validate_observations_rejects_duplicate_ids_and_fingerprints() -> None:
    """Batch identity collisions cannot enter downstream calibration data."""
    observation = _observation()
    with pytest.raises(AssessmentSpecError) as duplicate_fingerprint:
        validate_observations(
            (observation, observation),
            assessment=approved_assessment(),
            rubrics=approved_rubrics(),
        )
    assert duplicate_fingerprint.value.code == "duplicate_observation_id"

    same_id_different_content = _observation(confidence=0.5)
    with pytest.raises(AssessmentSpecError) as duplicate_id:
        validate_observations(
            (observation, same_id_different_content),
            assessment=approved_assessment(),
            rubrics=approved_rubrics(),
        )
    assert duplicate_id.value.code == "duplicate_observation_id"


def test_direct_score_observation_construction_is_rejected() -> None:
    """Public callers cannot forge content identities through direct construction."""
    valid = _observation()
    values = {
        field: getattr(valid, field)
        for field in (
            "observation_id",
            "assessment_fingerprint",
            "rubric_fingerprint",
            "response_id",
            "rater_id",
            "rater_kind",
            "engine_id",
            "construct_id",
            "observation_level",
            "criterion_id",
            "status",
            "score_category",
            "reason_code",
            "confidence",
            "evidence_spans",
            "occasion_id",
            "scorer_family",
            "scorer_version",
            "prompt_template_version",
            "metadata",
        )
    }

    with pytest.raises(AssessmentSpecError) as captured:
        ScoreObservation(**values)

    assert captured.value.code == "unverified_score_observation"
