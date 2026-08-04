"""Replay-hardening tests for scoring-to-facets projection provenance."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    ObservationGranularity,
    build_scoring_facets_rating_records,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
automated_engine = _BASE["automated_engine"]
execution = _BASE["execution"]


def replay_execution():
    """Return one complete criterion-level execution for replay mutation tests."""
    return execution(
        request_id="projection_replay_request",
        response_id="projection_replay_response",
        respondent_id="projection_replay_respondent",
        task_id="projection_replay_task",
        engine=automated_engine(),
        claim_score=1,
        source_score=2,
    )


@pytest.mark.parametrize(
    ("field_name", "mutated_value", "expected_code"),
    (
        (
            "request_fingerprint",
            "a" * 64,
            "calibration_observation_request_mismatch",
        ),
        (
            "engine_fingerprint",
            "b" * 64,
            "calibration_observation_engine_mismatch",
        ),
        (
            "assessment_fingerprint",
            "c" * 64,
            "calibration_observation_assessment_mismatch",
        ),
        (
            "rubric_fingerprint",
            "d" * 64,
            "calibration_observation_rubric_mismatch",
        ),
        (
            "construct_id",
            "mutated_construct",
            "calibration_observation_construct_mismatch",
        ),
        (
            "granularity",
            ObservationGranularity.HOLISTIC,
            "calibration_observation_granularity_mismatch",
        ),
        (
            "criterion_id",
            "undeclared_criterion",
            "calibration_observation_criterion_mismatch",
        ),
    ),
)
def test_projection_rejects_mutated_observation_provenance(
    field_name: str,
    mutated_value,
    expected_code: str,
) -> None:
    """A result cannot silently discard observation-level provenance mutations."""
    request, result, engine = replay_execution()
    observation = result.observations[0]
    object.__setattr__(observation, field_name, mutated_value)

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == expected_code
    assert caught.value.path.endswith(f".{field_name}")
    assert str(mutated_value) not in str(caught.value)


def test_projection_rejects_mutated_result_criterion_scope() -> None:
    """Result-declared criterion scope must replay the supplied request exactly."""
    request, result, engine = replay_execution()
    object.__setattr__(result, "requested_criterion_ids", ("claim_support",))

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "calibration_result_criteria_mismatch"
    assert caught.value.path.endswith(".requested_criterion_ids")


def test_projection_rejects_mutated_result_observation_coverage() -> None:
    """A result cannot drop one requested criterion after factory validation."""
    request, result, engine = replay_execution()
    object.__setattr__(result, "observations", result.observations[:1])

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "calibration_observation_coverage_mismatch"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_duplicate_observation_criterion() -> None:
    """Duplicate criterion coverage remains distinct from simple omission."""
    request, result, engine = replay_execution()
    object.__setattr__(
        result,
        "observations",
        (*result.observations, result.observations[1]),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "duplicate_observation_criterion"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_duplicate_observation_identity() -> None:
    """Distinct criteria cannot reuse one observation identity after mutation."""
    request, result, engine = replay_execution()
    object.__setattr__(
        result.observations[1],
        "observation_id",
        result.observations[0].observation_id,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "duplicate_observation_id"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_untyped_observation_entry() -> None:
    """A privately replaced child cannot escape as an AttributeError."""
    request, result, engine = replay_execution()
    private_value = object()
    object.__setattr__(
        result,
        "observations",
        (result.observations[0], private_value),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_score_observation"
    assert caught.value.path.endswith(".observations[1]")
    assert str(private_value) not in str(caught.value)
