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
    request, result, engine = execution(
        request_id="projection_replay_request",
        response_id="projection_replay_response",
        respondent_id="projection_replay_respondent",
        task_id="projection_replay_task",
        engine=automated_engine(),
        claim_score=1,
        source_score=2,
    )
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
