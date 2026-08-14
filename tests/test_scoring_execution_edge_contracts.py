"""Boundary tests for the governed scoring execution contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import pytest

import fast_mlsirm.scoring.execution as execution_module
from fast_mlsirm.scoring import AssessmentResponseType

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
holistic_request = _FIXTURES["holistic_request"]
rubric = _FIXTURES["rubric"]
fixture_engine = _FIXTURES["fixture_engine"]


def _assert_code(code: str, callback) -> None:
    """Assert one structured execution boundary error without exposing callbacks."""
    with pytest.raises(execution_module.AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code


class _ExplodingIndex:
    """Index protocol object used to exercise hostile numeric callbacks."""

    def __index__(self) -> int:
        """Raise without exposing the callback payload through the contract."""
        raise RuntimeError("private index callback payload")


def test_private_numeric_and_granularity_boundaries_fail_closed() -> None:
    """Numeric callbacks, booleans, schema versions, and granularity are strict."""
    _assert_code(
        "invalid_schema_version",
        lambda: execution_module._scoring_request_schema_version("1.0"),
    )
    _assert_code(
        "invalid_count",
        lambda: execution_module._nonnegative_integer(True, "count", 10),
    )
    _assert_code(
        "invalid_count",
        lambda: execution_module._nonnegative_integer(_ExplodingIndex(), "count", 10),
    )
    _assert_code(
        "invalid_score_category",
        lambda: execution_module._score_integer(True),
    )
    _assert_code(
        "invalid_score_category",
        lambda: execution_module._score_integer(_ExplodingIndex()),
    )
    _assert_code(
        "invalid_evidence_reference",
        lambda: execution_module._evidence_values((object(),)),
    )
    assert execution_module._request_granularity_allowed(
        AssessmentResponseType.CRITERION_LEVEL,
        execution_module.ObservationGranularity.HOLISTIC,
    ) is False


def test_scoring_request_constructor_rejects_wire_and_shape_tampering() -> None:
    """The sealed request rejects unsupported formats, scores, and axes."""
    valid = criterion_request()
    token = execution_module._SCORING_REQUEST_TOKEN
    _assert_code(
        "invalid_response_format",
        lambda: replace(valid, response_format="unsupported_format", _request_token=token),
    )
    _assert_code(
        "invalid_allowed_scores",
        lambda: replace(valid, allowed_scores=(2, 1), _request_token=token),
    )
    _assert_code(
        "missing_criterion_ids",
        lambda: replace(valid, criterion_ids=(), _request_token=token),
    )
    holistic = holistic_request()
    _assert_code(
        "unexpected_criterion_ids",
        lambda: replace(
            holistic,
            criterion_ids=("claim_support",),
            _request_token=token,
        ),
    )
    assert valid.to_dict()["request_fingerprint"] == valid.request_fingerprint


def _request_builder_values(**overrides: Any) -> dict[str, Any]:
    """Return valid builder arguments with one focused override."""
    values: dict[str, Any] = {
        "request_id": "edge_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "granularity": execution_module.ObservationGranularity.CRITERION_LEVEL,
        "respondent_id": "edge_respondent",
        "response_id": "edge_response",
        "task_id": "edge_task",
        "task_revision_fingerprint": "d" * 64,
        "task_family_id": "evidence_review",
        "occasion_id": "edge_occasion",
        "criterion_ids": ("claim_support",),
        "response_content_fingerprint": "c" * 64,
        "response_character_count": 12,
        "response_unit_count": 3,
        "metadata": {},
    }
    values.update(overrides)
    return values


def test_public_builders_reject_untyped_and_undeclared_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request, observation, and result builders validate every object boundary."""
    _assert_code(
        "invalid_assessment_spec",
        lambda: execution_module.build_scoring_request(
            **_request_builder_values(assessment=object())
        ),
    )
    _assert_code(
        "invalid_rubric",
        lambda: execution_module.build_scoring_request(
            **_request_builder_values(rubric=object())
        ),
    )
    selected_assessment = assessment()
    monkeypatch.setattr(
        type(selected_assessment),
        "construct_ids",
        property(lambda _assessment: ()),
    )
    _assert_code(
        "unknown_rubric_construct",
        lambda: execution_module.build_scoring_request(
            **_request_builder_values(assessment=selected_assessment)
        ),
    )
    monkeypatch.undo()

    request = criterion_request()
    engine = automated_engine()
    _assert_code(
        "invalid_scoring_request",
        lambda: execution_module.build_score_observation(
            observation_id="edge_observation",
            request=object(),
            engine=engine,
            criterion_id="claim_support",
            status=execution_module.ObservationStatus.SCORED,
            score_category=1,
        ),
    )
    _assert_code(
        "invalid_engine_descriptor",
        lambda: execution_module.build_score_observation(
            observation_id="edge_observation",
            request=request,
            engine=object(),
            criterion_id="claim_support",
            status=execution_module.ObservationStatus.SCORED,
            score_category=1,
        ),
    )
    _assert_code(
        "invalid_scoring_request",
        lambda: execution_module.build_scoring_result(
            result_id="edge_result",
            request=object(),
            engine=engine,
            observations=(),
        ),
    )
    _assert_code(
        "invalid_engine_descriptor",
        lambda: execution_module.build_scoring_result(
            result_id="edge_result",
            request=request,
            engine=object(),
            observations=(),
        ),
    )
    _assert_code(
        "invalid_score_observation",
        lambda: execution_module.build_scoring_result(
            result_id="edge_result",
            request=request,
            engine=engine,
            observations=(object(),),
        ),
    )


def test_result_and_fixture_boundaries_cover_holistic_and_terminal_paths() -> None:
    """Results and fixture engines preserve exact holistic and terminal semantics."""
    request = holistic_request()
    engine = automated_engine()
    first = execution_module.build_score_observation(
        observation_id="holistic_first",
        request=request,
        engine=engine,
        criterion_id=None,
        status=execution_module.ObservationStatus.SCORED,
        score_category=1,
    )
    object.__setattr__(first, "criterion_id", "claim_support")
    _assert_code(
        "incomplete_observation_coverage",
        lambda: execution_module.build_scoring_result(
            result_id="holistic_edge_result",
            request=request,
            engine=engine,
            observations=(first,),
        ),
    )

    _assert_code(
        "missing_fixture_score",
        lambda: execution_module.FixtureOutcome(
            criterion_id="claim_support",
            status=execution_module.ObservationStatus.SCORED,
        ),
    )
    _assert_code(
        "unexpected_fixture_reason",
        lambda: execution_module.FixtureOutcome(
            criterion_id="claim_support",
            status=execution_module.ObservationStatus.SCORED,
            score_category=1,
            reason_code="not_terminal",
        ),
    )
    terminal = execution_module.FixtureOutcome(
        criterion_id="claim_support",
        status=execution_module.ObservationStatus.ABSTAINED,
        reason_code="insufficient_evidence",
    )
    assert terminal.reason_code == "insufficient_evidence"

    _assert_code(
        "invalid_engine_descriptor",
        lambda: execution_module.StaticFixtureEngine(
            descriptor=object(),
            outcomes=(terminal,),
        ),
    )
    _assert_code(
        "invalid_fixture_outcome",
        lambda: execution_module.StaticFixtureEngine(
            descriptor=engine,
            outcomes=(object(),),
        ),
    )
    duplicate = execution_module.FixtureOutcome(
        criterion_id="claim_support",
        status=execution_module.ObservationStatus.SCORED,
        score_category=1,
    )
    _assert_code(
        "duplicate_fixture_criterion",
        lambda: execution_module.StaticFixtureEngine(
            descriptor=engine,
            outcomes=(duplicate, duplicate),
        ),
    )
    _assert_code(
        "invalid_scoring_request",
        lambda: execution_module.StaticFixtureEngine(
            descriptor=engine,
            outcomes=(duplicate,),
        ).score(object()),
    )

    holistic_engine = execution_module.StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            execution_module.FixtureOutcome(
                criterion_id=None,
                status=execution_module.ObservationStatus.SCORED,
                score_category=1,
            ),
        ),
    )
    assert holistic_engine.score(request).observations[0].criterion_id is None
    _assert_code(
        "fixture_granularity_mismatch",
        lambda: holistic_engine.score(criterion_request()),
    )
    result = fixture_engine().score(criterion_request())
    assert result.to_dict()["result_fingerprint"] == result.result_fingerprint
