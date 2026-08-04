"""Deterministic fixtures for scoring-observation and engine-contract tests."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.scoring import AssessmentResponseType

_SHARED = runpy.run_path(str(Path(__file__).with_name("scoring_contract_fixtures.py")))
assessment = _SHARED["assessment"]
rubric = _SHARED["rubric"]


def approved_rubrics():
    """Return the exact rubric registry bound by the mixed fixture assessment."""
    return (
        rubric("argument_rubric", "argument_quality"),
        rubric("evidence_rubric", "evidence_use"),
    )


def approved_assessment():
    """Return an assessment that permits criterion-level and holistic ratings."""
    return assessment(
        rubrics=approved_rubrics(),
        response_type=AssessmentResponseType.MIXED,
    )


def argument_rubric():
    """Return the argument-quality rubric from the approved registry."""
    return approved_rubrics()[0]
