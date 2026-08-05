"""Adversarial nested-contract tests for enterprise calibration replay."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    build_enterprise_issue_facets_rating_records,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_enterprise_issue_calibration.py"))
)
_execution = _FIXTURES["_execution"]


def test_mutated_result_observation_fails_with_structured_error() -> None:
    """An exact result with a counterfeit nested observation never leaks AttributeError."""
    issue, request, result, engine = _execution(
        issue_label="nested_contract",
        task_label="nested_contract",
        engine_label="nested_contract",
        scores=(1, 2),
    )
    object.__setattr__(result, "observations", (object(),))

    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )

    assert captured.value.code == "invalid_score_observation"
    assert captured.value.path == "$.result.observations[0]"
