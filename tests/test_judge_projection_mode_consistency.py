"""Measurement-semantic regressions for judge-panel projection."""

from __future__ import annotations

import pytest

from fast_mlsirm.judge_construct import (
    JudgeFormatError,
    LLMJudgeResult,
    project_judge_results_to_matrix,
    validate_judge_construct,
)


CRITERIA = tuple(f"criterion_{index}" for index in range(5))


def _result(*, explicit_categories: bool, category: int = 2) -> LLMJudgeResult:
    scores = {criterion_id: category / 3 for criterion_id in CRITERIA}
    categories = (
        {criterion_id: category for criterion_id in CRITERIA}
        if explicit_categories
        else None
    )
    return LLMJudgeResult(
        score=category / 3,
        accepted=True,
        rationale="projection mode regression",
        criterion_scores=scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=4 if explicit_categories else None,
    )


def test_projection_rejects_mixed_explicit_and_score_binned_rows() -> None:
    spec = validate_judge_construct(CRITERIA, n_categories=4)

    with pytest.raises(
        JudgeFormatError,
        match="must not mix explicit categories and score-binned rows",
    ):
        project_judge_results_to_matrix(
            [
                _result(explicit_categories=True, category=1),
                _result(explicit_categories=False, category=2),
            ],
            spec,
        )


def test_projection_preserves_all_explicit_category_rows() -> None:
    spec = validate_judge_construct(CRITERIA, n_categories=4)
    matrix = project_judge_results_to_matrix(
        [
            _result(explicit_categories=True, category=1),
            _result(explicit_categories=True, category=3),
        ],
        spec,
    )
    assert matrix.tolist() == [[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]]


def test_projection_preserves_all_score_binned_rows() -> None:
    spec = validate_judge_construct(CRITERIA, n_categories=4)
    matrix = project_judge_results_to_matrix(
        [
            _result(explicit_categories=False, category=1),
            _result(explicit_categories=False, category=2),
        ],
        spec,
    )
    assert matrix.tolist() == [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
