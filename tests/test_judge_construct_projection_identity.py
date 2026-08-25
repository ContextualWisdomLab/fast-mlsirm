"""Regression tests for judge-construct identity and policy boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.judge_construct import (
    JudgeConstructPolicy,
    JudgeFormatError,
    MAX_JUDGE_CONSTRUCT_ITEMS,
    project_judge_results_to_matrix,
    validate_judge_construct,
)
from fast_mlsirm.llm_judge import LLMJudgeResult


def _result(categories: dict[str, int], *, category_count: int = 4) -> LLMJudgeResult:
    """Build one explicit-category judge result for projection tests."""
    scores = {
        criterion_id: category / (category_count - 1)
        for criterion_id, category in categories.items()
    }
    return LLMJudgeResult(
        score=sum(scores.values()) / len(scores),
        accepted=True,
        rationale="projection identity regression",
        criterion_scores=scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=category_count,
    )


def test_projection_rejects_same_count_different_criterion_set() -> None:
    """Equal row width must not substitute for exact item identity."""
    spec_ids = tuple(f"criterion_{index}" for index in range(5))
    spec = validate_judge_construct(spec_ids, n_categories=4)
    result = _result({f"other_{index}": index % 4 for index in range(5)})

    with pytest.raises(JudgeFormatError, match="criterion set does not match"):
        project_judge_results_to_matrix([result], spec)


def test_projection_columns_follow_spec_criterion_order() -> None:
    """Matrix column j must always represent spec.criterion_ids[j]."""
    spec_ids = (
        "criterion_c",
        "criterion_a",
        "criterion_e",
        "criterion_b",
        "criterion_d",
    )
    spec = validate_judge_construct(spec_ids, n_categories=4)
    result = _result(
        {
            "criterion_a": 0,
            "criterion_b": 1,
            "criterion_c": 3,
            "criterion_d": 1,
            "criterion_e": 2,
        }
    )

    matrix = project_judge_results_to_matrix([result], spec)

    assert matrix.dtype == np.int64
    assert matrix.tolist() == [[3, 0, 2, 1, 1]]


def test_custom_policy_cannot_raise_hard_facet_ceiling() -> None:
    """Custom quality policy may tighten, but never exceed, the package cap."""
    with pytest.raises(
        ValueError,
        match=rf"max_items cannot exceed {MAX_JUDGE_CONSTRUCT_ITEMS}",
    ):
        JudgeConstructPolicy(
            min_items=5,
            recommended_items=7,
            max_items=MAX_JUDGE_CONSTRUCT_ITEMS + 1,
        )
