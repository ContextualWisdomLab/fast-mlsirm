"""Ordering-contract regressions for judge-result projection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fast_mlsirm.judge_construct import (
    LLMJudgeResult,
    project_judge_results_to_matrix,
    validate_judge_construct,
)


def _result(categories: dict[str, int]) -> LLMJudgeResult:
    """Build one explicit-category result for ordering tests."""
    return LLMJudgeResult(
        score=0.5,
        accepted=True,
        rationale="ordering contract fixture",
        criterion_scores={key: value / 3.0 for key, value in categories.items()},
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=4,
    )


def test_projector_passes_authoritative_spec_order_to_row_projection(monkeypatch) -> None:
    """Projection must not depend on two independent lexical-sort implementations."""
    criterion_ids = (
        "criterion_zeta",
        "criterion_alpha",
        "criterion_delta",
        "criterion_beta",
        "criterion_gamma",
    )
    spec = validate_judge_construct(criterion_ids, n_categories=4)
    result = _result(
        {
            "criterion_alpha": 1,
            "criterion_beta": 3,
            "criterion_delta": 2,
            "criterion_gamma": 0,
            "criterion_zeta": 3,
        }
    )
    original = LLMJudgeResult.to_irt_row
    observed_orders: list[Sequence[str] | None] = []

    def capture_order(
        self: LLMJudgeResult,
        *,
        item_type: str = "polytomous",
        n_categories: int | None = None,
        criterion_order: Sequence[str] | None = None,
    ) -> tuple[int, ...]:
        observed_orders.append(criterion_order)
        if criterion_order is None:
            return original(self, item_type=item_type, n_categories=n_categories)
        return original(
            self,
            item_type=item_type,
            n_categories=n_categories,
            criterion_order=criterion_order,
        )

    monkeypatch.setattr(LLMJudgeResult, "to_irt_row", capture_order)

    matrix = project_judge_results_to_matrix([result], spec)

    assert observed_orders == [spec.criterion_ids]
    assert np.array_equal(matrix, np.array([[3, 1, 2, 3, 0]], dtype=np.int64))
