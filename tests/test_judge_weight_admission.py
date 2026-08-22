"""Regression coverage for LLM-judge criterion-weight admission."""

from __future__ import annotations

import pytest

from fast_mlsirm.llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge,
    JudgeCriterion,
)


class _FailIfCalledOrchestrator:
    """Transport sentinel proving invalid weights fail before model execution."""

    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, mode="auto"):
        """Record unexpected transport execution and fail loudly."""
        del messages, mode
        self.calls += 1
        raise AssertionError("judge transport executed before weight admission")


def test_nonfinite_total_criterion_weight_fails_before_transport() -> None:
    """Finite members whose aggregate overflows must not corrupt judge scoring."""
    orchestrator = _FailIfCalledOrchestrator()
    judge = ContextualOrchestratorJudge(orchestrator)
    criteria = (
        JudgeCriterion("criterion_one", "First observable criterion.", weight=1e308),
        JudgeCriterion("criterion_two", "Second observable criterion.", weight=1e308),
    )

    with pytest.raises(ValueError, match="criteria total weight must be finite"):
        judge.judge(
            task="Evaluate the answer.",
            answer="Candidate answer.",
            criteria=criteria,
            category_count=3,
            category_method="direct",
        )

    assert orchestrator.calls == 0
