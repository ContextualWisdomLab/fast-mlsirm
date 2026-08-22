"""Weighted judge evidence is authoritative and finite before transport."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge,
    JudgeCriterion,
)


class _StaticOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def complete(self, messages, mode="auto"):
        del messages, mode
        self.calls += 1
        return {"mode": "route", "answer": self.answer, "trace": []}


class _TransportSentinel:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, mode="auto"):
        del messages, mode
        self.calls += 1
        raise AssertionError("transport must not run for invalid aggregate criterion weight")


def test_plain_judge_derives_score_and_decision_from_weighted_criterion_evidence() -> None:
    response = json.dumps(
        {
            "score": 0.9,
            "accepted": True,
            "rationale": "Criterion evidence is mixed.",
            "criterion_scores": {
                "critical_evidence": 0.2,
                "supporting_evidence": 0.9,
            },
        }
    )
    orchestrator = _StaticOrchestrator(response)
    criteria = [
        JudgeCriterion("critical_evidence", "Critical evidence is present.", weight=3.0),
        JudgeCriterion("supporting_evidence", "Supporting evidence is present.", weight=1.0),
    ]

    result = ContextualOrchestratorJudge(orchestrator, accept_threshold=0.5).judge(
        task="Assess the evidence.",
        answer="Candidate answer.",
        criteria=criteria,
    )

    assert orchestrator.calls == 1
    assert result.score == pytest.approx(0.375)
    assert result.accepted is False
    assert dict(result.criterion_scores) == {
        "critical_evidence": 0.2,
        "supporting_evidence": 0.9,
    }


def test_nonfinite_aggregate_criterion_weight_fails_before_transport() -> None:
    orchestrator = _TransportSentinel()
    criteria = [
        JudgeCriterion("critical_evidence", "Critical evidence is present.", weight=1e308),
        JudgeCriterion("supporting_evidence", "Supporting evidence is present.", weight=1e308),
    ]

    with pytest.raises(ValueError, match="aggregate criterion weight must be finite"):
        ContextualOrchestratorJudge(orchestrator).judge(
            task="Assess the evidence.",
            answer="Candidate answer.",
            criteria=criteria,
        )

    assert orchestrator.calls == 0
