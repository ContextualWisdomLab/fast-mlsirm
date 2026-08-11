"""LLM judge parsing is strict and transport stays injected."""

from __future__ import annotations

import json

from fast_mlsirm.llm_judge import (
    ContextualOrchestratorJudge,
    JudgeCriterion,
    JudgeFormatError,
)


class _FakeOrchestrator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    def complete(self, messages, mode="auto"):
        self.calls.append((messages, mode))
        return {
            "mode": "route",
            "answer": self.answer,
            "trace": [{"usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}],
        }


CRITERIA = [
    JudgeCriterion("task_alignment", "The answer directly addresses the task."),
    JudgeCriterion("factual_support", "The answer avoids unsupported claims."),
]


def _payload(score=0.8, accepted=True):
    return json.dumps({
        "score": score,
        "accepted": accepted,
        "rationale": "The answer is concise and supported.",
        "criterion_scores": {"task_alignment": score, "factual_support": score},
    })


def _category_payload():
    return json.dumps({
        "score": 0.75,
        "accepted": False,
        "rationale": "The evidence supports the ordered criterion levels.",
        "criterion_categories": {"task_alignment": 4.0, "factual_support": 2.0},
    })


def test_judge_uses_contextual_orchestrator_route_and_reports_usage() -> None:
    orchestrator = _FakeOrchestrator(_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="Explain the release plan.",
        answer="Use a staged release with rollback.",
        criteria=CRITERIA,
    )
    assert result.accepted is True
    assert result.score == 0.8
    assert result.trace_step_count == 1
    assert dict(result.usage) == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    assert orchestrator.calls[0][1] == "route"


def test_judge_rejects_malformed_decisions_and_derives_acceptance() -> None:
    try:
        ContextualOrchestratorJudge(_FakeOrchestrator("not json")).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )
    except JudgeFormatError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid judge response should fail closed")

    result = ContextualOrchestratorJudge(
        _FakeOrchestrator(_payload(score=0.8, accepted=False))
    ).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.accepted is True


def test_judge_result_projects_only_multiple_criteria_to_irt_items() -> None:
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.to_irt_row(item_type="dichotomous") == (1, 1)
    assert result.to_irt_row(item_type="polytomous", n_categories=5) == (4, 4)

    single_criterion = ContextualOrchestratorJudge(
        _FakeOrchestrator(
            json.dumps(
                {
                    "score": 0.8,
                    "accepted": True,
                    "rationale": "supported",
                    "criterion_scores": {"task_alignment": 0.8},
                }
            )
        )
    ).judge(
        task="task",
        answer="answer",
        criteria=[CRITERIA[0]],
    )
    try:
        single_criterion.to_irt_row()
    except JudgeFormatError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a scalar criterion must not become an IRT row")


def test_category_judgment_derives_ordered_scores_and_irt_items() -> None:
    orchestrator = _FakeOrchestrator(_category_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
    )
    assert result.category_count == 5
    assert dict(result.criterion_categories) == {
        "factual_support": 2,
        "task_alignment": 4,
    }
    assert result.score == 0.75
    assert result.accepted is True
    assert result.to_irt_row() == (2, 4)
    prompt = orchestrator.calls[0][0][0]["content"]
    assert '"task_alignment"' in prompt
    assert "numeric keys" in prompt
    assert "whole-number values from [0, 1, 2, 3, 4]" in prompt
    assert "category 4 means fully satisfies" in prompt


def test_category_judgment_rejects_non_integral_categories() -> None:
    payload = json.dumps({
        "score": 0.5,
        "accepted": True,
        "rationale": "mixed evidence",
        "criterion_categories": {"task_alignment": 1.5, "factual_support": 1},
    })
    try:
        ContextualOrchestratorJudge(_FakeOrchestrator(payload)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=3,
        )
    except JudgeFormatError as exc:
        assert "integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-integral categories must fail closed")


if __name__ == "__main__":
    test_judge_uses_contextual_orchestrator_route_and_reports_usage()
    test_judge_rejects_malformed_decisions_and_derives_acceptance()
    test_judge_result_projects_only_multiple_criteria_to_irt_items()
    test_category_judgment_derives_ordered_scores_and_irt_items()
    test_category_judgment_rejects_non_integral_categories()
    print("ok")
