"""LLM judge parsing is strict and transport stays injected."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
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


class _CompletionOrchestrator:
    def __init__(self, completion):
        self.completion = completion

    def complete(self, messages, mode="auto"):
        return self.completion


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
    prompt = orchestrator.calls[0][0][1]["content"]
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["task"] == "Explain the release plan."
    assert payload["answer"] == "Use a staged release with rollback."


def test_judge_rejects_malformed_decisions_and_derives_acceptance() -> None:
    with pytest.raises(JudgeFormatError):
        ContextualOrchestratorJudge(_FakeOrchestrator("not json")).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )
    result = ContextualOrchestratorJudge(
        _FakeOrchestrator(_payload(score=0.8, accepted=False))
    ).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.accepted is True


def test_judge_rejects_wrapped_or_fenced_json() -> None:
    for answer in (
        f"prefix {_payload()}",
        f"{_payload()} suffix",
        f"```json\n{_payload()}\n```",
    ):
        with pytest.raises(JudgeFormatError):
            ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )


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
    with pytest.raises(JudgeFormatError):
        single_criterion.to_irt_row()


def test_irt_projection_rejects_malformed_result_mappings() -> None:
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    with pytest.raises(JudgeFormatError, match="keys must be strings"):
        replace(result, criterion_scores={1: 0.8, "factual_support": 0.8}).to_irt_row()
    with pytest.raises(JudgeFormatError, match="criterion_categories must be an object"):
        replace(
            result,
            criterion_categories=[0, 1],
            category_count=2,
        ).to_irt_row()


def test_criteria_limit_is_enforced_during_iteration() -> None:
    yielded = 0

    def criteria():
        nonlocal yielded
        for index in range(33):
            yielded += 1
            yield JudgeCriterion(f"criterion_{index}", "observable evidence")

    with pytest.raises(ValueError, match="1..32"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=criteria(),
        )
    assert yielded == 33


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
    with pytest.raises(JudgeFormatError, match="integer"):
        ContextualOrchestratorJudge(_FakeOrchestrator(payload)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=3,
        )


def test_judge_rejects_missing_or_malformed_model_fields() -> None:
    cases = [
        {},
        {"answer": _payload().replace("rationale", "explanation")},
        "not a completion mapping",
    ]
    for completion in cases:
        with pytest.raises(JudgeFormatError):
            ContextualOrchestratorJudge(_CompletionOrchestrator(completion)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )


def test_judge_criteria_reject_invalid_runtime_types() -> None:
    class _HookedFloat(float):
        invoked = False

        def __float__(self):
            type(self).invoked = True
            return super().__float__()

    with pytest.raises(ValueError, match="criterion_id must be a string"):
        JudgeCriterion(1, "description")
    with pytest.raises(ValueError, match="criterion description must be a string"):
        JudgeCriterion("task_alignment", 1)
    with pytest.raises(ValueError, match="criterion weight must be a number"):
        JudgeCriterion("task_alignment", "description", "1")
    with pytest.raises(ValueError, match="criterion weight must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[
                {
                    "criterion_id": "task_alignment",
                    "description": "ok",
                    "weight": "1",
                }
            ],
        )
    for criterion in (
        {"criterion_id": 1, "description": "ok"},
        {"criterion_id": "task_alignment", "description": 1},
        {
            "criterion_id": "task_alignment",
            "description": "ok",
            "weight": _HookedFloat(1.0),
        },
    ):
        with pytest.raises(ValueError):
            ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
                task="task",
                answer="answer",
                criteria=[criterion],
            )
    assert _HookedFloat.invoked is False


def test_judge_criteria_reject_non_contract_values_with_value_error() -> None:
    """Arbitrary criterion elements must fail through the stable benign error contract."""
    with pytest.raises(ValueError, match="JudgeCriterion or mapping"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[object()],
        )


if __name__ == "__main__":
    test_judge_uses_contextual_orchestrator_route_and_reports_usage()
    test_judge_rejects_malformed_decisions_and_derives_acceptance()
    test_judge_result_projects_only_multiple_criteria_to_irt_items()
    test_irt_projection_rejects_malformed_result_mappings()
    test_criteria_limit_is_enforced_during_iteration()
    test_category_judgment_derives_ordered_scores_and_irt_items()
    test_category_judgment_rejects_non_integral_categories()
    test_judge_rejects_missing_or_malformed_model_fields()
    test_judge_criteria_reject_invalid_runtime_types()
    test_judge_criteria_reject_non_contract_values_with_value_error()
    print("ok")
