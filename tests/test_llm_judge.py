"""LLM judge parsing is strict and transport stays injected."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fast_mlsirm.irt_contract import validate_irt_response_matrix
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


def _threshold_payload(thresholds=None):
    return json.dumps({
        "score": 0.0,
        "accepted": True,
        "rationale": "The ordered evidence supports separate cumulative thresholds.",
        "criterion_thresholds": thresholds or {
            "task_alignment": [True, True, True, True],
            "factual_support": [True, False, False, False],
        },
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


@pytest.mark.parametrize("accepted", [0, 1, "true", None])
def test_judge_rejects_non_boolean_advisory_acceptance(accepted) -> None:
    with pytest.raises(JudgeFormatError, match="accepted must be a boolean"):
        ContextualOrchestratorJudge(
            _FakeOrchestrator(_payload(accepted=accepted))
        ).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )


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



def test_judge_rejects_duplicate_and_unknown_top_level_fields() -> None:
    duplicate = (
        '{"score":0.8,"accepted":true,"rationale":"supported",'
        '"criterion_scores":{"task_alignment":0.8,"factual_support":0.8},'
        '"score":0.2}'
    )
    unknown = json.loads(_payload())
    unknown["unexpected"] = "ignored fields are unsafe"
    for answer in (duplicate, json.dumps(unknown)):
        with pytest.raises(JudgeFormatError, match="exactly|duplicate"):
            ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )


def test_judge_rejects_duplicate_nested_criterion_fields() -> None:
    answer = (
        '{"score":0.8,"accepted":true,"rationale":"supported",'
        '"criterion_scores":{"task_alignment":0.8,"task_alignment":0.2,'
        '"factual_support":0.8}}'
    )
    with pytest.raises(JudgeFormatError, match="duplicate"):
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
    with pytest.raises(JudgeFormatError, match="item_type"):
        result.to_irt_row(item_type=[])


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
    assert "no markdown fences" in prompt
    assert "category values are JSON integers" in prompt
    assert "category 4" in prompt


def test_cumulative_threshold_judgment_derives_monotone_polytomous_items() -> None:
    orchestrator = _FakeOrchestrator(_threshold_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
        category_method="cumulative_threshold",
    )

    assert result.category_method == "cumulative_threshold"
    assert dict(result.criterion_categories) == {
        "factual_support": 1,
        "task_alignment": 4,
    }
    assert result.score == 0.625
    assert result.accepted is False
    assert result.to_dict()["category_method"] == "cumulative_threshold"
    row = result.to_irt_row()
    assert row == (1, 4)
    matrix = validate_irt_response_matrix([row], "polytomous", n_categories=5)
    assert matrix.shape == (1, 2)
    prompt = orchestrator.calls[0][0][0]["content"]
    assert "criterion_thresholds" in prompt
    assert "cumulative thresholds" in prompt
    assert "must be monotone" in prompt
    assert "K-way choice" in prompt


@pytest.mark.parametrize(
    ("thresholds", "match"),
    [
        (
            {"task_alignment": [True, False, True, False], "factual_support": [False] * 4},
            "monotone",
        ),
        (
            {"task_alignment": [True, 1, False, False], "factual_support": [False] * 4},
            "boolean",
        ),
        (
            {"task_alignment": [True, True], "factual_support": [False] * 4},
            "boolean array",
        ),
    ],
)
def test_cumulative_threshold_rejects_malformed_thresholds(thresholds, match) -> None:
    with pytest.raises(JudgeFormatError, match=match):
        ContextualOrchestratorJudge(
            _FakeOrchestrator(_threshold_payload(thresholds))
        ).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=5,
            category_method="cumulative_threshold",
        )


def test_cumulative_threshold_requires_explicit_category_count() -> None:
    with pytest.raises(ValueError, match="explicit category_count"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_method="cumulative_threshold",
        )


@pytest.mark.parametrize("category_method", ["unknown", [], {}])
def test_judge_rejects_unknown_category_method(category_method) -> None:
    with pytest.raises(ValueError, match="category_method"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_method=category_method,
        )


def test_judge_rejects_unhashable_mode_before_membership() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload()), mode=[])


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


def test_category_count_and_category_values_reject_runtime_subclasses() -> None:
    class _ForgedInt(int):
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    judge = ContextualOrchestratorJudge(_FakeOrchestrator(_category_payload()))
    for value in (True, 1.0, 65, 10**1000, _ForgedInt(10**1000)):
        with pytest.raises(ValueError, match="category_count must be an integer"):
            judge.judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
                category_count=value,
            )

    result = judge.judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
    )
    with pytest.raises(JudgeFormatError, match="criterion_categories"):
        replace(
            result,
            criterion_categories={
                "task_alignment": _ForgedInt(10**1000),
                "factual_support": 1,
            },
        ).to_irt_row()


def test_category_judgment_rejects_malformed_top_level_score() -> None:
    payload = json.dumps({
        "score": {"factual_support": 0.8},
        "accepted": True,
        "rationale": "mixed evidence",
        "criterion_categories": {"task_alignment": 1, "factual_support": 1},
    })
    with pytest.raises(JudgeFormatError, match="score must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(payload)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=2,
        )


def test_judge_rejects_overflowing_and_runtime_subclass_scores() -> None:
    class _HookedFloat(float):
        invoked = False

        def __float__(self):
            type(self).invoked = True
            return super().__float__()

    overflowing = json.dumps({
        "score": 10**1000,
        "accepted": True,
        "rationale": "unsupported numeric shape",
        "criterion_scores": {"task_alignment": 0.8, "factual_support": 0.8},
    })
    with pytest.raises(JudgeFormatError, match="score must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(overflowing)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )

    subclass_score = _HookedFloat(0.8)
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    with pytest.raises(JudgeFormatError, match="criterion_scores"):
        replace(
            result,
            criterion_scores={
                "task_alignment": subclass_score,
                "factual_support": 0.8,
            },
        ).to_irt_row(item_type="dichotomous")
    assert _HookedFloat.invoked is False


def test_judge_text_and_usage_boundaries_reject_runtime_subclasses() -> None:
    class _HookedString(str):
        invoked = False

        def strip(self, *args, **kwargs):
            type(self).invoked = True
            return super().strip(*args, **kwargs)

    class _ForgedInt(int):
        invoked = False

        def __ge__(self, other):
            type(self).invoked = True
            return True

    with pytest.raises(ValueError, match="task must be"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task=_HookedString("task"),
            answer="answer",
            criteria=CRITERIA,
        )
    assert _HookedString.invoked is False

    forged = _ForgedInt(7)
    result = _CompletionOrchestrator({
        "mode": "route",
        "answer": _payload(),
        "trace": [{
            "usage": {
                "prompt_tokens": forged,
                "completion_tokens": forged,
                "total_tokens": forged,
            }
        }],
    })
    judged = ContextualOrchestratorJudge(result).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert dict(judged.usage) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert _ForgedInt.invoked is False


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


def test_judge_rejects_unhashable_criterion_id_before_category_template() -> None:
    class _UnhashableStr(str):
        __hash__ = None

    with pytest.raises(ValueError, match="criterion_id must be a string"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[
                {
                    "criterion_id": _UnhashableStr("task_alignment"),
                    "description": "ok",
                },
                CRITERIA[1],
            ],
            category_count=3,
        )


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

def test_judge_rejects_excessive_json_nesting() -> None:
    """Deeply nested JSON cannot expand into recursive parser DoS."""
    # Nesting depth 33 exceeds MAX_JUDGE_JSON_DEPTH (32).
    nested = "{" + '"k":{' * 32 + '"score": 0.8' + "}" * 32 + "}"
    assert nested.count("{") == 33
    with pytest.raises(JudgeFormatError, match="nesting exceeds maximum depth"):
        ContextualOrchestratorJudge(_FakeOrchestrator(nested)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )


def test_judge_accepts_bounded_json_nesting() -> None:
    """Nesting at the admitted depth still parses when the payload is valid."""
    # Build a valid judge payload with modest nesting under the limit.
    inner = {
        "score": 0.8,
        "accepted": True,
        "rationale": "ok",
        "criterion_scores": {"task_alignment": 0.8, "factual_support": 0.8},
    }
    raw = json.dumps(inner)
    result = ContextualOrchestratorJudge(_FakeOrchestrator(raw)).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.score == 0.8
