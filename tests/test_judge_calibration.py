"""Paired polytomous judge-control contracts."""

from __future__ import annotations

import pytest
from fast_mlsirm import (
    JudgeCalibrationCase,
    JudgeCriterion,
    JudgeFormatError,
    LLMJudgeResult,
    build_multiple_choice_calibration_cases,
    evaluate_paired_calibration,
)

CRITERIA = (
    JudgeCriterion("evidence_quality", "The answer is supported by concrete evidence."),
    JudgeCriterion("risk_awareness", "The answer addresses material failure risks."),
)


def _cases() -> tuple[JudgeCalibrationCase, ...]:
    return build_multiple_choice_calibration_cases(
        case_id="release_case",
        question="Which rollout plan is safest?",
        options=("canary", "big_bang", "rollback_only"),
        answer="canary",
        correct_option_index=0,
        replacement_distractor="unreviewed_rollout",
        reference_answer="canary",
        contamination_status="held_out",
        shuffle_seed=7,
        gold_categories={"evidence_quality": 2, "risk_awareness": 1},
    )


def test_multiple_choice_controls_change_only_declared_presentation_factor() -> None:
    cases = _cases()
    by_variant = {case.variant: case for case in cases}

    assert set(by_variant) == {
        "baseline",
        "option_only",
        "shuffled_options",
        "replaced_distractor",
    }
    assert "Question:" in by_variant["baseline"].task
    assert "Question:" not in by_variant["option_only"].task
    assert by_variant["baseline"].metadata["question_present"] == "true"
    assert by_variant["option_only"].metadata["question_present"] == "false"
    assert by_variant["shuffled_options"].task != by_variant["baseline"].task
    assert "unreviewed_rollout" in by_variant["replaced_distractor"].task
    assert all(case.contamination_status == "held_out" for case in cases)


class _ScriptedJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, **kwargs) -> LLMJudgeResult:
        task = kwargs["task"]
        self.calls.append(task)
        if "unreviewed_rollout" in task:
            raise JudgeFormatError(
                "provider parse failure",
                evidence={
                    "call_count": 1,
                    "answer": "must not be retained",
                    "records": [
                        {"meets_threshold": False, "output_preview": "secret"}
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                },
            )
        option_only = "Question:" not in task
        categories = (
            {"evidence_quality": 1, "risk_awareness": 1}
            if option_only
            else {"evidence_quality": 2, "risk_awareness": 1}
        )
        scores = {key: value / 2 for key, value in categories.items()}
        return LLMJudgeResult(
            score=sum(scores.values()) / len(scores),
            accepted=True,
            rationale="scripted",
            criterion_scores=scores,
            raw_output="not retained in calibration report",
            orchestration_mode="route",
            trace_step_count=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            criterion_categories=categories,
            category_count=3,
            category_method="binary_threshold",
        )


def test_paired_calibration_preserves_failures_and_reports_gold_and_deltas() -> None:
    judge = _ScriptedJudge()
    report = evaluate_paired_calibration(
        judge,
        _cases(),
        criteria=CRITERIA,
        category_count=3,
    )

    assert report.status_counts() == {"judge_failed": 1, "passed": 3}
    assert len(judge.calls) == 4
    assert all(len(outcome.irt_row or ()) == 2 for outcome in report.outcomes if outcome.status == "passed")
    serialized = report.to_dict()
    assert serialized["case_count"] == 1
    assert serialized["outcome_count"] == 4
    assert serialized["gold_scored_count"] == 3
    assert serialized["gold_exact_agreement"] == pytest.approx(2 / 3)
    assert all("raw_output" not in outcome for outcome in serialized["outcomes"])
    failure = next(outcome for outcome in serialized["outcomes"] if outcome["status"] == "judge_failed")
    assert failure["evidence"] == {
        "call_count": 1,
        "records": [{"meets_threshold": False}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }

    effects = {effect["variant"]: effect for effect in report.paired_effects()}
    assert effects["option_only"]["score_delta"] == pytest.approx(-0.25)
    assert effects["shuffled_options"]["score_delta"] == pytest.approx(0.0)
    assert effects["replaced_distractor"]["control_status"] == "judge_failed"
    assert "score_delta" not in effects["replaced_distractor"]


def test_calibration_requires_multiple_items_and_a_baseline_pair() -> None:
    case = JudgeCalibrationCase(
        case_id="single",
        variant="baseline",
        task="task",
        answer="answer",
    )
    with pytest.raises(ValueError, match="baseline and at least one control"):
        evaluate_paired_calibration(
            _ScriptedJudge(),
            [case],
            criteria=CRITERIA,
            category_count=3,
        )

    control = JudgeCalibrationCase(
        case_id="single",
        variant="option_only",
        task="options",
        answer="answer",
    )
    with pytest.raises(ValueError, match="multiple criterion items"):
        evaluate_paired_calibration(
            _ScriptedJudge(),
            [case, control],
            criteria=[CRITERIA[0]],
            category_count=3,
        )


def test_calibration_without_gold_does_not_claim_agreement() -> None:
    cases = tuple(
        JudgeCalibrationCase(
            case_id="no_gold",
            variant=variant,
            task="Question: q\n\nOptions:\n1. a\n2. b" if variant == "baseline" else "Options:\n1. a\n2. b",
            answer="a",
        )
        for variant in ("baseline", "option_only")
    )
    report = evaluate_paired_calibration(
        _ScriptedJudge(), cases, criteria=CRITERIA, category_count=3
    )
    assert report.to_dict()["gold_scored_count"] == 0
    assert report.to_dict()["gold_exact_agreement"] is None


def test_calibration_rejects_ambiguous_normalized_inputs() -> None:
    with pytest.raises(TypeError, match="iterable of distinct strings"):
        build_multiple_choice_calibration_cases(
            case_id="bad_options",
            question="Which option?",
            options="a",
            answer="a",
            correct_option_index=0,
            replacement_distractor="b",
        )

    with pytest.raises(ValueError, match="unique after trimming"):
        JudgeCalibrationCase(
            case_id="bad_metadata",
            variant="baseline",
            task="task",
            answer="answer",
            metadata={"key": "one", " key ": "two"},
        )

    with pytest.raises(ValueError, match="unique after trimming"):
        JudgeCalibrationCase(
            case_id="bad_gold",
            variant="baseline",
            task="task",
            answer="answer",
            gold_categories={"a": 0, " a ": 1},
        )
