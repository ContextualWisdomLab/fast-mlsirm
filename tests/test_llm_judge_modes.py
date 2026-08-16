"""Preserve explicit contextual-orchestrator mode overrides for judge ablations."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.llm_judge import ContextualOrchestratorJudge, JudgeCriterion


class _RecordingOrchestrator:
    """Return one valid decision while recording the requested orchestration mode."""

    def __init__(self) -> None:
        self.modes: list[str] = []

    def complete(self, messages: list[dict[str, str]], *, mode: str) -> dict[str, object]:
        """Record ``mode`` and return a bounded Rust-independent judge fixture."""
        assert messages
        self.modes.append(mode)
        return {
            "mode": mode,
            "answer": json.dumps(
                {
                    "score": 1.0,
                    "accepted": True,
                    "rationale": "The answer satisfies the criterion.",
                    "criterion_scores": {"task_alignment": 1.0},
                }
            ),
            "trace": [],
        }


@pytest.mark.parametrize("mode", ["route", "conduct"])
def test_explicit_judge_mode_reaches_contextual_orchestrator(mode: str) -> None:
    """Explicit route/conduct overrides remain observable at the injected boundary."""
    orchestrator = _RecordingOrchestrator()

    result = ContextualOrchestratorJudge(orchestrator, mode=mode).judge(
        task="Assess the release plan.",
        answer="Use a staged release with rollback.",
        criteria=[JudgeCriterion("task_alignment", "The answer addresses the task.")],
    )

    assert orchestrator.modes == [mode]
    assert result.orchestration_mode == mode
