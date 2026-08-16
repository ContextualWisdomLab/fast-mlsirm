#!/usr/bin/env python3
"""Stage the regression that makes contextual-orchestrator auto the Judge default."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests" / "test_llm_judge_adaptive_default.py"
CONTENT = '''"""Default LLM Judge traffic is delegated to contextual-orchestrator auto policy."""

from __future__ import annotations

import json

from fast_mlsirm.llm_judge import ContextualOrchestratorJudge, JudgeCriterion


class _ModeRecordingOrchestrator:
    """Return one strict decision while recording the requested orchestration mode."""

    def __init__(self) -> None:
        self.modes: list[str] = []

    def complete(self, messages, mode="route"):
        del messages
        self.modes.append(mode)
        return {
            "mode": mode,
            "answer": json.dumps(
                {
                    "score": 1.0,
                    "accepted": True,
                    "rationale": "Both observable criteria are satisfied.",
                    "criterion_scores": {
                        "factual_support": 1.0,
                        "task_alignment": 1.0,
                    },
                }
            ),
            "trace": [],
        }


CRITERIA = (
    JudgeCriterion("task_alignment", "The answer addresses the requested task."),
    JudgeCriterion("factual_support", "The answer is supported by available evidence."),
)


def _judge(orchestrator, **kwargs):
    return ContextualOrchestratorJudge(orchestrator, **kwargs).judge(
        task="Evaluate the answer.",
        answer="The answer is evidence-backed.",
        criteria=CRITERIA,
    )


def test_judge_defaults_to_adaptive_contextual_orchestration() -> None:
    orchestrator = _ModeRecordingOrchestrator()
    result = _judge(orchestrator)

    assert orchestrator.modes == ["auto"]
    assert result.orchestration_mode == "auto"


def test_explicit_route_override_remains_available_for_controlled_ablation() -> None:
    orchestrator = _ModeRecordingOrchestrator()
    result = _judge(orchestrator, mode="route")

    assert orchestrator.modes == ["route"]
    assert result.orchestration_mode == "route"
'''

if TEST_PATH.exists():
    if TEST_PATH.read_text(encoding="utf-8") != CONTENT:
        raise SystemExit(f"refusing to replace a different existing test: {TEST_PATH}")
else:
    TEST_PATH.write_text(CONTENT, encoding="utf-8")
