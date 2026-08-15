#!/usr/bin/env python3
"""Apply and document the adaptive contextual-orchestrator default."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "agent/adaptive-orchestrator-default"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Replace one exact repository fragment or fail closed."""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    """Patch the judge default, tests, changelog, and ADR."""
    branch = os.environ.get("GITHUB_REF_NAME", EXPECTED_BRANCH)
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"refusing to mutate unexpected branch: {branch}")

    judge_path = ROOT / "python" / "fast_mlsirm" / "llm_judge.py"
    judge = judge_path.read_text(encoding="utf-8")
    judge = replace_once(
        judge,
        '    def __init__(self, orchestrator: Any, *, mode: str = "route", accept_threshold: float = 0.7) -> None:\n',
        '    def __init__(self, orchestrator: Any, *, mode: str = "auto", accept_threshold: float = 0.7) -> None:\n',
        "judge default",
    )
    judge = replace_once(
        judge,
        'class ContextualOrchestratorJudge:\n    """Evaluate one answer through an injected contextual-orchestrator."""\n',
        'class ContextualOrchestratorJudge:\n    """Evaluate one answer through adaptive contextual orchestration by default.\n\n    ``mode=\"auto\"`` delegates the quality-versus-cost decision to\n    contextual-orchestrator. Explicit ``route`` and ``conduct`` values remain\n    available for controlled ablation, diagnostics, and operator overrides.\n    """\n',
        "judge docstring",
    )
    judge_path.write_text(judge, encoding="utf-8")

    test_path = ROOT / "tests" / "test_llm_judge.py"
    tests = test_path.read_text(encoding="utf-8")
    tests = replace_once(
        tests,
        "def test_judge_uses_contextual_orchestrator_route_and_reports_usage() -> None:\n",
        "def test_judge_uses_contextual_orchestrator_auto_mode_and_reports_usage() -> None:\n",
        "test name",
    )
    tests = replace_once(
        tests,
        '    assert orchestrator.calls[0][1] == "route"\n',
        '    assert orchestrator.calls[0][1] == "auto"\n',
        "default assertion",
    )
    anchor = '''    assert payload["answer"] == "Use a staged release with rollback."\n\n\n'''
    addition = '''    assert payload["answer"] == "Use a staged release with rollback."\n\n\ndef test_judge_preserves_explicit_route_override_for_ablation() -> None:\n    orchestrator = _FakeOrchestrator(_payload())\n    ContextualOrchestratorJudge(orchestrator, mode="route").judge(\n        task="Explain the release plan.",\n        answer="Use a staged release with rollback.",\n        criteria=CRITERIA,\n    )\n    assert orchestrator.calls[0][1] == "route"\n\n\n'''
    tests = replace_once(tests, anchor, addition, "explicit override regression")
    test_path.write_text(tests, encoding="utf-8")

    changelog_path = ROOT / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    changelog = replace_once(
        changelog,
        "## Unreleased\n\n### Fixed\n",
        "## Unreleased\n\n### Changed\n\n"
        "- Default `ContextualOrchestratorJudge` calls to `mode=\"auto\"`, allowing contextual-orchestrator to select the minimum-cost execution path that satisfies the detected quality requirement; explicit modes remain ablation and operator overrides.\n\n"
        "### Fixed\n",
        "changelog",
    )
    changelog_path.write_text(changelog, encoding="utf-8")

    adr_path = ROOT / "docs" / "adr" / "0016-adaptive-contextual-orchestrator-default.md"
    if adr_path.exists():
        raise RuntimeError(f"refusing to replace existing ADR: {adr_path}")
    adr_path.write_text(ADR, encoding="utf-8")


ADR = '''# ADR-0016: Adaptive contextual-orchestrator mode is the Judge default

- Status: Accepted
- Date: 2026-08-15

## Context

`ContextualOrchestratorJudge` previously forced `mode="route"`, which bypassed the
orchestrator's task-sensitive choice between a single worker and deeper verified
workflows. The Judge must remain provider-neutral and must not independently pick
a model, provider, or fixed amount of test-time compute.

Adaptive orchestration research indicates that model/workflow allocation should be
query-dependent rather than fixed. Fugu dynamically devises agentic scaffolds for
a request, while cost-aware routing research shows that no fixed
model-technique-budget choice dominates and that routing should move along a
quality-cost frontier.

## Decision

The constructor default is `mode="auto"`.

- contextual-orchestrator owns model, provider, workflow-depth, verification, and
  cost selection;
- fast-mlsirm continues to own rubric validation, strict JSON parsing, category
  semantics, and psychometric evidence;
- `route` and `conduct` remain explicit controls for ablation, diagnostics, and
  emergency operator policy;
- a caller that needs a reproducible study must record the requested mode and the
  returned orchestration trace.

## Consequences

A production Judge no longer silently reduces every evaluation to a single model.
The actual trace may contain one or several steps, so tests assert the requested
default policy rather than a fixed trace width. Cost and quality claims must be
validated from contextual-orchestrator evidence; fast-mlsirm does not infer them
from mode names.

## References

Omidvar, H., & Akhlaghi, V. (2026). *A communication-theoretic framework for LLM agents: Cost-aware adaptive reliability* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2605.09121

Tang, Y., Cetin, E., Xu, J., Sun, Q., Nielsen, S., Richard, V., Goda, H., Tymchenko, I., Nguyen, N., Lee, H., Ashiga, M., Kotyan, S., Kuroki, S., & Clanuwat, T. (2026). *Sakana Fugu technical report* [Technical report]. arXiv. https://doi.org/10.48550/arXiv.2606.21228
'''

if __name__ == "__main__":
    main()
