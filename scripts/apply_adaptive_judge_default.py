#!/usr/bin/env python3
"""Make contextual-orchestrator auto policy the LLM Judge product default."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "python" / "fast_mlsirm" / "llm_judge.py"
ADR_PATH = ROOT / "docs" / "adr" / "0021-adaptive-contextual-orchestrator-default.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"

source = SOURCE_PATH.read_text(encoding="utf-8")
old_signature = 'def __init__(self, orchestrator: Any, *, mode: str = "route", accept_threshold: float = 0.7) -> None:'
new_signature = 'def __init__(self, orchestrator: Any, *, mode: str = "auto", accept_threshold: float = 0.7) -> None:'
if new_signature not in source:
    if source.count(old_signature) != 1:
        raise RuntimeError(
            f"Judge constructor: expected one legacy signature, found {source.count(old_signature)}"
        )
    source = source.replace(old_signature, new_signature, 1)
source = source.replace(
    'class ContextualOrchestratorJudge:\n    """Evaluate one answer through an injected contextual-orchestrator."""',
    'class ContextualOrchestratorJudge:\n    """Evaluate one answer through contextual-orchestrator\'s adaptive policy by default.\n\n    ``auto`` delegates test-time-compute allocation to the orchestration plane.\n    Explicit ``route`` and ``conduct`` remain controlled ablation overrides.\n    """',
    1,
)
SOURCE_PATH.write_text(source, encoding="utf-8")

ADR_PATH.parent.mkdir(parents=True, exist_ok=True)
if not ADR_PATH.exists():
    ADR_PATH.write_text(
        '''# ADR-0021: LLM Judge defaults to contextual-orchestrator auto policy

- Status: Accepted
- Date: 2026-08-16

## Context

`ContextualOrchestratorJudge` previously defaulted to `route`, which forced every
judgment through one worker even though evaluation is exactly the workload that
benefits from an independent verifier. The reusable psychometric package must not
own provider or workflow selection; it should own rubric validity, bounded parsing,
and IRT projection while delegating test-time computation to the orchestration
plane.

## Decision

The Judge default is `mode="auto"`. Contextual-orchestrator decides whether the
request needs a direct route, a worker-plus-verifier judgment, or a deeper conducted
workflow. Explicit `route` and `conduct` remain available only for controlled
ablation, compatibility tests, and operator rollback.

The default does not assert that extra model calls are always superior. It lets the
central policy maximize quality first and minimize known cost among
quality-equivalent execution plans. Missing price metadata is not treated as free.

## Consequences

Consumers such as LineageWeave no longer need to encode orchestration policy.
`LLMJudgeResult` continues recording the returned mode, trace width, and usage so
calibration studies can compare execution tiers without changing the response
contract.

## References

Omidvar, H., & Akhlaghi, V. (2026). *A communication-theoretic framework for LLM agents: Cost-aware adaptive reliability* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2605.09121

Tang, Y., Cetin, E., Xu, J., Sun, Q., Nielsen, S., Richard, V., Goda, H., Tymchenko, I., Nguyen, N., Lee, H., Ashiga, M., Kotyan, S., Kuroki, S., & Clanuwat, T. (2026). *Sakana Fugu technical report* [Technical report]. arXiv. https://doi.org/10.48550/arXiv.2606.21228
''',
        encoding="utf-8",
    )

changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
entry = (
    "- Default provider-neutral LLM Judge calls to contextual-orchestrator `auto` "
    "policy so the central quality-first, cost-aware control plane chooses route, "
    "verification, or conducted workflow; explicit modes remain ablation overrides.\n"
)
if entry not in changelog:
    marker = "### Fixed\n\n"
    if marker not in changelog:
        raise RuntimeError("CHANGELOG Unreleased/Fixed marker was not found")
    changelog = changelog.replace(marker, marker + entry, 1)
    CHANGELOG_PATH.write_text(changelog, encoding="utf-8")
