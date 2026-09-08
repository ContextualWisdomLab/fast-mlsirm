"""Contracts for exact-head scientific evidence on compiler-baseline changes."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_STUDIES = REPO_ROOT / ".github" / "workflows" / "statistical-studies.yml"


def _event_block(workflow: str) -> str:
    """Return the workflow trigger block without parsing GitHub's ``on`` key as YAML 1.1."""

    return workflow.split("\npermissions:", 1)[0]


def test_rust_toolchain_change_runs_scientific_studies_on_pull_request() -> None:
    """Compiler-baseline PRs must execute recovery/parity studies before landing."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    events = _event_block(workflow)

    assert "\n  pull_request:\n" in events
    assert "    paths:\n      - \"rust-toolchain.toml\"\n" in events
