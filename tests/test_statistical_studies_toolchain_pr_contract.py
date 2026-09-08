"""Contracts for exact-head scientific evidence on compiler-baseline changes."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_STUDIES = REPO_ROOT / ".github" / "workflows" / "statistical-studies.yml"


def _event_block(workflow: str) -> str:
    """Return the workflow trigger block without parsing GitHub's ``on`` key as YAML 1.1."""

    return workflow.split("\npermissions:", 1)[0]


def _concurrency_block(workflow: str) -> str:
    """Return the workflow concurrency block that governs same-ref study runs."""

    return workflow.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]


def test_rust_toolchain_change_runs_scientific_studies_on_pull_request() -> None:
    """Compiler-baseline PRs must execute recovery/parity studies before landing."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    events = _event_block(workflow)

    assert "\n  pull_request:\n" in events
    assert "    paths:\n      - \"rust-toolchain.toml\"\n" in events


def test_new_toolchain_head_supersedes_stale_same_ref_scientific_run() -> None:
    """A synchronized PR head must not wait behind obsolete Monte Carlo evidence."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    concurrency = _concurrency_block(workflow)

    assert "  group: statistical-studies-${{ github.ref }}\n" in concurrency
    assert "  cancel-in-progress: true\n" in concurrency
