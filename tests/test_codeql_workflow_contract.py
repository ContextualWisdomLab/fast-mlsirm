"""Regression contract for the required Actions-language CodeQL PR gate."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "codeql.yml"
CODEQL_ACTION_SHA = "db488ddef3bf6cb639b32c2e9a7c0a7ea8271d28"


def _job_block(workflow: str, job_id: str, next_job_id: str | None = None) -> str:
    """Return one top-level workflow job block from the repository YAML text."""
    start = workflow.index(f"  {job_id}:\n")
    if next_job_id is None:
        return workflow[start:]
    end = workflow.index(f"  {next_job_id}:\n", start + 1)
    return workflow[start:end]


def test_actions_codeql_runs_on_pull_requests_while_python_stays_manual() -> None:
    """Keep the required Actions context reachable without duplicating Python CodeQL."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = workflow.split("\npermissions:\n", 1)[0]

    assert "  pull_request:\n" in trigger_block
    assert "  workflow_dispatch:\n" in trigger_block

    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "name: Analyze (actions)" in actions_job
    assert "languages: actions" in actions_job
    assert "github.event_name == 'workflow_dispatch'" not in actions_job

    assert "name: Analyze (python)" in python_job
    assert "languages: python" in python_job
    assert "if: github.event_name == 'workflow_dispatch'" in python_job


def test_advanced_jobs_do_not_upload_while_default_setup_is_enabled() -> None:
    """Run real CodeQL queries without competing with default setup SARIF ownership."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "upload: never" in actions_job
    assert "upload: never" in python_job


def test_codeql_workflow_keeps_pinned_actions_and_least_permissions() -> None:
    """Keep init/analyze on one reviewed CodeQL release SHA with least privilege."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    init_pin = f"github/codeql-action/init@{CODEQL_ACTION_SHA}"
    analyze_pin = f"github/codeql-action/analyze@{CODEQL_ACTION_SHA}"

    assert "permissions:\n  contents: read\n" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert workflow.count("github/codeql-action/init@") == 2
    assert workflow.count("github/codeql-action/analyze@") == 2
    assert workflow.count(init_pin) == 2
    assert workflow.count(analyze_pin) == 2
