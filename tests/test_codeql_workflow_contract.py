"""Regression contract for the required Actions-language CodeQL PR gate."""

from __future__ import annotations
from tests.workflow_contract_source import workflow_source

from pathlib import Path


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "codeql.yml"


def _job_block(workflow: str, job_id: str, next_job_id: str | None = None) -> str:
    """Return one top-level workflow job block from the repository YAML text."""
    start = workflow.index(f"  {job_id}:\n")
    if next_job_id is None:
        return workflow[start:]
    end = workflow.index(f"  {next_job_id}:\n", start + 1)
    return workflow[start:end]


def test_actions_codeql_runs_on_pull_requests_while_python_stays_manual() -> None:
    """Keep the required Actions context reachable without duplicating Python CodeQL."""
    wrapper = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = wrapper.split("\npermissions:\n", 1)[0]

    assert "  pull_request:\n" in trigger_block
    assert "  workflow_dispatch:\n" in trigger_block

    workflow = workflow_source(WORKFLOW_PATH)
    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "name: Analyze (actions)" in actions_job
    assert "languages: actions" in actions_job
    assert "\n    if:" not in actions_job

    assert "name: Analyze (python)" in python_job
    assert "languages: python" in python_job
    assert "if: github.event_name == 'workflow_dispatch'" in python_job


def test_advanced_jobs_do_not_upload_while_default_setup_is_enabled() -> None:
    """Run real CodeQL queries without competing with default setup SARIF ownership."""
    workflow = workflow_source(WORKFLOW_PATH)
    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "upload: never" in actions_job
    assert "upload: never" in python_job


def test_codeql_workflow_keeps_pinned_actions_and_least_permissions() -> None:
    """The trigger repair must not loosen action pinning or workflow permissions."""
    workflow = workflow_source(WORKFLOW_PATH)

    assert "permissions:\n  contents: read\n" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "github/codeql-action/init@ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd" in workflow
    assert "github/codeql-action/analyze@ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd" in workflow


def test_self_hosted_diagnostics_use_isolated_runners_without_persisted_credentials() -> None:
    workflow = workflow_source(WORKFLOW_PATH)
    for job in (_job_block(workflow, "analyze-actions", "analyze-python"),
                _job_block(workflow, "analyze-python")):
        assert "    runs-on:\n      group: CWL CI isolated\n      labels: [self-hosted, Linux, X64]" in job
        assert "CWL central CodeQL" not in job
        assert "ubuntu-latest" not in job
        assert "persist-credentials: false" in job
        assert "if-no-files-found: error" in job
        assert "upload: never" in job
