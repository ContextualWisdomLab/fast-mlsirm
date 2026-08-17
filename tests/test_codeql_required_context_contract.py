"""Regression contract for the required CodeQL Actions-language check."""

from pathlib import Path


def _workflow_text() -> str:
    """Return the repository CodeQL workflow source."""
    return Path(".github/workflows/codeql.yml").read_text(encoding="utf-8")


def test_codeql_actions_context_runs_for_pull_requests() -> None:
    """PRs must have a producer for the protected `Analyze (actions)` context."""
    workflow = _workflow_text()

    assert "pull_request:" in workflow
    assert "analyze-actions:" in workflow
    assert "name: Analyze (actions)" in workflow


def test_codeql_python_advanced_job_remains_manual_only() -> None:
    """Default setup keeps Python coverage; advanced Python runs stay manual."""
    workflow = _workflow_text()

    assert "analyze-python:" in workflow
    assert "if: github.event_name == 'workflow_dispatch'" in workflow
    assert "name: Analyze (python)" in workflow
