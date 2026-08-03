"""Contract tests for the hourly PR-queue governance workflow."""

from __future__ import annotations

from pathlib import Path


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "hourly-pr-governance.yml"


def _workflow_text() -> str:
    """Return the scheduled workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def test_hourly_governance_workflow_exists_and_runs_every_hour():
    """The repository records queue-governance evidence on an hourly cadence."""
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert 'cron: "0 * * * *"' in text


def test_hourly_governance_workflow_is_read_only_and_bounded():
    """The scheduled loop cannot mutate pull requests or repository contents."""
    text = _workflow_text()
    assert "contents: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "cancel-in-progress: true" in text
    assert "timeout-minutes: 10" in text


def test_hourly_governance_workflow_builds_and_publishes_evidence():
    """Each run invokes the governed builder and uploads both audit artifacts."""
    text = _workflow_text()
    assert "python scripts/build_pr_queue_governance.py" in text
    assert "hourly-pr-queue-governance" in text
    assert "pr_queue_governance_manifest.json" in text
    assert "pr_queue_governance_report.html" in text
    assert "actions/upload-artifact" in text
