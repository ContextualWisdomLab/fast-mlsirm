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
    assert "if: always()" in text


def test_hourly_governance_workflow_tracks_implementation_and_contract_tests():
    """Changes to duplicate-claim governance trigger the evidence workflow."""
    text = _workflow_text()
    assert "scripts/build_pr_queue_governance.py" in text
    assert "tests/test_pr_queue_governance.py" in text
    assert "tests/test_hourly_pr_governance_workflow.py" in text


def test_governance_workflow_fails_closed_when_no_tests_are_discovered():
    """A renamed or removed test suite cannot make self-verification silently pass."""
    text = _workflow_text()
    assert "test_names = [" in text
    assert "if not test_names:" in text
    guard = 'raise RuntimeError("no hourly governance contract tests were discovered")'
    assert guard in text
    assert text.index("if not test_names:") < text.index(guard)
    assert text.index(guard) < text.index("for name in test_names:")
