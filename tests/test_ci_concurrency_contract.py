"""Contract tests for pull-request CI concurrency."""

from __future__ import annotations

from pathlib import Path


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    """Return the repository CI workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def test_ci_cancels_superseded_runs_for_the_same_pull_request():
    """A newer PR head invalidates queued or running evidence for the old head."""
    workflow = _workflow_text()
    pre_jobs = workflow.split("jobs:", 1)[0]
    assert "concurrency:" in pre_jobs
    assert "github.event.pull_request.number" in pre_jobs
    assert "cancel-in-progress: true" in pre_jobs


def test_ci_push_runs_remain_scoped_by_ref():
    """Main/develop push evidence cannot cancel an unrelated branch or PR run."""
    pre_jobs = _workflow_text().split("jobs:", 1)[0]
    assert "github.ref" in pre_jobs
    assert "github.event.pull_request.head.sha" not in pre_jobs
    assert "github.run_id" not in pre_jobs
