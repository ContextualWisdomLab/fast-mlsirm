"""Contract tests for the thin hourly PR-review autofix caller."""

from __future__ import annotations

from pathlib import Path


_WORKFLOW = Path(".github/workflows/hourly-pr-review-fix.yml")
_CENTRAL_SHA = "5983b41ace75040c1d81818171ca7d0f3653254e"


def _workflow_text() -> str:
    """Read the repository-local cadence-only workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def test_hourly_caller_uses_an_off_peak_hourly_schedule():
    """The requested hourly cadence runs away from the congested hour boundary."""
    text = _workflow_text()
    assert 'cron: "37 * * * *"' in text
    assert "workflow_dispatch:" in text


def test_hourly_caller_reuses_the_pinned_central_scheduler():
    """No repository-local autofix implementation may drift from org governance."""
    text = _workflow_text()
    assert (
        "uses: ContextualWisdomLab/.github/.github/workflows/"
        f"pr-review-fix-scheduler.yml@{_CENTRAL_SHA}"
    ) in text
    assert "runs-on:" not in text
    assert "steps:" not in text


def test_hourly_caller_targets_only_fast_mlsirm_with_bounded_dispatch():
    """Each hourly sweep dispatches at most one same-repository autofix candidate."""
    text = _workflow_text()
    assert "target_repository: ContextualWisdomLab/fast-mlsirm" in text
    assert "base_branch: main" in text
    assert 'max_dispatches: "1"' in text
    assert 'retry_hours: "1"' in text
    assert "secrets: inherit" in text


def test_hourly_caller_grants_only_scheduler_required_permissions():
    """The caller exposes no broader token permissions than the central contract."""
    text = _workflow_text()
    for permission in (
        "actions: write",
        "contents: read",
        "issues: write",
        "pull-requests: read",
        "statuses: read",
    ):
        assert permission in text
    for forbidden in ("contents: write", "pull-requests: write", "id-token: write"):
        assert forbidden not in text
