"""Contract tests for the fast-mlsirm hourly review-repair caller."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "hourly-review-repair.yml"
CENTRAL_REVIEW_REPAIR_SHA = "2f16cca4aae2d11ccc928f8e03fdcbd97a96d5a2"


def _workflow_text() -> str:
    """Return the committed caller workflow as UTF-8 text."""
    return WORKFLOW.read_text(encoding="utf-8")


def test_hourly_caller_is_default_branch_schedule_only_and_offset() -> None:
    """The maintenance heartbeat runs hourly away from minute-zero congestion."""
    workflow = _workflow_text()
    assert 'cron: "37 * * * *"' in workflow
    assert "workflow_dispatch:" not in workflow
    assert "repository_dispatch:" not in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow


def test_hourly_caller_uses_one_immutable_central_engine() -> None:
    """The caller cannot select mutable central scheduler implementation."""
    workflow = _workflow_text()
    expected = (
        "ContextualWisdomLab/.github/.github/workflows/"
        f"pr-review-fix-scheduler.yml@{CENTRAL_REVIEW_REPAIR_SHA}"
    )
    assert f"uses: {expected}" in workflow
    assert not re.search(r"uses: .*@(main|master|develop|HEAD)(?:\s|$)", workflow)
    assert "canonical_ref:" not in workflow


def test_hourly_caller_is_bounded_to_fast_mlsirm_current_head_repairs() -> None:
    """Exactly one same-head repair dispatch is allowed per hourly scan."""
    workflow = _workflow_text()
    assert "target_repository: ContextualWisdomLab/fast-mlsirm" in workflow
    assert "base_branch: main" in workflow
    assert 'max_prs: "50"' in workflow
    assert 'max_dispatches: "1"' in workflow
    assert 'retry_hours: "1"' in workflow
    assert "cancel-in-progress: true" in workflow


def test_hourly_caller_preserves_least_privilege_and_secret_boundaries() -> None:
    """The GitHub token stays read-only while explicit scheduler credentials cross."""
    workflow = _workflow_text()
    assert "permissions:\n  contents: read" in workflow
    assert "permissions:\n      contents: read" in workflow
    assert workflow.count("contents: read") == 2
    assert "write" not in workflow
    assert "secrets: inherit" not in workflow
    assert "COPILOT_GITHUB_TOKEN" not in workflow
    assert "NVIDIA_NIM_API_KEY" not in workflow
    assert (
        "PR_REVIEW_MERGE_TOKEN: ${{ secrets.PR_REVIEW_MERGE_TOKEN }}" in workflow
    )
    assert (
        "OPENCODE_APPROVE_TOKEN: ${{ secrets.OPENCODE_APPROVE_TOKEN }}" in workflow
    )
