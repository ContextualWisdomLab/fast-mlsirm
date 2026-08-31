"""Fail-first validation tests for the cumulative PR queue capture budget."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the queue-capture script without requiring ``scripts`` to be a package."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot_budget_validation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "capture_budget_seconds",
    [float("nan"), float("inf"), float("-inf"), "30", None],
)
def test_invalid_capture_budget_fails_before_any_live_query(capture_budget_seconds):
    """Non-finite or nonnumeric budgets cannot disable the cumulative deadline."""
    module = _module()
    commands: list[list[str]] = []

    def run_json(command):
        commands.append(list(command))
        return {}, None

    with pytest.raises(
        ValueError,
        match="capture_budget_seconds must be a finite positive number",
    ):
        module.capture_pr_queue_snapshot(
            "owner/repo",
            run_json=run_json,
            capture_budget_seconds=capture_budget_seconds,
        )

    assert commands == []
