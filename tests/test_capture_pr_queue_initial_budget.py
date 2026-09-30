"""Focused cumulative-budget tests for live PR queue capture."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the capture script without requiring ``scripts`` to be a package."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot_budget", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_budget_stops_initial_queries_after_expiry():
    """No later live query starts once the cumulative capture budget expires."""
    module = _module()
    commands: list[list[str]] = []
    ticks = iter([0.0, 0.0, 2.0])

    def monotonic() -> float:
        return next(ticks)

    def run_json(command: Sequence[str]):
        commands.append(list(command))
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        raise AssertionError(f"capture budget must prevent another live query: {command}")

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=run_json,
        monotonic=monotonic,
        capture_budget_seconds=1.0,
    )

    assert [command[1:3] for command in commands] == [["repo", "view"]]
    assert snapshot["open_prs"] == []
    assert snapshot["pr_history"] == []
    assert snapshot["base_sha"] == ""
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 124
    assert "cumulative capture budget" in snapshot["errors"][0]["stderr"]


def test_detail_deadline_failure_does_not_duplicate_skipped_base_error():
    """One deadline failure stops dependent capture without a second skipped-query error."""
    module = _module()
    commands: list[list[str]] = []
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])

    def monotonic() -> float:
        return next(ticks)

    def run_json(command: Sequence[str]):
        command = list(command)
        commands.append(command)
        identity = command[1:3]
        if identity == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if identity == ["pr", "list"] and "open" in command:
            return [{"number": 17}], None
        if identity == ["pr", "list"]:
            return [], None
        if identity == ["pr", "view"]:
            return None, {
                "command": ["pr", "view"],
                "stderr": "GitHub command exceeded the cumulative capture deadline",
                "returncode": 124,
            }
        raise AssertionError(f"base query must be skipped after deadline failure: {command}")

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=run_json,
        monotonic=monotonic,
        capture_budget_seconds=1.0,
    )

    assert [command[1:3] for command in commands] == [
        ["repo", "view"],
        ["pr", "list"],
        ["pr", "list"],
        ["pr", "view"],
    ]
    assert snapshot["base_sha"] == ""
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 124
    assert "cumulative capture deadline" in snapshot["errors"][0]["stderr"]
