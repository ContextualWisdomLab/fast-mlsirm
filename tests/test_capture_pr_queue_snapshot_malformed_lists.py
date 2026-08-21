"""Regression tests for malformed PR queue list payloads."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the queue snapshot script as an isolated module."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot_malformed", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detail(number: int) -> dict[str, Any]:
    """Return one complete open-PR detail payload."""
    return {
        "number": number,
        "title": f"PR {number}",
        "body": "",
        "headRefName": f"feat/{number}",
        "headRefOid": f"{number:040x}",
        "baseRefName": "main",
        "isDraft": False,
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "REVIEW_REQUIRED",
        "state": "OPEN",
        "updatedAt": "2026-08-21T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
        "url": f"https://github.com/o/r/pull/{number}",
        "labels": [],
        "files": [{"path": "scripts/example.py"}],
    }


def _state(command: Sequence[str]) -> str:
    """Return the requested PR-list state from one GitHub CLI command."""
    return str(command[command.index("--state") + 1])


def test_mixed_open_identity_payload_cannot_claim_complete_snapshot():
    """A non-object open-list entry must remain visible as malformed evidence."""
    module = _module()

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [{"number": 11}, "malformed-entry"], None
        if list(command[1:3]) == ["pr", "view"]:
            return _detail(int(command[3])), None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "all":
            return [], None
        if command[1] == "api":
            return {"sha": "a" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert [item["number"] for item in snapshot["open_prs"]] == [11]
    assert snapshot["open_pr_identity_count"] == 2
    assert any(
        error["returncode"] == 65
        and "open PR identity payload contained non-object entries" in error["stderr"]
        for error in snapshot["errors"]
    )


def test_mixed_history_payload_cannot_claim_complete_snapshot():
    """A non-object history entry must fail closed without discarding valid evidence."""
    module = _module()
    history_record = {
        "number": 7,
        "title": "Historical PR",
        "body": "",
        "headRefName": "feat/history",
        "headRefOid": "b" * 40,
        "state": "MERGED",
        "updatedAt": "2026-08-20T00:00:00Z",
        "closedAt": "2026-08-20T00:00:00Z",
        "mergedAt": "2026-08-20T00:00:00Z",
        "url": "https://github.com/o/r/pull/7",
    }

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [], None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "all":
            return [history_record, 7], None
        if command[1] == "api":
            return {"sha": "c" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["pr_history"] == [history_record]
    assert any(
        error["returncode"] == 65
        and "PR history payload contained non-object entries" in error["stderr"]
        for error in snapshot["errors"]
    )
