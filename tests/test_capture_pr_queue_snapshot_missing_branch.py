"""Regression for fail-closed default-branch snapshot evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the snapshot capture script without requiring a package import."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_fails_closed_when_default_branch_name_is_missing():
    """Incomplete repository identity evidence must be explicit in the snapshot."""
    module = _module()

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {}}, None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["default_branch"] == ""
    assert snapshot["base_sha"] == ""
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 65
    assert "default branch name was missing" in snapshot["errors"][0]["stderr"]
