"""Regression tests for bounded GitHub CLI subprocess execution."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType


_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pr_queue_gh_timeout_fails_closed_without_retry(monkeypatch):
    """A hung GitHub CLI call returns bounded redacted evidence after one attempt."""
    module = _load_script("build_pr_queue_governance")
    calls: list[tuple[list[str], int | None]] = []

    def fake_run(command, *, capture_output=True, text=True, timeout=None):
        calls.append((list(command), timeout))
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload, error = module._run_gh_json(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            "ContextualWisdomLab/fast-mlsirm",
            "--json",
            "number",
        ],
        max_attempts=3,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert calls == [
        (
            [
                "gh",
                "pr",
                "list",
                "--repo",
                "ContextualWisdomLab/fast-mlsirm",
                "--json",
                "number",
            ],
            module._GH_COMMAND_TIMEOUT_SECONDS,
        )
    ]
    assert module._GH_COMMAND_TIMEOUT_SECONDS == 60
    assert error == {
        "command": ["pr", "list"],
        "stderr": "command timed out after 60 seconds",
        "returncode": 124,
    }


def test_procurement_github_snapshot_records_each_timeout(monkeypatch):
    """Procurement evidence bounds repo, PR, and release GitHub CLI calls."""
    module = _load_script("build_procurement_due_diligence")
    calls: list[tuple[list[str], int | None]] = []

    def fake_run(command, *, capture_output=True, text=True, timeout=None):
        calls.append((list(command), timeout))
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    snapshot = module._github_snapshot(
        "ContextualWisdomLab/fast-mlsirm",
        offline=False,
    )

    assert module._GH_COMMAND_TIMEOUT_SECONDS == 60
    assert len(calls) == 3
    assert all(timeout == module._GH_COMMAND_TIMEOUT_SECONDS for _, timeout in calls)
    assert [command[1:3] for command, _ in calls[:2]] == [
        ["repo", "view"],
        ["pr", "list"],
    ]
    assert calls[2][0][1:3] == ["release", "list"]
    expected_timeout = {
        "ok": False,
        "returncode": 124,
        "data": None,
        "stderr": "command timed out after 60 seconds",
    }
    assert snapshot["repo"] == expected_timeout
    assert snapshot["open_prs"] == expected_timeout
    assert snapshot["releases"] == {
        "ok": False,
        "returncode": 124,
        "lines": [],
        "stderr": "command timed out after 60 seconds",
    }
