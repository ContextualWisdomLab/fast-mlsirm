"""Focused regressions for bounded GitHub CLI output capture."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_gh_json_uses_exact_capture_bounds(monkeypatch):
    """The queue reader applies reviewed stdout/stderr ceilings to every gh call."""
    module = _module()
    observed: dict[str, object] = {}

    def bounded(command, **kwargs):
        observed["command"] = list(command)
        observed.update(kwargs)
        return subprocess.CompletedProcess(list(command), 0, '{"ok": true}', "")

    monkeypatch.setattr(module, "run_bounded_capture", bounded)

    payload, error = module._run_gh_json(["gh", "api", "x"])

    assert payload == {"ok": True}
    assert error is None
    assert observed == {
        "command": ["gh", "api", "x"],
        "timeout_seconds": 30.0,
        "max_stdout_bytes": 10 * 1024 * 1024,
        "max_stderr_bytes": 1024 * 1024,
    }


def test_run_gh_json_caps_attempt_to_remaining_capture_deadline(monkeypatch):
    """A command admitted near the cumulative deadline cannot consume 30 seconds."""
    module = _module()
    observed: dict[str, object] = {}

    def bounded(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(list(command), 0, '{"ok": true}', "")

    monkeypatch.setattr(module, "run_bounded_capture", bounded)

    payload, error = module._run_gh_json(
        ["gh", "api", "x"],
        deadline=100.25,
        monotonic=lambda: 100.0,
    )

    assert payload == {"ok": True}
    assert error is None
    assert observed["timeout_seconds"] == 0.25


def test_run_gh_json_fails_closed_without_retry_on_capture_overflow(monkeypatch):
    """Oversized command output is an audit failure, not a transient gateway retry."""
    module = _module()
    calls = 0

    def overflow(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise module.BoundedSubprocessOutputError("stdout", 10 * 1024 * 1024)

    monkeypatch.setattr(module, "run_bounded_capture", overflow)

    payload, error = module._run_gh_json(
        ["gh", "api", "x"],
        max_attempts=3,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 125
    assert "output exceeded bounds" in error["stderr"]
    assert calls == 1
