"""Regression coverage for transient GitHub CLI timeout handling."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_governance():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_gh_json_retries_timeout_within_existing_attempt_budget(monkeypatch) -> None:
    """A transport timeout is transient and must consume another bounded attempt."""

    module = _load_governance()
    attempts: list[int] = []
    sleeps: list[float] = []

    def fake_run(command, capture_output=True, text=True, timeout=None):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps([{"number": 1}]),
            "",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=3,
        retry_sleep_seconds=0.01,
    )

    assert error is None
    assert payload == [{"number": 1}]
    assert attempts == [30, 30]
    assert sleeps == [0.01]
