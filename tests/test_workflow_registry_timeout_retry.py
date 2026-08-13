"""Fail-first timeout retry coverage for the read-only workflow-registry audit."""

from __future__ import annotations

import subprocess

import scripts.audit_workflow_registry as registry


def test_run_gh_api_retries_timeout_then_succeeds(monkeypatch):
    """A transient GitHub CLI timeout should consume one bounded retry."""
    calls = 0

    def fake_run(args, *, capture_output, text, timeout):
        del capture_output, text
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"ok": true}',
            stderr="",
        )

    monkeypatch.setattr(registry.subprocess, "run", fake_run)

    assert registry._run_gh_api(
        "repos/ContextualWisdomLab/fast-mlsirm",
        max_attempts=2,
        retry_sleep_seconds=0,
    ) == {"ok": True}
    assert calls == 2
