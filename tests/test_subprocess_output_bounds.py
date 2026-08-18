"""Regression coverage for bounded subprocess capture in operator scripts."""

from __future__ import annotations

import subprocess
import sys

import pytest

from scripts import build_pr_queue_governance as governance
from scripts import build_procurement_due_diligence as procurement


def test_bounded_capture_rejects_stdout_overflow() -> None:
    """The shared runner must cap stdout while the child is still running."""
    from scripts._bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stdout"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_rejects_stderr_overflow() -> None:
    """The shared runner must cap stderr independently from stdout."""
    from scripts._bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stderr"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stderr.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_governance_parse_failure_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed successful gh output must not crash the governance builder."""
    monkeypatch.setattr(
        governance.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", ""),
    )

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 65
    assert "JSON" in error["stderr"]


def test_governance_timeout_remains_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounded-output successor must preserve the parent timeout contract."""
    def raise_timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(governance.subprocess, "run", raise_timeout)

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 124
    assert "timed out" in error["stderr"]


def test_procurement_parse_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gh JSON must be recorded as evidence failure, not raised."""
    monkeypatch.setattr(
        procurement.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", ""),
    )

    snapshot = procurement._github_snapshot("example/project", offline=False)

    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 65
    assert snapshot["repo"]["data"] is None
    assert "JSON" in snapshot["repo"]["stderr"]
