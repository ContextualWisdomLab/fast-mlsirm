"""Regression coverage for bounded subprocess capture in operator scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from scripts import build_pr_queue_governance as governance
from scripts import build_procurement_due_diligence as procurement
from scripts._bounded_subprocess import BoundedSubprocessOutputError


def test_bounded_capture_rejects_stdout_overflow() -> None:
    """The shared runner must cap stdout while the child is still running."""
    from scripts._bounded_subprocess import run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stdout"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_rejects_stderr_overflow() -> None:
    """The shared runner must cap stderr independently from stdout."""
    from scripts._bounded_subprocess import run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stderr"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stderr.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_rejects_invalid_utf8_stdout() -> None:
    """Machine-readable stdout must never be silently replacement-decoded."""
    from scripts._bounded_subprocess import run_bounded_capture

    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'{\"record\":\"ok' + bytes([255]) + b'\"}')",
    ]
    with pytest.raises(UnicodeError):
        run_bounded_capture(
            command,
            timeout_seconds=5,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_bounded_capture_deadline_kills_pipe_inheriting_descendants() -> None:
    """A descendant holding captured pipes cannot extend the configured deadline."""
    from scripts._bounded_subprocess import run_bounded_capture

    grandchild = "import time; time.sleep(2)"
    child = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "sys.exit(0)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded_capture(
            [sys.executable, "-c", child],
            timeout_seconds=0.2,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )
    assert time.monotonic() - started < 1.0


def test_governance_parse_failure_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed successful gh output must not crash the governance builder."""
    monkeypatch.setattr(
        governance,
        "run_bounded_capture",
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


def test_governance_decode_failure_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTF-8 from gh must map to the existing data-error status."""
    def raise_decode(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(governance, "run_bounded_capture", raise_decode)
    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )
    assert payload is None
    assert error is not None
    assert error["returncode"] == 65


def test_governance_timeout_remains_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounded-output successor must preserve the parent timeout contract."""
    def raise_timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(governance, "run_bounded_capture", raise_timeout)

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 124
    assert "timed out" in error["stderr"]


def test_governance_overflow_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized gh output must fail closed through the governance error schema."""
    def raise_overflow(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise BoundedSubprocessOutputError("stdout", 64)

    monkeypatch.setattr(governance, "run_bounded_capture", raise_overflow)

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 75
    assert "stdout" in error["stderr"]


def test_procurement_parse_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gh JSON must be recorded as evidence failure, not raised."""
    monkeypatch.setattr(
        procurement,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", ""),
    )

    snapshot = procurement._github_snapshot("example/project", offline=False)

    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 65
    assert snapshot["repo"]["data"] is None
    assert "JSON" in snapshot["repo"]["stderr"]


def test_procurement_decode_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTF-8 must be recorded as a stable procurement evidence failure."""
    def raise_decode(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(procurement, "run_bounded_capture", raise_decode)
    snapshot = procurement._github_snapshot("example/project", offline=False)
    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 65
    assert snapshot["repo"]["data"] is None


def test_procurement_overflow_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized gh output must be recorded in procurement evidence, not raised."""
    def raise_overflow(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise BoundedSubprocessOutputError("stdout", 64)

    monkeypatch.setattr(procurement, "run_bounded_capture", raise_overflow)

    snapshot = procurement._github_snapshot("example/project", offline=False)

    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 75
    assert snapshot["repo"]["data"] is None
    assert "stdout" in snapshot["repo"]["stderr"]
