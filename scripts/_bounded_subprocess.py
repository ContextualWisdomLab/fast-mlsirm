"""Bound subprocess stdout/stderr in memory while preserving a hard deadline."""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

_READ_CHUNK_BYTES = 64 * 1024


class BoundedSubprocessOutputError(RuntimeError):
    """Raised when a captured subprocess stream exceeds its configured limit."""

    def __init__(self, stream: str, limit_bytes: int) -> None:
        self.stream = stream
        self.limit_bytes = limit_bytes
        super().__init__(f"{stream} exceeded bounded capture limit of {limit_bytes} bytes")


def _drain_bounded(
    stream: object,
    *,
    limit_bytes: int,
    buffer: bytearray,
    overflow: threading.Event,
) -> None:
    """Drain one binary pipe without retaining more than ``limit_bytes + 1`` bytes."""
    read = getattr(stream, "read")
    while True:
        chunk = read(_READ_CHUNK_BYTES)
        if not chunk:
            return
        remaining = (limit_bytes + 1) - len(buffer)
        if remaining > 0:
            buffer.extend(chunk[:remaining])
        if len(buffer) > limit_bytes:
            overflow.set()


def run_bounded_capture(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` with deadline and byte-bounded stdout/stderr capture.

    Both pipes are drained concurrently so neither can deadlock the child. As
    soon as either retained stream crosses its byte budget, the child is
    terminated and the overflow is reported without retaining additional
    output. The returned text uses replacement decoding so diagnostics remain
    available even when a tool emits malformed UTF-8.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_stdout_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("output limits must be non-negative")
    if not command:
        raise ValueError("command must not be empty")

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    readers = [
        threading.Thread(
            target=_drain_bounded,
            kwargs={
                "stream": process.stdout,
                "limit_bytes": max_stdout_bytes,
                "buffer": stdout,
                "overflow": stdout_overflow,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded,
            kwargs={
                "stream": process.stderr,
                "limit_bytes": max_stderr_bytes,
                "buffer": stderr,
                "overflow": stderr_overflow,
            },
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None:
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            process.kill()
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            process.kill()
            break
        time.sleep(min(0.01, remaining))

    process.wait()
    for reader in readers:
        reader.join()

    if timed_out:
        raise subprocess.TimeoutExpired(list(command), timeout_seconds)
    if stdout_overflow.is_set():
        raise BoundedSubprocessOutputError("stdout", max_stdout_bytes)
    if stderr_overflow.is_set():
        raise BoundedSubprocessOutputError("stderr", max_stderr_bytes)

    return subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
