# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT
"""Actual tiny-child deadline regressions, not numerical execution."""
from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest
import fast_mlsirm.remote_exec as remote_exec


@pytest.mark.parametrize("mode", ["held_stdout", "unread_stdin"])
def test_deadline_bounds_descendant_output_and_unread_input(mode, monkeypatch):
    """A stalled local process cannot extend I/O until a descendant exits."""
    bodies = {
        "held_stdout": (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)']); "
            "time.sleep(3)"
        ),
        "unread_stdin": "import time; time.sleep(3)",
    }
    popen = subprocess.Popen
    created = []

    def tiny_child(argv, **kwargs):
        process = popen([sys.executable, "-c", bodies[mode]], **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(remote_exec.subprocess, "Popen", tiny_child)
    started = time.monotonic()
    completed = remote_exec._invoke_worker_process(
        "x" * (2 * 1024 * 1024) if mode == "unread_stdin" else "{}",
        worker_host=socket.gethostname(), remote_interpreter=sys.executable,
        stdout_limit=1024, timeout_seconds=0.2,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, f"configured 0.2s timeout returned after {elapsed:.3f}s"
    assert completed.returncode != 0
    assert "timed out" in completed.stderr
    assert len(created) == 1
    assert created[0].poll() is not None


def test_worker_can_write_output_before_reading_large_input(monkeypatch):
    """Simultaneous bounded reads and partial writes avoid pipe deadlock."""
    body = (
        "import sys; "
        "sys.stdout.buffer.write(b'y' * 131072); sys.stdout.buffer.flush(); "
        "request = sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(str(len(request)).encode())"
    )
    popen = subprocess.Popen
    monkeypatch.setattr(
        remote_exec.subprocess, "Popen",
        lambda argv, **kwargs: popen([sys.executable, "-c", body], **kwargs),
    )
    completed = remote_exec._invoke_worker_process(
        "x" * 131072, worker_host=socket.gethostname(), remote_interpreter=sys.executable,
        stdout_limit=262144, timeout_seconds=2.0,
    )
    assert completed.returncode == 0
    assert completed.stdout == "y" * 131072 + "131072"


def test_denied_group_signal_still_bounds_driver_io(monkeypatch):
    """A denied group signal falls back to the owned direct child only."""
    popen = subprocess.Popen
    created = []
    signals = []

    def tiny_child(argv, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(3)"], **kwargs)
        created.append(process)
        return process

    def denied_group(pid, sig):
        signals.append((pid, sig))
        raise PermissionError("group cancellation denied")

    monkeypatch.setattr(remote_exec.subprocess, "Popen", tiny_child)
    monkeypatch.setattr(remote_exec.os, "killpg", denied_group, raising=False)
    started = time.monotonic()
    completed = remote_exec._invoke_worker_process(
        "{}", worker_host=socket.gethostname(), remote_interpreter=sys.executable,
        stdout_limit=1024, timeout_seconds=0.2,
    )
    assert time.monotonic() - started < 1.5
    assert completed.returncode != 0
    assert "timed out" in completed.stderr
    assert created[0].poll() is not None
    if remote_exec.os.name == "posix":
        assert len(signals) == 1
        assert signals[0][0] == created[0].pid
    else:
        assert not signals


@pytest.mark.parametrize("failure", [
    OSError("fixture-owned pipe failed"),
    subprocess.TimeoutExpired(["fixture-owned-worker"], 1.0),
])
def test_worker_transport_exception_is_failed_outcome(monkeypatch, failure):
    """A bounded wait expiry is a transport failure, not a batch exception.

    Popen.wait raises TimeoutExpired (Python Software Foundation, n.d.,
    subprocess documentation, Popen.wait and TimeoutExpired entries).
    """
    from test_remote_exec import _envelope, _payload_manifest

    payload: dict[str, object] = {"fixture_control": True}
    manifest = _payload_manifest(payload)
    envelope = _envelope(manifest=manifest)
    calls = []

    def failed_transport(*args, **kwargs):
        calls.append(True)
        raise failure

    monkeypatch.setattr(remote_exec, "_invoke_worker_process", failed_transport)
    ledger = remote_exec.OutcomeCommitLedger()
    outcome = remote_exec.SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
        (envelope,), worker_manifest=manifest, payload=payload,
    )[0]
    assert outcome.delivery_state is remote_exec.RemoteJobDeliveryState.FAILED
    assert outcome.result is None
    assert outcome.output_identity_sha256 is None
    assert outcome.error_message == str(failure)
    assert calls == [True]
    assert ledger.successful_count(remote_exec.envelope_fingerprint(envelope)) == 0


def test_reap_timeout_is_not_retried_with_a_second_allowance(monkeypatch):
    """One reap timeout consumes the documented allowance once, not twice."""
    import io

    class Pipe(io.BytesIO):
        def fileno(self):
            return 123

    class Reader(io.RawIOBase):
        def fileno(self):
            return 123

        def read(self, size: int = -1) -> bytes | None:
            return None

    class Process:
        pid = 4242

        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Reader()
            self.waits = []
            self.kills = 0

        def poll(self):
            return None

        def kill(self):
            self.kills += 1

        def wait(self, timeout: float | None = None):
            assert timeout is not None
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired(["fixture-owned-worker"], timeout)

    process = Process()
    monkeypatch.setattr(remote_exec.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(remote_exec.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(remote_exec.os, "killpg", lambda *args: process.kill(), raising=False)
    with pytest.raises(subprocess.TimeoutExpired):
        remote_exec._invoke_worker_process(
            "{}", worker_host=socket.gethostname(), remote_interpreter=sys.executable,
            stdout_limit=1024, timeout_seconds=0.01,
        )
    assert process.waits == [1.0]
    assert process.kills == 1
    assert process.stdin.closed
    assert process.stdout.closed
