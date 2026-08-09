"""Regression for descendants surviving a timed-out POSIX process-group leader."""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest


_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines_group_leak", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


class _LeaderExitsProcess:
    """Timed-out leader that exits after SIGTERM while a descendant group remains."""

    def __init__(self) -> None:
        self.pid = 6262
        self.returncode = 0
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout=None):
        """Timeout once, then model the leader exiting before its descendant."""
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) == 1:
            raise subprocess.TimeoutExpired(["opaque-child"], timeout)
        return (None, None)


def test_posix_timeout_kills_surviving_group_after_leader_exits(monkeypatch) -> None:
    """Cleanup must verify the process group, not only the timed-out leader."""
    fake = _LeaderExitsProcess()
    signals: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        if sig == 0:
            # The process-group probe sees a surviving descendant.
            return

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(deadlines.os, "killpg", fake_killpg)

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "test"],
            operation=deadlines.SubprocessOperation.STATISTICAL_TEST,
        )

    assert (fake.pid, signal.SIGTERM) in signals
    assert (fake.pid, 0) in signals
    assert (fake.pid, signal.SIGKILL) in signals
