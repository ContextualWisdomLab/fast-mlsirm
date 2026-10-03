# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fault checks for the GIL observer, not numerical or speedup evidence."""

import threading
import time

import pytest
import test_gil_detach_fipc as probe


def test_observer_surfaces_worker_exception():
    error = RuntimeError("injected fit failure")
    def broken():
        raise error
    with pytest.raises(RuntimeError, match="injected fit failure") as caught:
        probe._observe_fit(broken, timeout_seconds=1.)
    assert caught.value is error


def test_observer_deadline_does_not_wait_indefinitely(monkeypatch):
    release = threading.Event()
    threads = []
    real_thread = threading.Thread
    def owned_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(probe.threading, "Thread", owned_thread)
    try:
        with pytest.raises(TimeoutError, match="observer deadline"):
            probe._observe_fit(release.wait, timeout_seconds=.02)
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1.)
            assert not thread.is_alive()


def test_observer_keeps_constant_size_timing_state():
    elapsed, gap = probe._observe_fit(lambda: time.sleep(.04), timeout_seconds=1.)
    assert elapsed >= .02
    assert 0 <= gap <= elapsed
    assert isinstance(gap, float)


def test_observer_still_detects_a_gil_held_python_control():
    import sys
    interval = sys.getswitchinterval()
    def hold_gil():
        end = time.perf_counter() + .06
        while time.perf_counter() < end:
            pass
    try:
        sys.setswitchinterval(.2)
        elapsed, gap = probe._observe_fit(hold_gil, timeout_seconds=1.)
    finally:
        sys.setswitchinterval(interval)
    assert gap >= elapsed / 2


def test_numeric_wrapper_has_an_external_process_timeout(monkeypatch):
    import subprocess
    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 90.
        assert command[-1] == "--numeric-probe"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(probe.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        probe.test_fit_poly_fipc_lets_other_python_threads_run()
