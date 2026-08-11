"""Reliability contracts for procurement source commit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_procurement_builder():
    """Load the procurement due-diligence builder as a standalone module."""
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_procurement_due_diligence.py"
    )
    spec = importlib.util.spec_from_file_location("procurement_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path):
    """A hung local Git lookup must fail promptly with a stable package error."""
    module = _load_procurement_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=kwargs.get("timeout", 999),
            output="PROCUREMENT_TIMEOUT_STDOUT_SECRET",
            stderr="PROCUREMENT_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_forwards_bounded_deadline_on_success(monkeypatch, tmp_path):
    """Successful Git metadata lookup uses the same package-owned deadline."""
    module = _load_procurement_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="abc123\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._source_commit(tmp_path) == "abc123"
    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_keeps_non_timeout_unknown_fallback(monkeypatch, tmp_path):
    """Ordinary non-timeout Git failures retain the historical unknown fallback."""
    module = _load_procurement_builder()

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._source_commit(tmp_path) == "unknown"
