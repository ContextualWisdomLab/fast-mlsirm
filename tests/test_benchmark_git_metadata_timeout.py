"""Fail-first reliability contract for benchmark Git metadata lookup."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_benchmark_report():
    """Load the benchmark-report script as an importable module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("build_benchmark_report_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path) -> None:
    """A hung Git metadata lookup must fail closed on a short package deadline."""
    module = _load_benchmark_report()
    calls: list[dict[str, object]] = []

    def timeout_run(*args, **kwargs):
        calls.append(dict(kwargs))
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout", 0),
            output="GIT_TIMEOUT_STDOUT_SECRET",
            stderr="GIT_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    with pytest.raises(RuntimeError) as captured:
        module._source_commit(tmp_path)

    assert len(calls) == 1
    assert calls[0]["timeout"] == module.GIT_METADATA_TIMEOUT_SECONDS
    assert 0 < module.GIT_METADATA_TIMEOUT_SECONDS <= 30
    message = str(captured.value)
    assert message == "source commit lookup timed out"
    assert "GIT_TIMEOUT_STDOUT_SECRET" not in message
    assert "GIT_TIMEOUT_STDERR_SECRET" not in message
