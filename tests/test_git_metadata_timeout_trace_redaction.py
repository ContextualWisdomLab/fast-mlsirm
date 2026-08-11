"""Fail-first privacy contract for Git metadata timeout exceptions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


_SCRIPTS = (
    "build_benchmark_report.py",
    "build_buyer_packet.py",
    "build_commercial_release.py",
    "build_figma_evidence_sync.py",
    "build_procurement_due_diligence.py",
)


def _load_script(script_name: str):
    """Load one governed evidence builder as an isolated module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    module_name = f"timeout_trace_{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", _SCRIPTS)
def test_git_timeout_does_not_expose_subprocess_exception_as_public_cause(
    monkeypatch, tmp_path, script_name: str
) -> None:
    """Timeout errors must suppress child exception state from the public chain."""
    module = _load_script(script_name)

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="GIT_TIMEOUT_STDOUT_SECRET",
            stderr="GIT_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$") as captured:
        module._source_commit(tmp_path)

    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True
