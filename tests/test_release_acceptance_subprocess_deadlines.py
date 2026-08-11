"""Fail-first deadline contract for release-acceptance CLI subprocesses."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_release_acceptance():
    """Load the release-acceptance script as a testable module."""
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    script = Path(__file__).resolve().parents[1] / "scripts" / "release_acceptance.py"
    spec = importlib.util.spec_from_file_location("release_acceptance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fit_cli_timeout_is_bounded_and_fails_closed_without_reflection(
    monkeypatch,
) -> None:
    """A hung fit subprocess must use a bounded deadline and a redacted stable error."""
    module = _load_release_acceptance()
    observed: dict[str, object] = {}

    def fake_run(command, *args, timeout=None, **kwargs):
        observed["timeout"] = timeout
        raise subprocess.TimeoutExpired(
            cmd=["release-acceptance-secret-command"],
            timeout=timeout if timeout is not None else 999_999,
            output="RELEASE_ACCEPTANCE_SECRET_STDOUT",
            stderr="RELEASE_ACCEPTANCE_SECRET_STDERR",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as captured:
        module._run_cli(["fit", "--backend", "rust"], "fit_auto")

    timeout = observed.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < float(timeout) <= 900

    message = str(captured.value)
    assert message == "fit_auto timed out"
    assert "RELEASE_ACCEPTANCE_SECRET" not in message
    assert "release-acceptance-secret-command" not in message
