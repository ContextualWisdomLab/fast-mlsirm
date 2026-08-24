"""Fail-first reliability contracts for release evidence Git metadata reads."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_release_index():
    """Load the release evidence index builder for boundary tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_release_evidence_index.py"
    spec = importlib.util.spec_from_file_location("build_release_evidence_index", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_bounds_git_metadata_lookup(monkeypatch, tmp_path: Path) -> None:
    """A hung ``git rev-parse`` must fail closed under a package-owned deadline."""
    module = _load_release_index()
    observed_timeouts: list[object] = []

    def timeout_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args", ["git", "rev-parse", "HEAD"]),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    with pytest.raises(RuntimeError, match="source commit lookup timed out"):
        module._source_commit(tmp_path)

    assert observed_timeouts == [module.GIT_METADATA_TIMEOUT_SECONDS]
