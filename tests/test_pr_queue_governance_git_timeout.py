"""Fail-first reliability contracts for PR queue governance Git metadata reads."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_governance():
    """Load the PR queue governance script as a module for boundary tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_bounds_git_metadata_lookup(monkeypatch, tmp_path: Path) -> None:
    """A hung ``git rev-parse`` must fail closed under a package-owned deadline."""
    module = _load_governance()
    observed_timeouts: list[object] = []

    def timeout_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args", ["git", "rev-parse", "HEAD"]),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    assert observed_timeouts == [module.GIT_METADATA_TIMEOUT_SECONDS]


@pytest.mark.parametrize(
    "expected",
    [
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ],
)
def test_source_commit_accepts_canonical_full_identity(
    monkeypatch, tmp_path: Path, expected: str
) -> None:
    """Canonical full SHA-1 and SHA-256 identities remain supported."""
    module = _load_governance()
    observed_timeouts: list[object] = []

    def completed_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=f"{expected}\n")

    monkeypatch.setattr(module.subprocess, "run", completed_run)

    assert module._source_commit(tmp_path) == expected
    assert observed_timeouts == [module.GIT_METADATA_TIMEOUT_SECONDS]


def test_source_commit_rejects_nonzero_git_exit(monkeypatch, tmp_path: Path) -> None:
    """A failed Git command must not degrade governance provenance to ``unknown``."""
    module = _load_governance()

    def failed_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(module.subprocess, "run", failed_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(
    monkeypatch, tmp_path: Path
) -> None:
    """An unavailable Git executable must block governance evidence generation."""
    module = _load_governance()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", missing_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "unknown",
        "abc123",
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "g" * 40,
        "g" * 64,
        "A" * 40,
        "A" * 64,
    ],
)
def test_source_commit_rejects_noncanonical_identity(
    monkeypatch, tmp_path: Path, stdout: str
) -> None:
    """Malformed or abbreviated Git identities must fail closed."""
    module = _load_governance()

    def completed_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", completed_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup returned invalid identity$"):
        module._source_commit(tmp_path)
