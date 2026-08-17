"""Regression contracts for bounded repository-owned external commands."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_BOUNDED_FAILURES = (
    "JSON input exceeds maximum allowed size 33554432 bytes",
    "JSON input exceeds maximum allowed depth 128",
)


def _load_script(filename: str) -> ModuleType:
    """Load one repository script under a unique import identity."""
    module_name = f"external_command_bounds_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _successful_command(stdout: str = "{}") -> subprocess.CompletedProcess[str]:
    """Return one deterministic successful external-command result."""
    return subprocess.CompletedProcess(
        args=["external-command"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


@pytest.mark.parametrize("bounded_failure", _BOUNDED_FAILURES)
def test_pr_queue_rejects_bounded_command_json_failures(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Queue governance must not deserialize oversized command output."""
    module = _load_script("build_pr_queue_governance.py")
    monkeypatch.setattr(
        module,
        "parse_json_bounded",
        lambda _content: (_ for _ in ()).throw(ValueError(bounded_failure)),
    )

    assert module._json_from_completed(_successful_command()) is None


@pytest.mark.parametrize("bounded_failure", _BOUNDED_FAILURES)
def test_procurement_rejects_bounded_command_json_failures(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Procurement snapshots fail closed when GitHub JSON exceeds bounds."""
    module = _load_script("build_procurement_due_diligence.py")
    monkeypatch.setattr(
        module,
        "parse_json_bounded",
        lambda _content: (_ for _ in ()).throw(ValueError(bounded_failure)),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: _successful_command(),
    )

    with pytest.raises(ValueError, match=bounded_failure):
        module._github_snapshot("example/project", offline=False)


def test_pr_queue_normalizes_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung GitHub CLI request becomes bounded retry evidence."""
    module = _load_script("build_pr_queue_governance.py")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=30)

    monkeypatch.setattr(module.subprocess, "run", _timeout)
    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=2,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 124
    assert error["stderr"] == "GitHub API request timed out"


def test_commercial_release_normalizes_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A hung commercial-release child becomes structured failed evidence."""
    module = _load_script("build_commercial_release.py")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=300)

    monkeypatch.setattr(module.subprocess, "run", _timeout)
    completed = module._run_command(["python", "stage.py"], tmp_path)

    assert completed.returncode == 124
    assert completed.stdout == ""
    assert completed.stderr == "command timed out after 300 seconds"


def test_procurement_normalizes_all_github_transport_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hung repository, PR, and release snapshots fail closed independently."""
    module = _load_script("build_procurement_due_diligence.py")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=30)

    monkeypatch.setattr(module.subprocess, "run", _timeout)
    snapshot = module._github_snapshot("example/project", offline=False)

    for name in ("repo", "open_prs", "releases"):
        assert snapshot[name]["ok"] is False
        assert snapshot[name]["returncode"] == 124
        assert snapshot[name]["stderr"] == "GitHub API request timed out"
