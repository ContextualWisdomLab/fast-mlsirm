"""Regression contracts for bounded JSON emitted by external commands."""

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

    module_name = f"bounded_output_{Path(filename).stem}"
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
def test_workflow_registry_normalizes_bounded_json_failures(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Registry API output limits become stable GitHub API evidence."""

    module = _load_script("audit_workflow_registry.py")
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

    with pytest.raises(module.GitHubApiError) as captured:
        module._run_gh_api(
            "/repos/example/project/actions/workflows",
            max_attempts=1,
            retry_sleep_seconds=0,
        )

    assert captured.value.stderr == "GitHub API returned invalid JSON"
    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == bounded_failure


@pytest.mark.parametrize("bounded_failure", _BOUNDED_FAILURES)
def test_pr_queue_treats_bounded_json_failures_as_unavailable_payloads(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Queue governance cannot consume oversized or over-nested command JSON."""

    module = _load_script("build_pr_queue_governance.py")
    monkeypatch.setattr(
        module,
        "parse_json_bounded",
        lambda _content: (_ for _ in ()).throw(ValueError(bounded_failure)),
    )

    assert module._json_from_completed(_successful_command()) is None


@pytest.mark.parametrize("bounded_failure", _BOUNDED_FAILURES)
def test_procurement_snapshot_fails_closed_on_bounded_json_errors(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Procurement evidence cannot silently omit a bounded GitHub payload failure."""

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


@pytest.mark.parametrize("bounded_failure", _BOUNDED_FAILURES)
def test_ignored_rust_shard_normalizes_bounded_metadata_failures(
    monkeypatch: pytest.MonkeyPatch,
    bounded_failure: str,
) -> None:
    """Cargo metadata limits remain a stable shard-runner contract error."""

    module = _load_script("run_ignored_rust_shard.py")
    monkeypatch.setattr(
        module,
        "parse_json_bounded",
        lambda _content: (_ for _ in ()).throw(ValueError(bounded_failure)),
    )

    with pytest.raises(ValueError, match="Cargo metadata output is not valid JSON") as captured:
        module.parse_workspace_targets("{}")

    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == bounded_failure
