"""Collector transport contracts; actual provider evidence is recorded separately."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts/collect_worker_artifact_readiness.py"


def _load_collector():
    """Load only the owned script without importing the numerical package."""
    spec = importlib.util.spec_from_file_location("_artifact_collector_under_test", COLLECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _ReportFixture:
    """Explicit stand-in for observing consumer projection, not file verification."""
    state: str
    reason_code: str
    scope: str = "artifact_files_only"
    artifact_policy_digest: str | None = None
    target_id: str | None = None
    source_commit: str | None = None
    archive_sha256: str | None = None
    observed_members_sha256: str | None = None
    worker_pid: int | None = None
    installation_epoch: str | None = None


def _provider(monkeypatch, callback):
    """Install a fixture-only provider seam without numerical initialization."""
    package = types.ModuleType("fast_mlsirm")
    package.__path__ = []
    provider = types.ModuleType("fast_mlsirm._artifact_readiness")
    provider.verify_artifact_readiness = callback
    monkeypatch.setitem(sys.modules, "fast_mlsirm", package)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._artifact_readiness", provider)


def _request():
    """Return distinct explicit forwarding sentinels; no actual startup trust."""
    return {
        "policy_json": '{"schema":"fixture-only"}',
        "pinned_policy_digest": "c" * 64,
        "installation_root": "/explicit-fixture-root",
        "startup_observation": {"fixture": "caller-owned"},
        "deadline_monotonic": 123.5,
    }


@pytest.mark.parametrize("state,reason,exit_code", [
    ("ready", "artifact_files_verified", 0),
    ("unknown", "startup_missing", 1),
    ("mismatch", "member_digest_mismatch", 1),
])
def test_cli_forwards_exact_inputs_and_keeps_provider_report_artifact_only(
    monkeypatch, capsys, state, reason, exit_code,
):
    """Observe forwarding and exit semantics with declared provider-return fixtures."""
    collector = _load_collector()
    request = _request()
    calls = []
    observation = _ReportFixture(
        state, reason, artifact_policy_digest="c" * 64,
        target_id="fixture", source_commit="a" * 40, archive_sha256="b" * 64,
        observed_members_sha256="d" * 64 if state == "ready" else None,
        worker_pid=4242, installation_epoch="fixture-epoch",
    )

    def provider(policy_json, **kwargs):
        calls.append((policy_json, kwargs))
        return observation

    _provider(monkeypatch, provider)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert collector.main() == exit_code
    assert calls == [(request["policy_json"], {k: v for k, v in request.items() if k != "policy_json"})]
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert set(report) == {"schema", "artifact"}
    assert report["schema"] == "worker_artifact_readiness_report/1.0"
    assert report["artifact"] == vars(observation)
    assert report["artifact"]["scope"] == "artifact_files_only"
    assert output.err == ""


@pytest.mark.parametrize("raw", [
    "{}", "{", '{"policy_json":null,"policy_json":null}',
    '{"deadline_monotonic":NaN}', "[" * 70 + "0" + "]" * 70,
    "x" * ((1 << 20) + 1),
])
def test_cli_invalid_input_never_reaches_provider(monkeypatch, capsys, raw):
    """Malformed requests must fail before the supplied provider can run."""
    collector = _load_collector()
    monkeypatch.setattr(collector, "collect_worker_artifact_readiness", lambda **kwargs: pytest.fail("invalid input dispatched"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    assert collector.main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "invalid artifact-readiness request\n"


def test_function_preserves_provider_invalid_record_exception(monkeypatch):
    """Library callers receive the original provider exception, not false ready."""
    collector = _load_collector()
    error = ValueError("fixture invalid-record refusal")

    def provider(*args, **kwargs):
        raise error

    _provider(monkeypatch, provider)
    with pytest.raises(ValueError) as caught:
        collector.collect_worker_artifact_readiness(**_request())
    assert caught.value is error


def test_cli_missing_provider_returns_unavailable_without_importing_candidate(monkeypatch, capsys):
    """The collector must not copy, install or manufacture its missing dependency."""
    collector = _load_collector()
    package = types.ModuleType("fast_mlsirm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "fast_mlsirm", package)
    monkeypatch.delitem(sys.modules, "fast_mlsirm._artifact_readiness", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_request())))
    assert collector.main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "approved artifact-readiness helper unavailable\n"
