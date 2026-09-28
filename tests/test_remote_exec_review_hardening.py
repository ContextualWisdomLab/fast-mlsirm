# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fail-closed contracts from the #2072 review threads on L4 subprocess execution."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest
from importlib.metadata import PackageNotFoundError, version

try:
    LIBRARY_VERSION = version("fast-mlsirm")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.11.4"

import fast_mlsirm.remote_exec as remote_exec
import fast_mlsirm.remote_worker as remote_worker
from fast_mlsirm import FitConfig, fit
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath,
    LoopbackExecutor,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteRunManifest,
    SEED_DERIVATION_RULE,
    SubprocessExecutor,
    payload_identity_sha256,
    result_identity_sha256,
)

_SHA_B = "b" * 64
_SHA_C = "c" * 64
_MC_PAYLOAD = {
    "config": {"n_persons": 10, "n_dims": 1, "items_per_dim": 3, "latent_dim": 1, "gamma": 1.0}
}


def _manifest(payload: dict[str, object]) -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256=_SHA_B,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_identity_sha256(payload),
        integration_nodes_sha256=_SHA_C,
    )


def _envelope(
    family: RemoteJobFamily,
    payload: dict[str, object],
    *,
    unit_index: int = 0,
    run_id: str = "review_hardening",
) -> RemoteJobEnvelope:
    manifest = _manifest(payload)
    return RemoteJobEnvelope(
        run_id=run_id,
        family=family,
        unit_index=unit_index,
        base_seed=20260929,
        payload_ref=manifest.payload_sha256,
        manifest=manifest,
    )


def _fake_worker(stdout: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _completed_payload(result: object, output_identity: str) -> str:
    return json.dumps(
        {
            "delivery_state": RemoteJobDeliveryState.COMPLETED.value,
            "result": result,
            "output_identity_sha256": output_identity,
            "worker_pid": 4242,
            "hostname": "remote-worker",
            "architecture": "arm64",
            "operating_system": "Darwin",
            "library_version": LIBRARY_VERSION,
            "wall_clock_seconds": 0.01,
        }
    )


def _executable(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake_interpreter"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_mc_replicate_requires_payload_before_dispatch() -> None:
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    with pytest.raises(ValueError, match="payload is required"):
        SubprocessExecutor(socket.gethostname()).run_batch(
            (envelope,), worker_manifest=envelope.manifest
        )


def test_mc_replicate_worker_rejects_mismatched_payload() -> None:
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    other = {"config": {**_MC_PAYLOAD["config"], "n_persons": 11}}
    with pytest.raises(ValueError, match="payload identity"):
        remote_worker.execute_envelope(envelope, other)


def test_mc_replicate_simulates_the_manifested_payload() -> None:
    pytest.importorskip("fast_mlsirm._core")
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    result = remote_worker.execute_envelope(envelope, _MC_PAYLOAD)
    assert (result["n_persons"], result["n_items"]) == (10, 3)


def test_supplied_output_identity_must_hash_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    forged = result_identity_sha256({"objective": 11.0})
    monkeypatch.setattr(
        remote_exec,
        "_invoke_worker_process",
        lambda *a, **k: _fake_worker(_completed_payload({"objective": 12.0}, forged)),
    )
    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,), worker_manifest=envelope.manifest, payload=_MC_PAYLOAD
    )[0]
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "output_identity_sha256" in (outcome.error_message or "")


def test_outcomes_have_a_total_order_across_runs() -> None:
    payload = {"x": 1}
    envelopes = (
        _envelope(RemoteJobFamily.MC_REPLICATE, payload, run_id="run_b"),
        _envelope(RemoteJobFamily.MC_REPLICATE, payload, run_id="run_a"),
    )
    outcomes = LoopbackExecutor().run_batch(
        envelopes,
        lambda envelope, _seed: {"run": envelope.run_id},
        worker_manifest=envelopes[0].manifest,
    )
    assert [outcome.run_id for outcome in outcomes] == ["run_a", "run_b"]


def test_stalled_worker_times_out_as_failed_outcome(tmp_path: Path) -> None:
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    executor = SubprocessExecutor(
        socket.gethostname(),
        remote_interpreter=_executable(tmp_path, "sleep 30"),
        timeout_seconds=0.5,
    )
    outcome = executor.run_batch(
        (envelope,), worker_manifest=envelope.manifest, payload=_MC_PAYLOAD
    )[0]
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "timed out" in (outcome.error_message or "")


def test_noisy_worker_output_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SubprocessExecutor, "_MAX_WORKER_STDOUT_BYTES", 1024)
    envelope = _envelope(RemoteJobFamily.MC_REPLICATE, _MC_PAYLOAD)
    executor = SubprocessExecutor(
        socket.gethostname(),
        remote_interpreter=_executable(tmp_path, "yes | head -c 50000000"),
    )
    outcome = executor.run_batch(
        (envelope,), worker_manifest=envelope.manifest, payload=_MC_PAYLOAD
    )[0]
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "exceeded bound" in (outcome.error_message or "")


def test_ssh_remote_command_is_shell_quoted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdin = io.BytesIO()
        stdout = io.BytesIO(b"")
        stderr = io.BytesIO(b"")

        def __init__(self, argv, **kwargs) -> None:
            captured.append(argv)

        def wait(self, timeout=None) -> int:
            return 0

        def kill(self) -> None:
            pass

    monkeypatch.setattr(remote_exec.subprocess, "Popen", _Proc)
    remote_exec._invoke_worker_process(
        "{}",
        worker_host="user@host",
        remote_interpreter="/opt/py 3/python;touch /tmp/pwned",
        stdout_limit=1024,
        timeout_seconds=5.0,
    )
    argv = captured[0]
    remote_command = argv[argv.index("--") + 2 :]
    assert remote_command == [
        "'/opt/py 3/python;touch /tmp/pwned' -m fast_mlsirm.remote_worker"
    ]


def test_worker_rejects_oversized_request(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(remote_worker, "_MAX_REQUEST_CHARS", 16)
    monkeypatch.setattr(remote_worker.sys, "stdin", io.StringIO("x" * 17))
    assert remote_worker.main() == 1
    response = json.loads(capsys.readouterr().out)
    assert response["delivery_state"] == RemoteJobDeliveryState.FAILED.value
    assert "exceeds" in response["error_message"]


def _fit_payload(responses: list[list[float]], *, n_restarts: int = 1) -> dict[str, object]:
    return {
        "responses": responses,
        "factor_id": [0, 0, 0],
        "config": {
            "model": "MLS2PLM",
            "latent_dim": 1,
            "optimizer": "adam",
            "max_iter": 3,
            "n_restarts": n_restarts,
            "backend": "rust",
            "rust_device": "cpu",
        },
    }


@pytest.mark.parametrize("marker", [-1, None])
def test_fit_restart_preserves_missing_responses(marker: object) -> None:
    """Canonical JSON has no NaN, so payloads mark missing cells with -1 or null."""
    pytest.importorskip("fast_mlsirm._core")
    rows = [[1, marker, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1]]
    payload = _fit_payload(rows)
    envelope = _envelope(RemoteJobFamily.FIT_RESTART, payload)
    remote = remote_worker.execute_envelope(envelope, payload)

    local = fit(
        np.asarray(rows, dtype=np.float64),
        np.zeros(3, dtype=np.int64),
        FitConfig(**{**payload["config"], "seed": envelope.unit_seed()}),
    )
    assert remote["objective"] == float(local.objective)


def test_fit_restart_unit_is_one_independently_seeded_restart() -> None:
    pytest.importorskip("fast_mlsirm._core")
    rows = [[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1]]
    payload = _fit_payload(rows, n_restarts=3)
    envelope = _envelope(RemoteJobFamily.FIT_RESTART, payload)
    remote = remote_worker.execute_envelope(envelope, payload)

    single = fit(
        np.asarray(rows, dtype=np.float64),
        np.zeros(3, dtype=np.int64),
        FitConfig(**{**payload["config"], "n_restarts": 1, "seed": envelope.unit_seed()}),
    )
    assert remote["objective"] == float(single.objective)


def test_payload_identity_is_sha256_of_canonical_json() -> None:
    encoded = json.dumps(_MC_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    assert payload_identity_sha256(_MC_PAYLOAD) == hashlib.sha256(encoded).hexdigest()
