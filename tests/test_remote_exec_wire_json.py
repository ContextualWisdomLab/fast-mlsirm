# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Finite result-wire admission; not estimator or convergence certification.

Basis: Bray (2017, Section 6, p. 7; Section 10, p. 10) excludes nonfinite
number literals from generated JSON. Python Software Foundation (n.d.,
"Infinite and NaN Number Values") documents the permissive default encoder.

References:
    Bray, T. (Ed.). (2017). The JavaScript Object Notation (JSON) data
        interchange format (RFC 8259). Internet Engineering Task Force.
    Python Software Foundation. (n.d.). json—JSON encoder and decoder.
        Python 3.14 documentation.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import socket
import subprocess

import pytest

import fast_mlsirm.remote_exec as remote_exec

_PAYLOAD = {"config": {"n_persons": 10, "n_dims": 1, "items_per_dim": 3}}


def _envelope(index: int) -> remote_exec.RemoteJobEnvelope:
    manifest = remote_exec.RemoteRunManifest(
        schema_version="1.0", library_version=version("fast-mlsirm"),
        source_sha256="b" * 64,
        seed_derivation_rule=remote_exec.SEED_DERIVATION_RULE,
        float_path=remote_exec.ExecutionFloatPath.F64,
        payload_sha256=remote_exec.payload_identity_sha256(_PAYLOAD),
        integration_nodes_sha256="c" * 64,
    )
    return remote_exec.RemoteJobEnvelope(
        run_id="finite_wire_contract", family=remote_exec.RemoteJobFamily.MC_REPLICATE,
        unit_index=index, base_seed=20260920, payload_ref=manifest.payload_sha256,
        manifest=manifest,
    )


def _worker_reply(result: object) -> subprocess.CompletedProcess[str]:
    # Independent permissive encoding models a malformed/older worker, not
    # the production hash helper being tested.
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    reply = {
        "delivery_state": "completed", "result": result,
        "output_identity_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "worker_pid": 4242, "hostname": "wire-worker", "architecture": "arm64",
        "operating_system": "Darwin", "library_version": version("fast-mlsirm"),
        "wall_clock_seconds": 0.01,
    }
    return subprocess.CompletedProcess([], 0, json.dumps(reply), "")


@pytest.mark.parametrize("marker", [float("nan"), float("inf"), -float("inf")])
def test_invalid_wire_result_does_not_commit_or_abort_valid_sibling(marker, monkeypatch):
    envelopes = (_envelope(0), _envelope(1))
    replies = iter((_worker_reply({"nested": [marker]}), _worker_reply({"value": 1.25})))
    monkeypatch.setattr(remote_exec, "_invoke_worker_process", lambda *a, **k: next(replies))
    ledger = remote_exec.OutcomeCommitLedger()
    outcomes = remote_exec.SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
        envelopes, worker_manifest=envelopes[0].manifest, payload=_PAYLOAD,
    )
    failed, valid = outcomes
    assert failed.delivery_state is remote_exec.RemoteJobDeliveryState.FAILED
    assert failed.result is None
    assert failed.output_identity_sha256 is None
    assert "JSON" in failed.error_message
    assert ledger.successful_count(remote_exec.envelope_fingerprint(envelopes[0])) == 0
    assert valid.delivery_state is remote_exec.RemoteJobDeliveryState.COMPLETED
    assert valid.result == {"value": 1.25}
    assert ledger.successful_count(remote_exec.envelope_fingerprint(envelopes[1])) == 1
    json.dumps([outcome.to_dict() for outcome in outcomes], allow_nan=False)


@pytest.mark.parametrize("marker", [float("nan"), float("inf"), -float("inf")])
def test_worker_serializes_nonfinite_result_as_failure(marker, monkeypatch, capsys):
    """Invalid output becomes a failure reply, not nonfinite JSON or an escape."""
    import io
    import fast_mlsirm.remote_worker as worker

    envelope = _envelope(0)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps({
        "envelope": envelope.to_dict(), "payload": _PAYLOAD,
    })))
    monkeypatch.setattr(worker, "execute_envelope", lambda *a, **k: {"value": marker})
    assert worker.main() == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["delivery_state"] == "failed"
    assert "error_message" in reply
    assert "result" not in reply
    assert "output_identity_sha256" not in reply
    json.dumps(reply, allow_nan=False)


def test_worker_preserves_finite_result_and_digest(monkeypatch, capsys):
    """Finite producer output keeps its canonical digest and completion state."""
    import io
    import fast_mlsirm.remote_worker as worker

    envelope = _envelope(0)
    result = {"text": "한국어", "value": 1.25, "optional": None}
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps({
        "envelope": envelope.to_dict(), "payload": _PAYLOAD,
    })))
    monkeypatch.setattr(worker, "execute_envelope", lambda *a, **k: result)
    assert worker.main() == 0
    reply = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256(json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert reply["delivery_state"] == "completed"
    assert reply["result"] == result
    assert reply["output_identity_sha256"] == expected
    json.dumps(reply, allow_nan=False)


@pytest.mark.parametrize("marker", [float("nan"), float("inf"), -float("inf")])
def test_loopback_rejects_invalid_result_without_aborting_sibling(marker):
    """Shared strict hashing must leave valid loopback sibling outcomes intact."""
    envelopes = (_envelope(0), _envelope(1))
    outcomes = remote_exec.LoopbackExecutor().run_batch(
        envelopes, lambda envelope, seed: {"value": marker if envelope.unit_index == 0 else 1.25},
        worker_manifest=envelopes[0].manifest,
    )
    failed, valid = outcomes
    assert failed.delivery_state is remote_exec.RemoteJobDeliveryState.FAILED
    assert failed.result is None
    assert failed.output_identity_sha256 is None
    assert valid.delivery_state is remote_exec.RemoteJobDeliveryState.COMPLETED
    assert valid.result == {"value": 1.25}
    json.dumps([outcome.to_dict() for outcome in outcomes], allow_nan=False)


@pytest.mark.parametrize("marker", [float("nan"), float("inf"), -float("inf")])
def test_result_identity_rejects_nonfinite_numbers(marker):
    """The public output digest helper does not hash non-JSON numbers."""
    with pytest.raises(ValueError, match="JSON"):
        remote_exec.result_identity_sha256({"nested": [marker]})
