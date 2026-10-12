# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Untrusted executor replies cannot partially populate bootstrap aggregates."""

from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
import hashlib
import json

import numpy as np
import pytest

import fast_mlsirm.bifactor_bootstrap as bb
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath, RemoteJobDeliveryState, RemoteJobOutcome,
    RemoteRunManifest, SEED_DERIVATION_RULE, envelope_fingerprint,
    local_worker_provenance, result_identity_sha256,
)


def _task(index=0, multigroup=False):
    return (index, np.zeros((4, 2)), np.array([0, 0]), 3, 1,
            np.array([0, 0, 1, 1]) if multigroup else None,
            2 if multigroup else 1, None, 7, 7, 2, 1e-3, 1, 0, False, "cpu")


def _record(index, multigroup=False):
    slopes = [[1., 1.], [1., 1.]] if multigroup else [1., 1.]
    threshold = [[[1., -1.], [1., -1.]]] * 2 if multigroup else [[1., -1.], [1., -1.]]
    return dict(family="bifactor_bootstrap_replicate", replicate_id=index,
                converged=True, rejected=False, error="", a_general=slopes,
                a_specific=deepcopy(slopes), threshold=threshold,
                general_mean=[0., 0.] if multigroup else [0.],
                general_sd=[1., 1.] if multigroup else [1.],
                specific_sd=[[1.], [1.]] if multigroup else [[1.]], loglik=-10.)


def _batch(mutate, *, multigroup=False):
    manifest = RemoteRunManifest("1.0", version("fast-mlsirm"), "b" * 64,
        SEED_DERIVATION_RULE, ExecutionFloatPath.F64, "a" * 64, "c" * 64)

    class Executor:
        def run_batch(self, envelopes, *, worker_manifest, payload):
            outcomes = []
            for envelope in envelopes:
                fingerprint = envelope_fingerprint(envelope)
                record = _record(envelope.unit_index, multigroup)
                outcomes.append(RemoteJobOutcome(
                    run_id=envelope.run_id, unit_index=envelope.unit_index,
                    unit_seed=envelope.unit_seed(), family=envelope.family,
                    delivery_state=RemoteJobDeliveryState.COMPLETED,
                    result=record, error_message=None,
                    provenance=local_worker_provenance(worker_manifest,
                        requested_device="cpu", effective_device="cpu", wall_clock_seconds=0.),
                    input_identity_sha256=fingerprint, output_identity_sha256=result_identity_sha256(record),
                    envelope_fingerprint=fingerprint, driver_host="test", driver_pid=1))
            return mutate(outcomes)

    return [_task(0, multigroup), _task(1, multigroup)], bb.RemoteBootstrapBackend(Executor(), manifest, "run"), manifest


def _run(mutate, *, multigroup=False):
    batch, backend, manifest = _batch(mutate, multigroup=multigroup)
    results = [None, None]
    bb._run_remote_batch(batch, backend, manifest, {}, 123, results)
    return results


def _change_record(outcomes, key, value):
    record = deepcopy(outcomes[1].result)
    record[key] = value
    # Independent permissive encoding models an older/malformed executor.
    # The production helper rejects nonfinite values before admission is reached.
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    outcomes[1] = replace(outcomes[1], result=record,
                          output_identity_sha256=hashlib.sha256(encoded).hexdigest())
    return outcomes


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[:-1],
    lambda rows: rows + [rows[0]],
    lambda rows: [rows[0], rows[0]],
    lambda rows: [rows[0], replace(rows[1], run_id="other-run")],
    lambda rows: [rows[0], replace(rows[1], unit_seed=0)],
    lambda rows: [rows[0], replace(rows[1], envelope_fingerprint="f" * 64)],
    lambda rows: [rows[0], replace(rows[1], output_identity_sha256="f" * 64)],
    lambda rows: [rows[0], replace(rows[1], input_identity_sha256="f" * 64)],
    lambda rows: [rows[0], replace(rows[1], unit_index=9)],
    lambda rows: [rows[0], replace(rows[1], provenance=replace(rows[1].provenance, library_version="0.0.0"))],
    lambda rows: _change_record(rows, "family", "mc_replicate"),
    lambda rows: _change_record(rows, "error", "x" * 513),
    lambda rows: _change_record(rows, "error", "unexpected error on successful fit"),
    lambda rows: _change_record(rows, "rejected", True),
    lambda rows: _change_record(rows, "a_general", [True, 1.]),
    lambda rows: _change_record(rows, "a_general", ["1", 1.]),
    lambda rows: _change_record(rows, "a_general", [float("inf"), 1.]),
    lambda rows: _change_record(rows, "replicate_id", 0),
    lambda rows: _change_record(rows, "replicate_id", True),
    lambda rows: _change_record(rows, "converged", "false"),
    lambda rows: _change_record(rows, "a_general", [1.]),
    lambda rows: _change_record(rows, "threshold", [[1., -1.]]),
    lambda rows: _change_record(rows, "general_sd", [None]),
    lambda rows: _change_record(rows, "loglik", [1.]),
])
def test_invalid_remote_batch_never_mutates_results(mutation):
    batch, backend, manifest = _batch(mutation)
    results = [None, None]
    with pytest.raises(RuntimeError, match="remote bootstrap"):
        bb._run_remote_batch(batch, backend, manifest, {}, 123, results)
    assert results == [None, None]


@pytest.mark.parametrize("marker", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reply_fixture_reaches_consumer_rejection(marker):
    """An older-worker fixture must reach admission, not the strict producer hash."""
    captured = []

    def malformed_reply(rows):
        changed = _change_record(rows, "a_general", [marker, 1.])
        captured.append(changed)
        return changed

    batch, backend, manifest = _batch(malformed_reply)
    results = [None, None]
    with pytest.raises(RuntimeError, match="remote bootstrap invalid output identity"):
        bb._run_remote_batch(batch, backend, manifest, {}, 123, results)
    assert len(captured) == 1
    assert captured[0][1].result["a_general"][0] is marker
    assert captured[0][0].result == _record(0)
    assert results == [None, None]


def test_valid_reordered_multigroup_batch_is_accepted():
    results = _run(lambda rows: rows[::-1], multigroup=True)
    assert [row[0] for row in results] == [0, 1]
    assert results[0][2].shape == (2, 2)


@pytest.mark.parametrize("error", [RuntimeError("core unavailable"), OSError("disk full"),
    MemoryError("allocation failed"), TypeError("bad adapter"), KeyError("missing parameter"),
    ValueError("invalid quadrature")])
def test_unexpected_fit_error_is_not_a_statistical_rejection(monkeypatch, error):
    def broken(**kwargs):
        raise error
    monkeypatch.setattr(bb, "fit_bifactor_grm", broken)
    with pytest.raises(type(error)):
        bb._fit_single_replicate(*_task())


@pytest.mark.parametrize("multigroup,message", [
    (False, "item 0 category 2 is never observed (unidentified GRM boundary); every declared category must be observed"),
    (False, "item 0 has no observed responses"),
    (True, "free item 0 category 2 is never observed in group 1 (unidentified per-group GRM boundary)"),
    (True, "free item 0 has no observed responses in group 1"),
])
def test_documented_resample_rejection_stays_a_result(monkeypatch, multigroup, message):
    def rejected(**kwargs):
        raise ValueError(message)
    monkeypatch.setattr(bb, "fit_bifactor_grm_multigroup" if multigroup else "fit_bifactor_grm", rejected)
    result = bb._fit_single_replicate(*_task(multigroup=multigroup))
    assert result[1] is False and result[10] is True
    assert result[9] == f"ValueError: {message}"


def test_worker_serializes_infrastructure_error_as_failed_not_rejected(monkeypatch, capsys):
    import io
    import json
    import fast_mlsirm.remote_worker as worker
    from fast_mlsirm.remote_exec import RemoteJobEnvelope, RemoteJobFamily, payload_identity_sha256

    def broken(**kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(bb, "fit_bifactor_grm", broken)
    task = _task()
    payload = bb.bootstrap_replicate_payload(*task[1:13], task[14], task[15])
    _, _, manifest = _batch(lambda rows: rows)
    manifest = replace(manifest, payload_sha256=payload_identity_sha256(payload))
    envelope = RemoteJobEnvelope("run", RemoteJobFamily.BIFACTOR_BOOTSTRAP_REPLICATE,
                                 0, 123, manifest.payload_sha256, manifest)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(
        {"envelope": envelope.to_dict(), "payload": payload})))
    assert worker.main() == 0  # structured FAILED reply, not a broken protocol
    reply = json.loads(capsys.readouterr().out)
    assert reply["delivery_state"] == RemoteJobDeliveryState.FAILED.value
    assert reply["error_message"] == "storage unavailable"
    assert "result" not in reply


def test_multigroup_rejected_reply_has_matching_null_shapes():
    def reject(rows):
        record = rows[1].result
        record.update(converged=False, rejected=True,
            error="ValueError: free item 0 has no observed responses in group 1")
        for key in ("a_general", "a_specific"):
            record[key] = [[None, None], [None, None]]
        record["threshold"] = [[[None, None], [None, None]]] * 2
        record["general_mean"] = record["general_sd"] = [None, None]
        record["specific_sd"] = [[None], [None]]
        record["loglik"] = None
        rows[1] = replace(rows[1], output_identity_sha256=result_identity_sha256(record))
        return rows
    results = _run(reject, multigroup=True)
    assert results[1][10] is True
    assert results[1][2].shape == (2, 2)
    assert np.isnan(results[1][2]).all()
