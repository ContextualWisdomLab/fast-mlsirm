"""Two-phase disposable-daemon evidence, not installed-wheel/remote-device evidence."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError("run without -O/PYTHONOPTIMIZE so evidence assertions remain active")

import redis

from fast_mlsirm.remote_exec import (
    RemoteJobDeliveryState, RemoteJobEnvelope,
    ValkeyStreamsOutcomeStore, envelope_fingerprint, result_identity_sha256,
)
from fast_mlsirm.remote_worker import ValkeyStreamsWorker, execute_envelope
from test_remote_exec_valkey import _MC_PAYLOAD, _envelope


class FaultClient:
    def __init__(self, client, jobs, outcomes, fault):
        self.client, self.jobs, self.outcomes, self.fault = client, jobs, outcomes, fault

    def __getattr__(self, name):
        return getattr(self.client, name)

    def xadd(self, name, fields):
        if name == self.outcomes and self.fault == "publish":
            raise ConnectionError("injected disconnect before outcome append")
        return self.client.xadd(name, fields)

    def xack(self, name, group, *ids):
        if name == self.jobs and self.fault == "ack":
            raise ConnectionError("injected disconnect before job ACK")
        return self.client.xack(name, group, *ids)


def worker(client, jobs, outcomes, manifest, *, consumer, executor=None):
    return ValkeyStreamsWorker(
        client, jobs_stream=jobs, outcomes_stream=outcomes, group="workers",
        consumer=consumer, worker_manifest=manifest, executor=executor,
        min_idle_ms=0, batch_size=1, block_ms=0,
    )


def store(client, outcomes):
    return ValkeyStreamsOutcomeStore(
        client, stream=outcomes, group="drivers", consumer="probe",
        min_idle_ms=0, batch_size=1, block_ms=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    client = redis.Redis.from_url(
        f"redis://127.0.0.1:{args.port}/0", decode_responses=True,
        socket_timeout=3, socket_connect_timeout=3,
    )
    assert client.ping()
    assert client.config_get("appendonly")["appendonly"] == "yes"
    assert client.config_get("appendfsync")["appendfsync"] == "always"
    server_id = client.info("server")["run_id"]
    if args.phase == "before":
        suffix = uuid.uuid4().hex
        records = []
        for index, fault in enumerate(("ack", "publish")):
            envelope = _envelope(unit_index=index)
            jobs = f"fast-mlsirm:worker-restart:{suffix}:jobs:{index}"
            outcomes = f"fast-mlsirm:worker-restart:{suffix}:outcomes:{index}"
            expected_digest = result_identity_sha256(execute_envelope(envelope, _MC_PAYLOAD))
            client.xadd(jobs, {
                "fingerprint": envelope_fingerprint(envelope),
                "envelope": json.dumps(envelope.to_dict()),
                "payload": json.dumps(_MC_PAYLOAD),
                "requested_device": "cpu", "effective_device": "gpu",
                "driver_host": socket.gethostname(), "driver_pid": str(os.getpid()),
            })
            try:
                worker(FaultClient(client, jobs, outcomes, fault), jobs, outcomes,
                       envelope.manifest, consumer="crashed").run_once()
            except ConnectionError:
                pass
            else:
                raise AssertionError("fault was not observed")
            assert client.xpending(jobs, "workers")["pending"] == 1
            fingerprint = envelope_fingerprint(envelope)
            if fault == "ack":
                outcome = store(client, outcomes).committed_success(fingerprint)
                assert outcome is not None
                assert outcome.output_identity_sha256 == expected_digest
                assert outcome.provenance.effective_device == "unknown"
            else:
                assert client.xlen(outcomes) == 0
            records.append({
                "jobs": jobs, "outcomes": outcomes, "envelope": envelope.to_dict(),
                "fingerprint": fingerprint, "expected_digest": expected_digest,
                "fault": fault,
            })
        with args.state.open("x") as state_file:
            state_file.write(json.dumps({
                "server_id": server_id, "suffix": suffix, "records": records,
            }))
        print(json.dumps({"phase": "before", "pending_jobs": 2,
                          "committed_successes": 1, "appendfsync": "always"}))
        return

    state = json.loads(args.state.read_text())
    assert server_id != state["server_id"], "daemon did not restart"
    calls = []
    for record in state["records"]:
        jobs, outcomes = record["jobs"], record["outcomes"]
        prefix = f"fast-mlsirm:worker-restart:{state['suffix']}:"
        assert jobs.startswith(prefix) and outcomes.startswith(prefix)
        envelope = RemoteJobEnvelope.from_dict(record["envelope"])
        assert client.xpending(jobs, "workers")["pending"] == 1

        def execute(e, payload):
            calls.append(e.unit_index)
            return execute_envelope(e, payload)

        tally = worker(client, jobs, outcomes, envelope.manifest,
                       consumer="recovered", executor=execute).run_once()
        assert tally.skipped == int(record["fault"] == "ack")
        assert tally.completed == int(record["fault"] == "publish")
        assert client.xpending(jobs, "workers")["pending"] == 0
        outcome = store(client, outcomes).committed_success(record["fingerprint"])
        assert outcome is not None
        assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
        assert outcome.input_identity_sha256 == record["fingerprint"]
        assert outcome.unit_seed == envelope.unit_seed()
        assert outcome.output_identity_sha256 == record["expected_digest"]
        assert outcome.output_identity_sha256 == result_identity_sha256(outcome.result)
        assert outcome.provenance.effective_device == "unknown"
        assert store(client, outcomes).successful_count(record["fingerprint"]) == 1
        client.delete(jobs, outcomes, f"{outcomes}:committed", f"{outcomes}:failed")
    assert calls == [1], calls
    print(json.dumps({"phase": "after", "daemon_restart_verified": True,
                      "pending_recovered": 2, "executed": 1, "skipped": 1,
                      "fingerprint_seed_result_digest": "PASS",
                      "scope": "existing-editable-local-numerics, remote-disposable-daemon"}))


if __name__ == "__main__":
    main()
