"""Valkey Streams contracts for durable remote outcome delivery."""

from __future__ import annotations

import json
import os
import uuid

import pytest

from fast_mlsirm.remote_exec import (
    RemoteJobDeliveryState,
    RemoteJobOutcome,
    ValkeyStreamsBackend,
    ValkeyStreamsOutcomeStore,
    envelope_fingerprint,
)

from test_remote_exec import _envelope, _manifest


class FakeValkey:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.pending: list[tuple[str, dict[str, str]]] = []
        self.acked: list[str] = []
        self.hashes: dict[str, dict[str, str]] = {}
        self.commands: list[str] = []

    def xgroup_create(self, name, groupname, id="0", mkstream=False):
        self.commands.append("XGROUP CREATE")

    def xadd(self, name, fields):
        entries = self.streams.setdefault(name, [])
        record_id = f"{len(entries) + len(self.pending) + 1}-0"
        entries.append((record_id, dict(fields)))
        return record_id

    def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id, count):
        self.commands.append("XAUTOCLAIM")
        claimed, self.pending = self.pending[:count], self.pending[count:]
        next_id = claimed[-1][0] if claimed else start_id
        return (next_id, claimed, [])

    def xreadgroup(self, groupname, consumername, streams, count, block):
        self.commands.append("XREADGROUP")
        name = next(iter(streams))
        entries = self.streams.get(name, [])[:count]
        self.streams[name] = self.streams.get(name, [])[len(entries) :]
        return [] if not entries else [(name, entries)]

    def xack(self, name, groupname, *ids):
        self.commands.append("XACK")
        self.acked.extend(ids)
        return len(ids)

    def hsetnx(self, name, key, value):
        bucket = self.hashes.setdefault(name, {})
        if key in bucket:
            return 0
        bucket[key] = value
        return 1

    def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))


def _completed_outcome(unit_index: int, result: object) -> RemoteJobOutcome:
    envelope = _envelope(unit_index=unit_index)
    from fast_mlsirm.remote_exec import LoopbackExecutor

    return LoopbackExecutor().run_batch(
        (envelope,), lambda _envelope, _seed: result, worker_manifest=envelope.manifest
    )[0]


def _record(outcome: RemoteJobOutcome) -> dict[str, str]:
    return {
        "fingerprint": outcome.envelope_fingerprint,
        "outcome": json.dumps(outcome.to_dict(), sort_keys=True, separators=(",", ":")),
    }


def test_valkey_store_reclaims_reads_acks_and_keeps_first_success() -> None:
    client = FakeValkey()
    first = _completed_outcome(1, {"attempt": 1})
    last = _completed_outcome(1, {"attempt": 2})
    client.pending.append(("1-0", _record(first)))
    client.streams["outcomes"] = [("2-0", _record(last))]
    store = ValkeyStreamsOutcomeStore(client, stream="outcomes", group="drivers", consumer="d1")

    assert store.committed_success(last.envelope_fingerprint) == first
    assert store.successful_count(last.envelope_fingerprint) == 1
    assert client.commands.count("XAUTOCLAIM") >= 1
    assert "XREADGROUP" in client.commands
    assert client.acked == ["1-0", "2-0"]


def test_valkey_backend_publishes_envelope_and_consumes_completed_outcome() -> None:
    class WorkerValkey(FakeValkey):
        def xadd(self, name, fields):
            record_id = super().xadd(name, fields)
            if name == "jobs":
                outcome = _completed_outcome(4, {"worker": "valkey"})
                self.streams["outcomes"] = [("2-0", _record(outcome))]
            return record_id

    client = WorkerValkey()
    envelope = _envelope(unit_index=4)
    backend = ValkeyStreamsBackend(
        client,
        jobs_stream="jobs",
        outcomes_stream="outcomes",
        group="drivers",
        consumer="d1",
    )

    outcome = backend.run_batch((envelope,), worker_manifest=_manifest())[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.result == {"worker": "valkey"}
    job = next(fields for _id, fields in client.streams["jobs"] if "envelope" in fields)
    assert job["fingerprint"] == envelope_fingerprint(envelope)


def test_valkey_store_restart_recovery_and_first_success_commit() -> None:
    client = FakeValkey()
    first = _completed_outcome(11, {"attempt": 1})
    second = _completed_outcome(11, {"attempt": 2})
    fingerprint = first.envelope_fingerprint
    store_a = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="a", block_ms=0
    )
    store_b = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="b", block_ms=0
    )
    winner_a = store_a.commit_success(fingerprint, first)
    winner_b = store_b.commit_success(fingerprint, second)
    assert winner_a == winner_b == first
    assert store_b.committed_success(fingerprint) == first


def test_valkey_backend_publishes_device_fields_and_waits_for_delayed_outcomes() -> None:
    class DelayedWorkerValkey(FakeValkey):
        def __init__(self) -> None:
            super().__init__()
            self.published = 0

        def xadd(self, name, fields):
            record_id = super().xadd(name, fields)
            if name == "jobs":
                self.published += 1
                if self.published == 2:
                    for unit_index in (5, 6):
                        outcome = _completed_outcome(unit_index, {"worker": unit_index})
                        self.streams.setdefault("outcomes", []).append(
                            ("2-0", _record(outcome))
                        )
            return record_id

    client = DelayedWorkerValkey()
    envelopes = (_envelope(unit_index=5), _envelope(unit_index=6))
    backend = ValkeyStreamsBackend(
        client,
        jobs_stream="jobs",
        outcomes_stream="outcomes",
        group="drivers",
        consumer="d1",
        block_ms=0,
        wait_timeout_s=1.0,
    )

    outcomes = backend.run_batch(
        envelopes,
        worker_manifest=_manifest(),
        requested_device="gpu",
        effective_device="cpu",
    )

    assert len(outcomes) == 2
    job_fields = [fields for _id, fields in client.streams["jobs"]]
    assert all(job["requested_device"] == "gpu" for job in job_fields)
    assert all(job["effective_device"] == "cpu" for job in job_fields)


def test_valkey_store_against_host_daemon() -> None:
    """Run only when the host explicitly supplies a disposable Valkey endpoint."""
    url = os.environ.get("FAST_MLSIRM_VALKEY_URL")
    if not url:
        pytest.skip("FAST_MLSIRM_VALKEY_URL is not configured")
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(url, decode_responses=True)
    suffix = uuid.uuid4().hex
    stream = f"fast-mlsirm:test:outcomes:{suffix}"
    group = f"fast-mlsirm-test-{suffix}"
    try:
        outcome = _completed_outcome(9, {"daemon": True})
        store = ValkeyStreamsOutcomeStore(client, stream=stream, group=group, consumer="driver")
        store.commit_success(outcome.envelope_fingerprint, outcome)
        restarted = ValkeyStreamsOutcomeStore(
            client, stream=stream, group=group, consumer="driver-restart", block_ms=100
        )
        assert restarted.committed_success(outcome.envelope_fingerprint) == outcome
    finally:
        client.delete(stream, f"{stream}:committed")
