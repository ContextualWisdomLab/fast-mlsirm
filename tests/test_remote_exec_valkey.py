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
        claimed, self.pending = self.pending, []
        return ("0-0", claimed, [])

    def xreadgroup(self, groupname, consumername, streams, count, block):
        self.commands.append("XREADGROUP")
        name = next(iter(streams))
        entries, self.streams[name] = self.streams.get(name, []), []
        return [] if not entries else [(name, entries)]

    def xack(self, name, groupname, *ids):
        self.commands.append("XACK")
        self.acked.extend(ids)
        return len(ids)


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


def test_valkey_store_reclaims_reads_acks_and_keeps_last_record() -> None:
    client = FakeValkey()
    first = _completed_outcome(1, {"attempt": 1})
    last = _completed_outcome(1, {"attempt": 2})
    client.pending.append(("1-0", _record(first)))
    client.streams["outcomes"] = [("2-0", _record(last))]
    store = ValkeyStreamsOutcomeStore(client, stream="outcomes", group="drivers", consumer="d1")

    assert store.committed_success(last.envelope_fingerprint) == last
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
        assert store.committed_success(outcome.envelope_fingerprint) == outcome
    finally:
        client.delete(stream)
