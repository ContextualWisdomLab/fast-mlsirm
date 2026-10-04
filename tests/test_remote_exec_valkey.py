"""Valkey Streams contracts for durable remote outcome delivery."""

from __future__ import annotations

import json
import time

import pytest

from fast_mlsirm.remote_exec import (
    CohortMismatchError,
    LoopbackExecutor,
    RemoteJobDeliveryState,
    RemoteJobOutcome,
    ValkeyStreamsBackend,
    ValkeyStreamsOutcomeStore,
    envelope_fingerprint,
)

from test_remote_exec import _MC_PAYLOAD, _payload_manifest
from test_remote_exec import _envelope as _base_envelope

_PAYLOAD_MANIFEST = _payload_manifest(_MC_PAYLOAD)


def _manifest():
    return _PAYLOAD_MANIFEST


def _envelope(**kwargs):
    kwargs.setdefault("manifest", _PAYLOAD_MANIFEST)
    return _base_envelope(**kwargs)


class FakeValkey:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.pending: list[tuple[str, dict[str, str]]] = []
        self.acked: list[str] = []
        self.hashes: dict[str, dict[str, str]] = {}
        self.commands: list[str] = []
        self.xreadgroup_blocks: list[int | None] = []
        # Optional scripted XAUTOCLAIM replies: (next_id, claimed_records).
        self.xautoclaim_script: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []

    def xgroup_create(self, name, groupname, id="0", mkstream=False):
        self.commands.append("XGROUP CREATE")

    def xadd(self, name, fields):
        entries = self.streams.setdefault(name, [])
        record_id = f"{len(entries) + len(self.pending) + 1}-0"
        entries.append((record_id, dict(fields)))
        return record_id

    def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id, count):
        self.commands.append("XAUTOCLAIM")
        if self.xautoclaim_script:
            next_id, claimed = self.xautoclaim_script.pop(0)
            return (next_id, claimed, [])
        claimed, self.pending = self.pending[:count], self.pending[count:]
        next_id = "0-0" if not self.pending else (claimed[-1][0] if claimed else start_id)
        return (next_id, claimed, [])

    def xreadgroup(self, groupname, consumername, streams, count, block=None):
        self.commands.append("XREADGROUP")
        self.xreadgroup_blocks.append(block)
        if block == 0:
            raise AssertionError(
                "BLOCK 0 is infinite wait on Redis/Valkey; omit BLOCK instead"
            )
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
    # Follow-up reads must omit BLOCK (None), never pass BLOCK 0.
    assert None in client.xreadgroup_blocks
    assert 0 not in client.xreadgroup_blocks


def test_valkey_store_continues_xautoclaim_until_cursor_zero() -> None:
    client = FakeValkey()
    early = _completed_outcome(21, {"phase": "ineligible-skip"})
    late = _completed_outcome(22, {"phase": "eligible"})
    client.xautoclaim_script = [
        ("1-0", []),
        ("0-0", [("2-0", _record(late))]),
    ]
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", batch_size=1, block_ms=0
    )

    assert store.committed_success(late.envelope_fingerprint) == late
    assert store.committed_success(early.envelope_fingerprint) is None
    assert client.commands.count("XAUTOCLAIM") >= 2
    assert client.acked == ["2-0"]


def test_valkey_store_drains_multi_batch_stream_without_block_zero() -> None:
    client = FakeValkey()
    outcomes = [_completed_outcome(i, {"idx": i}) for i in range(30, 33)]
    client.streams["outcomes"] = [
        (f"{i}-0", _record(outcome)) for i, outcome in enumerate(outcomes, start=1)
    ]
    store = ValkeyStreamsOutcomeStore(
        client,
        stream="outcomes",
        group="drivers",
        consumer="d1",
        batch_size=1,
        block_ms=50,
    )

    available = store.consume_available()

    assert set(available) == {outcome.envelope_fingerprint for outcome in outcomes}
    assert client.xreadgroup_blocks[0] == 50
    assert client.xreadgroup_blocks[1:] == [None] * (len(client.xreadgroup_blocks) - 1)
    assert 0 not in client.xreadgroup_blocks


def test_valkey_wait_for_terminal_empty_stream_respects_deadline() -> None:
    client = FakeValkey()
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=50
    )
    fingerprint = _completed_outcome(40, {"missing": True}).envelope_fingerprint
    wait_s = 0.05
    started = time.monotonic()
    deadline = started + wait_s

    found = store.wait_for_terminal((fingerprint,), deadline=deadline)
    elapsed = time.monotonic() - started

    assert found == {}
    assert elapsed >= wait_s
    # Upper bound: must not hang well past the deadline (BLOCK 0 / unbounded drain).
    assert elapsed <= wait_s + 0.5
    assert all(block != 0 for block in client.xreadgroup_blocks)


def test_valkey_wait_for_terminal_final_lookup_after_drain_past_deadline() -> None:
    """Last drain may persist after the deadline; return that fingerprint, not timeout."""

    class PersistThenStall(FakeValkey):
        def __init__(self) -> None:
            super().__init__()
            self._served = False

        def xreadgroup(self, groupname, consumername, streams, count, block=None):
            self.commands.append("XREADGROUP")
            self.xreadgroup_blocks.append(block)
            name = next(iter(streams))
            if self.streams.get(name) and not self._served:
                self._served = True
                entries = self.streams[name][:count]
                self.streams[name] = self.streams[name][len(entries) :]
                time.sleep(0.08)
                return [(name, entries)] if entries else []
            return []

    client = PersistThenStall()
    outcome = _completed_outcome(41, {"late": True})
    client.streams["outcomes"] = [("1-0", _record(outcome))]
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=10
    )
    deadline = time.monotonic() + 0.05

    found = store.wait_for_terminal((outcome.envelope_fingerprint,), deadline=deadline)

    assert found == {outcome.envelope_fingerprint: outcome}
    assert client.acked == ["1-0"]


def test_valkey_drain_respects_deadline_under_endless_fresh_stream() -> None:
    """Fresh XREADGROUP batches must stop at the deadline even if messages keep arriving."""

    class EndlessFresh(FakeValkey):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0
            self._template = _record(_completed_outcome(200, {"n": 0}))

        def xreadgroup(self, groupname, consumername, streams, count, block=None):
            self.commands.append("XREADGROUP")
            self.xreadgroup_blocks.append(block)
            self.reads += 1
            if self.reads > 40:
                raise AssertionError(f"unbounded fresh drain: {self.reads} reads")
            # Simulate continuous arrivals without burning LoopbackExecutor each time.
            time.sleep(0.02)
            fields = dict(self._template)
            return [("outcomes", [(f"{self.reads}-0", fields)])]

    client = EndlessFresh()
    store = ValkeyStreamsOutcomeStore(
        client,
        stream="outcomes",
        group="drivers",
        consumer="d1",
        batch_size=1,
        block_ms=10,
    )
    wait_s = 0.15
    started = time.monotonic()
    deadline = started + wait_s

    store._drain(deadline=deadline)
    elapsed = time.monotonic() - started

    assert elapsed <= wait_s + 0.35
    assert 1 <= client.reads <= 20
    assert len(client.acked) == client.reads
    # Same template fingerprint → HSETNX keeps a single durable winner.
    assert len(client.hashes.get("outcomes:committed", {})) == 1
    assert 0 not in client.xreadgroup_blocks


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

    outcome = backend.run_batch(
        (envelope,), worker_manifest=_manifest(), payload=_MC_PAYLOAD
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.result == {"worker": "valkey"}
    job = next(fields for _id, fields in client.streams["jobs"] if "envelope" in fields)
    assert job["fingerprint"] == envelope_fingerprint(envelope)
    assert json.loads(job["payload"]) == _MC_PAYLOAD


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
        requested_device="cpu",
        effective_device="cpu",
        payload=_MC_PAYLOAD,
    )

    assert len(outcomes) == 2
    job_fields = [fields for _id, fields in client.streams["jobs"]]
    assert all(job["requested_device"] == "cpu" for job in job_fields)
    assert all(job["effective_device"] == "cpu" for job in job_fields)


def test_worker_provenance_ignores_host_environment_overrides(monkeypatch) -> None:
    """Worker argv and library version must not be redirectable by env vars."""
    from fast_mlsirm import remote_exec, remote_worker

    monkeypatch.setenv("FAST_MLSIRM_WORKER_ENTRY", "/tmp/stub_entry.py")
    monkeypatch.setenv("FAST_MLSIRM_LIBRARY_VERSION", "0.0.0-spoofed")

    assert remote_exec._worker_module_command("/usr/bin/python3") == [
        "/usr/bin/python3",
        "-m",
        "fast_mlsirm.remote_worker",
    ]
    assert remote_worker._library_version() != "0.0.0-spoofed"


def _backend(client: FakeValkey) -> ValkeyStreamsBackend:
    return ValkeyStreamsBackend(
        client,
        jobs_stream="jobs",
        outcomes_stream="outcomes",
        group="drivers",
        consumer="d1",
        block_ms=0,
        wait_timeout_s=0.2,
    )


def test_valkey_backend_requires_payload_before_publishing() -> None:
    client = FakeValkey()
    with pytest.raises(ValueError, match="payload is required"):
        _backend(client).run_batch((_envelope(),), worker_manifest=_manifest())
    assert "jobs" not in client.streams


def test_valkey_backend_rejects_payload_outside_manifest_identity() -> None:
    client = FakeValkey()
    with pytest.raises(CohortMismatchError, match="payload identity"):
        _backend(client).run_batch(
            (_envelope(),), worker_manifest=_manifest(), payload={"config": {"n_persons": 7}}
        )
    assert "jobs" not in client.streams


def test_valkey_backend_orders_outcomes_by_unit_then_run_id() -> None:
    class WorkerValkey(FakeValkey):
        def xadd(self, name, fields):
            record_id = super().xadd(name, fields)
            if name == "jobs":
                envelope = json.loads(fields["envelope"])
                outcome = LoopbackExecutor().run_batch(
                    (_envelope(unit_index=envelope["unit_index"], run_id=envelope["run_id"]),),
                    lambda _e, _s: envelope["run_id"],
                    worker_manifest=_manifest(),
                )[0]
                self.streams.setdefault("outcomes", []).append(("9-0", _record(outcome)))
            return record_id

    envelopes = (_envelope(unit_index=0, run_id="run_b"), _envelope(unit_index=0, run_id="run_a"))
    outcomes = _backend(WorkerValkey()).run_batch(
        envelopes, worker_manifest=_manifest(), payload=_MC_PAYLOAD
    )
    assert [outcome.run_id for outcome in outcomes] == ["run_a", "run_b"]


def _failed_outcome(unit_index: int) -> RemoteJobOutcome:
    def fail(_envelope, _seed):
        raise RuntimeError("did not converge")

    return LoopbackExecutor().run_batch(
        (_envelope(unit_index=unit_index),), fail, worker_manifest=_manifest()
    )[0]


def test_valkey_backend_returns_failed_outcome_instead_of_timing_out() -> None:
    class FailingWorkerValkey(FakeValkey):
        def xadd(self, name, fields):
            record_id = super().xadd(name, fields)
            if name == "jobs":
                self.streams.setdefault("outcomes", []).append(
                    ("7-0", _record(_failed_outcome(2)))
                )
            return record_id

    client = FailingWorkerValkey()
    started = time.monotonic()
    (outcome,) = _backend(client).run_batch(
        (_envelope(unit_index=2),), worker_manifest=_manifest(), payload=_MC_PAYLOAD
    )

    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "did not converge" in outcome.error_message
    assert time.monotonic() - started < 0.2
    assert "7-0" in client.acked


def test_valkey_store_prefers_later_success_over_recorded_failure() -> None:
    client = FakeValkey()
    failed = _failed_outcome(3)
    completed = _completed_outcome(3, {"retry": "ok"})
    client.streams["outcomes"] = [("1-0", _record(failed)), ("2-0", _record(completed))]
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )

    found = store.wait_for_terminal(
        (completed.envelope_fingerprint,), deadline=time.monotonic() + 0.2
    )

    assert found[completed.envelope_fingerprint] == completed
    assert store.committed_success(completed.envelope_fingerprint) == completed


def test_valkey_store_failure_is_terminal_but_not_a_committed_success() -> None:
    client = FakeValkey()
    failed = _failed_outcome(4)
    client.streams["outcomes"] = [("1-0", _record(failed))]
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )

    found = store.wait_for_terminal(
        (failed.envelope_fingerprint,), deadline=time.monotonic() + 0.2
    )

    assert found[failed.envelope_fingerprint] == failed
    assert store.committed_success(failed.envelope_fingerprint) is None


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, "5"])
def test_valkey_backend_rejects_non_finite_or_non_numeric_wait_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="wait_timeout_s"):
        ValkeyStreamsBackend(
            FakeValkey(),
            jobs_stream="jobs",
            outcomes_stream="outcomes",
            group="drivers",
            consumer="d1",
            wait_timeout_s=timeout,
        )


@pytest.mark.parametrize(
    "outcome_json",
    ["[" * 100 + "]" * 100, json.dumps({"pad": "x" * 2_000_000})],
    ids=["deeply-nested", "oversized"],
)
def test_valkey_store_rejects_unbounded_outcome_json(outcome_json) -> None:
    client = FakeValkey()
    client.streams["outcomes"] = [
        ("1-0", {"fingerprint": "a" * 64, "outcome": outcome_json})
    ]
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )

    with pytest.raises(ValueError, match="outcome JSON"):
        store.consume_available()
    assert client.acked == []


def test_transport_executors_share_the_dispatch_protocol_signature() -> None:
    import inspect

    from fast_mlsirm.remote_exec import RemoteDispatchBackend, SubprocessExecutor

    expected = inspect.signature(RemoteDispatchBackend.run_batch)
    for backend in (SubprocessExecutor, ValkeyStreamsBackend):
        assert inspect.signature(backend.run_batch).parameters.keys() == (
            expected.parameters.keys()
        ), backend.__name__


def test_valkey_outcome_depth_check_ignores_brackets_inside_strings() -> None:
    from fast_mlsirm.remote_exec import _json_nesting_exceeds

    assert not _json_nesting_exceeds(json.dumps({"s": "[" * 500}), 64)
    assert not _json_nesting_exceeds(json.dumps({"s": '\\"[[' * 100}), 64)
    assert _json_nesting_exceeds("[" * 65 + "]" * 65, 64)
    assert not _json_nesting_exceeds("[" * 64 + "]" * 64, 64)
