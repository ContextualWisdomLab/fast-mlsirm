"""Valkey Streams worker: the reference consumer of ValkeyStreamsBackend jobs."""

from __future__ import annotations

import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from fast_mlsirm.remote_exec import (
    RemoteJobDeliveryState,
    ValkeyStreamsBackend,
    ValkeyStreamsOutcomeStore,
    envelope_fingerprint,
)
from fast_mlsirm.remote_worker import ValkeyStreamsWorker

from test_remote_exec_valkey import (
    _MC_PAYLOAD,
    FakeValkey,
    _completed_outcome,
    _envelope,
    _manifest,
    _record,
)

_JOBS = "jobs"
_OUTCOMES = "outcomes"


def _worker(
    client, *, executor=lambda envelope, payload: {"unit": envelope.unit_index},
    manifest=None, consumer="w1", batch_size=10,
):
    return ValkeyStreamsWorker(
        client,
        jobs_stream=_JOBS,
        outcomes_stream=_OUTCOMES,
        group="workers",
        consumer=consumer,
        worker_manifest=manifest or _manifest(),
        executor=executor,
        batch_size=batch_size,
        block_ms=0,
    )


def _job_record(job_envelope, payload=_MC_PAYLOAD, **overrides) -> dict[str, str]:
    record = {
        "fingerprint": envelope_fingerprint(job_envelope),
        "envelope": json.dumps(job_envelope.to_dict(), sort_keys=True),
        "payload": json.dumps(payload, sort_keys=True),
        "requested_device": "cpu",
        "effective_device": "cpu",
        "driver_host": "driver-host",
        "driver_pid": "4242",
    }
    record.update(overrides)
    return record


def _published_outcomes(client):
    from fast_mlsirm.remote_exec import _outcome_from_dict

    return [
        _outcome_from_dict(json.loads(fields["outcome"]))
        for _id, fields in client.streams.get(_OUTCOMES, [])
    ]


def test_driver_and_worker_round_trip_a_completed_outcome() -> None:
    class RoundTripValkey(FakeValkey):
        worker = None

        def xadd(self, name, fields):
            record_id = super().xadd(name, fields)
            if name == _JOBS:
                self.worker.run_once()
            return record_id

    client = RoundTripValkey()
    client.worker = _worker(client)
    backend = ValkeyStreamsBackend(
        client,
        jobs_stream=_JOBS,
        outcomes_stream=_OUTCOMES,
        group="drivers",
        consumer="d1",
        block_ms=0,
        wait_timeout_s=0.5,
    )

    (outcome,) = backend.run_batch(
        (_envelope(unit_index=3),), worker_manifest=_manifest(), payload=_MC_PAYLOAD
    )

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.result == {"unit": 3}
    assert outcome.driver_host == socket.gethostname()
    assert outcome.driver_pid == os.getpid()


def test_worker_publishes_completed_outcome_with_driver_identity_and_acks() -> None:
    client = FakeValkey()
    envelope = _envelope(unit_index=1)
    client.streams[_JOBS] = [("1-0", _job_record(envelope))]

    tally = _worker(client).run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.completed == 1
    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.envelope_fingerprint == envelope_fingerprint(envelope)
    assert (outcome.driver_host, outcome.driver_pid) == ("driver-host", 4242)
    assert outcome.provenance.worker_host == socket.gethostname()
    assert outcome.provenance.cross_host_execution is (socket.gethostname() != "driver-host")
    assert client.acked == ["1-0"]


def test_worker_skips_already_committed_job_without_executing() -> None:
    client = FakeValkey()
    envelope = _envelope(unit_index=2)
    committed = _completed_outcome(2, {"first": True})
    client.hashes[f"{_OUTCOMES}:committed"] = {
        committed.envelope_fingerprint: _record(committed)["outcome"]
    }
    client.streams[_JOBS] = [("1-0", _job_record(envelope))]
    calls = []

    tally = _worker(client, executor=lambda e, p: calls.append(e) or {}).run_once()

    assert tally.skipped == 1
    assert calls == []
    assert _published_outcomes(client) == []
    assert client.acked == ["1-0"]


def test_worker_publishes_failed_outcome_when_executor_raises() -> None:
    def explode(_envelope, _payload):
        raise RuntimeError("did not converge")

    client = FakeValkey()
    client.streams[_JOBS] = [("1-0", _job_record(_envelope(unit_index=4)))]

    tally = _worker(client, executor=explode).run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.failed == 1
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "did not converge" in outcome.error_message
    assert client.acked == ["1-0"]


def test_worker_fails_closed_on_cohort_mismatch_without_executing() -> None:
    from test_remote_exec import _manifest as other_manifest

    client = FakeValkey()
    client.streams[_JOBS] = [("1-0", _job_record(_envelope(unit_index=5)))]
    calls = []

    tally = _worker(
        client,
        executor=lambda e, p: calls.append(e) or {},
        manifest=other_manifest(source_sha256="c" * 64),
    ).run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.failed == 1
    assert calls == []
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "cohort" in outcome.error_message


@pytest.mark.parametrize("payload", [{"tampered": True}, None, []])
def test_worker_validates_payload_before_calling_injected_executor(payload) -> None:
    client = FakeValkey()
    client.streams[_JOBS] = [("1-0", _job_record(_envelope(), payload))]
    calls = []

    tally = _worker(client, executor=lambda e, p: calls.append(p) or {}).run_once()

    (outcome,) = _published_outcomes(client)
    assert calls == []
    assert tally.failed == 1
    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "payload" in outcome.error_message
    assert client.acked == ["1-0"]


@pytest.mark.parametrize("claimed_device", ["gpu", "cpu", None])
@pytest.mark.parametrize("use_default_executor", [False, True])
def test_worker_does_not_attest_producer_device_labels(claimed_device, use_default_executor) -> None:
    client = FakeValkey()
    record = _job_record(_envelope(), requested_device="gpu")
    if claimed_device is None:
        del record["effective_device"]
    else:
        record["effective_device"] = claimed_device
    client.streams[_JOBS] = [("1-0", record)]
    if use_default_executor:
        worker = ValkeyStreamsWorker(
            client, jobs_stream=_JOBS, outcomes_stream=_OUTCOMES, group="workers",
            consumer="w1", worker_manifest=_manifest(), block_ms=0,
        )
    else:
        worker = _worker(client)

    tally = worker.run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.completed == 1
    assert outcome.provenance.requested_device == "gpu"
    assert outcome.provenance.effective_device == "unknown"


def test_worker_fails_closed_on_oversized_payload_field() -> None:
    client = FakeValkey()
    record = _job_record(_envelope(unit_index=6))
    record["payload"] = json.dumps({"pad": "x" * 2_000_000})
    client.streams[_JOBS] = [("1-0", record)]

    tally = _worker(client).run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.failed == 1
    assert "JSON exceeds" in outcome.error_message


def test_worker_acks_poison_job_and_keeps_processing_the_batch() -> None:
    client = FakeValkey()
    good = _envelope(unit_index=7)
    client.streams[_JOBS] = [
        ("1-0", {"fingerprint": "not-a-fingerprint", "envelope": "{"}),
        ("2-0", _job_record(_envelope(unit_index=8), envelope="[[[")),
        ("3-0", _job_record(good)),
    ]

    tally = _worker(client).run_once()

    assert tally.poisoned == 2
    assert tally.completed == 1
    assert [o.unit_index for o in _published_outcomes(client)] == [7]
    assert client.acked == ["1-0", "2-0", "3-0"]


def test_worker_rejects_envelope_that_does_not_match_its_fingerprint() -> None:
    client = FakeValkey()
    record = _job_record(_envelope(unit_index=9))
    record["fingerprint"] = "b" * 64
    client.streams[_JOBS] = [("1-0", record)]

    tally = _worker(client).run_once()

    assert tally.poisoned == 1
    assert _published_outcomes(client) == []


def test_worker_reclaims_a_crashed_members_pending_job() -> None:
    client = FakeValkey()
    client.pending.append(("1-0", _job_record(_envelope(unit_index=10))))

    tally = _worker(client).run_once()

    assert tally.completed == 1
    assert "XAUTOCLAIM" in client.commands
    assert client.acked == ["1-0"]


def test_default_executor_runs_the_production_mc_replicate_family() -> None:
    client = FakeValkey()
    client.streams[_JOBS] = [("1-0", _job_record(_envelope(unit_index=0)))]

    worker = ValkeyStreamsWorker(
        client,
        jobs_stream=_JOBS,
        outcomes_stream=_OUTCOMES,
        group="workers",
        consumer="w1",
        worker_manifest=_manifest(),
        block_ms=0,
    )
    tally = worker.run_once()

    (outcome,) = _published_outcomes(client)
    assert tally.completed == 1, outcome.error_message
    assert outcome.result["library_function"] == "fast_mlsirm.simulate"


@pytest.mark.parametrize("append_before_disconnect", [False, True])
def test_worker_does_not_ack_when_outcome_publication_fails(append_before_disconnect) -> None:
    class DisconnectingValkey(FakeValkey):
        disconnect = True

        def xadd(self, name, fields):
            if name == _OUTCOMES and self.disconnect:
                if append_before_disconnect:
                    super().xadd(name, fields)
                raise ConnectionError("outcome publication disconnected")
            return super().xadd(name, fields)

    client = DisconnectingValkey()
    envelope = _envelope(unit_index=11)
    job = ("1-0", _job_record(envelope))
    client.streams[_JOBS] = [job]
    calls = []

    def execute(e, p):
        calls.append(e.unit_index)
        return {"unit": e.unit_index}

    with pytest.raises(ConnectionError, match="publication disconnected"):
        _worker(client, executor=execute).run_once()
    assert client.acked == []
    assert "XACK" not in client.commands

    # FakeValkey has no PEL: explicitly model redelivery after a disconnect.
    client.disconnect = False
    client.pending.append(job)
    assert _worker(client, executor=execute, consumer="w2").run_once().completed == 1
    outcomes = _published_outcomes(client)
    assert len(outcomes) == 1 + int(append_before_disconnect)
    assert calls == [11, 11]
    assert {o.envelope_fingerprint for o in outcomes} == {envelope_fingerprint(envelope)}
    assert {o.input_identity_sha256 for o in outcomes} == {envelope_fingerprint(envelope)}
    assert len({o.output_identity_sha256 for o in outcomes}) == 1
    assert {o.unit_seed for o in outcomes} == {envelope.unit_seed()}
    assert client.acked == ["1-0"]


@pytest.mark.parametrize("commit_before_retry", [False, True])
def test_worker_ack_failure_retries_or_skips_after_first_success(commit_before_retry) -> None:
    class AckFailingValkey(FakeValkey):
        fail_ack = True
        events = None

        def xadd(self, name, fields):
            self.events.append(("publish", name))
            return super().xadd(name, fields)

        def xack(self, name, groupname, *ids):
            self.events.append(("ack", name))
            if name == _JOBS and self.fail_ack:
                raise ConnectionError("job ACK disconnected")
            return super().xack(name, groupname, *ids)

    client = AckFailingValkey()
    client.events = []
    envelope = _envelope(unit_index=12)
    job = ("1-0", _job_record(envelope))
    client.streams[_JOBS] = [job]
    calls = []

    def execute(e, p):
        calls.append(e.unit_index)
        return {"unit": e.unit_index}

    with pytest.raises(ConnectionError, match="ACK disconnected"):
        _worker(client, executor=execute).run_once()
    (first,) = _published_outcomes(client)
    assert client.events == [("publish", _OUTCOMES), ("ack", _JOBS)]
    assert client.acked == []
    store = ValkeyStreamsOutcomeStore(
        client, stream=_OUTCOMES, group="drivers", consumer="d1", block_ms=0,
    )
    if commit_before_retry:
        assert store.committed_success(first.envelope_fingerprint) == first

    client.fail_ack = False
    # Redelivery is modeled explicitly, not evidence of actual server recovery.
    client.pending.append(job)
    tally = _worker(client, executor=execute, consumer="w2").run_once()
    assert tally.skipped == int(commit_before_retry)
    assert tally.completed == int(not commit_before_retry)
    assert calls == [12] * (1 if commit_before_retry else 2)
    if not commit_before_retry:
        assert {o.output_identity_sha256 for o in _published_outcomes(client)} == {
            first.output_identity_sha256
        }
    assert store.committed_success(first.envelope_fingerprint) == first
    assert store.successful_count(first.envelope_fingerprint) == 1


def test_reclaimed_live_job_can_execute_twice_but_commit_one_success() -> None:
    client = FakeValkey()
    envelope = _envelope(unit_index=13)
    job = ("1-0", _job_record(envelope))
    client.streams[_JOBS] = [job]
    first_executing = Event()
    both_executing = Barrier(2)
    calls = []

    def execute(e, p):
        calls.append(e.unit_index)
        first_executing.set()
        both_executing.wait(timeout=5)
        return {"unit": e.unit_index}

    workers = [
        _worker(client, executor=execute, consumer=name, batch_size=1)
        for name in ("w1", "w2")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(workers[0].run_once)
        assert first_executing.wait(timeout=5)
        # Model an eligible reclaim while the original executor is still live.
        client.pending.append(job)
        second_future = pool.submit(workers[1].run_once)
        tallies = [future.result(timeout=10) for future in (first_future, second_future)]

    first, second = _published_outcomes(client)
    assert sum(t.completed for t in tallies) == 2
    assert calls == [13, 13]
    assert client.acked == ["1-0", "1-0"]
    assert first.envelope_fingerprint == second.envelope_fingerprint
    assert first.output_identity_sha256 == second.output_identity_sha256
    store = ValkeyStreamsOutcomeStore(
        client, stream=_OUTCOMES, group="drivers", consumer="d1", block_ms=0,
    )
    assert store.committed_success(first.envelope_fingerprint) == first
    assert store.successful_count(first.envelope_fingerprint) == 1
