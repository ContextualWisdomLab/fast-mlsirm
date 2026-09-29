"""Valkey Streams worker: the reference consumer of ValkeyStreamsBackend jobs."""

from __future__ import annotations

import json
import os
import socket

from fast_mlsirm.remote_exec import (
    RemoteJobDeliveryState,
    ValkeyStreamsBackend,
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


def _worker(client, *, executor=lambda envelope, payload: {"unit": envelope.unit_index}, manifest=None):
    return ValkeyStreamsWorker(
        client,
        jobs_stream=_JOBS,
        outcomes_stream=_OUTCOMES,
        group="workers",
        consumer="w1",
        worker_manifest=manifest or _manifest(),
        executor=executor,
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
