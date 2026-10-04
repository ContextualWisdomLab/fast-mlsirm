# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Behavioral contract for L4 remote execution envelopes and loopback executor."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue
import socket
import sqlite3
import time

import pytest
from importlib.metadata import PackageNotFoundError, version

try:
    LIBRARY_VERSION = version("fast-mlsirm")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.11.4"

from fast_mlsirm.remote_exec import (
    CohortMismatchError,
    ExecutionFloatPath,
    INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES,
    INDEX_SEED_STEP,
    LoopbackExecutor,
    OutcomeCommitLedger,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteJobOutcome,
    RemoteRunManifest,
    RemoteWorkerProvenance,
    SEED_DERIVATION_RULE,
    SQLiteOutcomeCommitLedger,
    SubprocessExecutor,
    admit_remote_job_family,
    admit_remote_job_internal_shard,
    derive_index_seed,
    envelope_fingerprint,
    result_identity_sha256,
)

_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _run_sqlite_executor(database: str, result_queue) -> None:
    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(
        family=RemoteJobFamily.MC_REPLICATE,
        unit_index=1,
        base_seed=77,
        manifest=manifest,
    )
    outcome = SubprocessExecutor(
        socket.gethostname(),
        ledger=SQLiteOutcomeCommitLedger(database),
    ).run_batch((envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD)[0]
    result_queue.put(outcome.to_dict())


def _commit_sqlite_candidate(database: str, candidate: str, start, result_queue) -> None:
    manifest = _manifest()
    envelope = _envelope(manifest=manifest)
    fingerprint = envelope_fingerprint(envelope)
    outcome = RemoteJobOutcome(
        run_id=envelope.run_id,
        unit_index=envelope.unit_index,
        unit_seed=derive_index_seed(envelope.base_seed, envelope.unit_index),
        family=envelope.family,
        delivery_state=RemoteJobDeliveryState.COMPLETED,
        result={"candidate": candidate},
        error_message=None,
        provenance=RemoteWorkerProvenance(
            hostname=socket.gethostname(),
            architecture="test",
            operating_system="test",
            library_version=manifest.library_version,
            source_sha256=manifest.source_sha256,
            requested_device="cpu",
            effective_device="cpu",
            wall_clock_seconds=0.0,
            worker_host=socket.gethostname(),
            worker_pid=os.getpid(),
            cross_host_execution=False,
        ),
        input_identity_sha256=fingerprint,
        output_identity_sha256=result_identity_sha256({"candidate": candidate}),
        envelope_fingerprint=fingerprint,
        driver_host=socket.gethostname(),
        driver_pid=os.getpid(),
    )
    start.wait()
    committed = SQLiteOutcomeCommitLedger(database).commit_success(fingerprint, outcome)
    result_queue.put(committed.to_dict())


def _stall(stop) -> None:
    stop.wait()


def _collect_process_results(processes, result_queue, *, timeout: float):
    deadline = time.monotonic() + timeout
    try:
        try:
            results = [
                result_queue.get(timeout=max(0, deadline - time.monotonic()))
                for _ in processes
            ]
        except queue.Empty as error:
            raise TimeoutError("child process did not return a result") from error
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive():
                raise TimeoutError("child process did not exit")
            assert process.exitcode == 0
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 2
        for process in processes:
            if process.pid is not None:
                process.join(max(0, cleanup_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        kill_deadline = time.monotonic() + 2
        for process in processes:
            if process.is_alive():
                process.join(max(0, kill_deadline - time.monotonic()))
            assert not process.is_alive()
        result_queue.close()
        result_queue.join_thread()


def _manifest(*, payload_sha256: str = _SHA, source_sha256: str = _SHA_B) -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256=source_sha256,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_sha256,
        integration_nodes_sha256=_SHA_C,
    )


def _envelope(
    *,
    family: RemoteJobFamily = RemoteJobFamily.MC_REPLICATE,
    unit_index: int = 3,
    base_seed: int = 20260917,
    run_id: str = "cp3_bootstrap_001",
    manifest: RemoteRunManifest | None = None,
) -> RemoteJobEnvelope:
    man = manifest or _manifest()
    return RemoteJobEnvelope(
        run_id=run_id,
        family=family,
        unit_index=unit_index,
        base_seed=base_seed,
        payload_ref=man.payload_sha256,
        manifest=man,
    )


def _payload_manifest(payload: dict[str, object]) -> RemoteRunManifest:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _manifest(payload_sha256=hashlib.sha256(encoded).hexdigest())


_MC_PAYLOAD = {
    "config": {"n_persons": 24, "n_dims": 1, "items_per_dim": 4, "latent_dim": 1, "gamma": 1.0}
}


def test_derive_index_seed_matches_bifactor_bootstrap_golden_step() -> None:
    """Replicate seeds must match the in-tree bootstrap driver formula."""
    base_seed = 20260917
    for unit_index in (0, 1, 7, 41):
        expected = int((base_seed + unit_index * INDEX_SEED_STEP) & 0xFFFF_FFFF_FFFF_FFFF)
        assert derive_index_seed(base_seed, unit_index) == expected


@pytest.mark.parametrize(
    "family",
    [
        RemoteJobFamily.FIT_RESTART,
        RemoteJobFamily.SCORING_PERSON,
        RemoteJobFamily.MC_REPLICATE,
    ],
)
def test_legal_job_families_accepted(family: RemoteJobFamily) -> None:
    envelope = _envelope(family=family)
    assert envelope.family is family
    assert admit_remote_job_family(family.value) is family


@pytest.mark.parametrize(
    "family", sorted(INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES)
)
def test_sequential_families_allow_whole_call_remote_placement(family: str) -> None:
    admitted = admit_remote_job_family(family)
    envelope = _envelope(family=admitted)

    assert envelope.family.value == family
    assert RemoteJobEnvelope.from_dict(envelope.to_dict()) == envelope


@pytest.mark.parametrize(
    "family", sorted(INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES)
)
def test_sequential_families_reject_internal_sharding(family: str) -> None:
    with pytest.raises(ValueError, match="cannot be sharded inside one call"):
        admit_remote_job_internal_shard(family)


@pytest.mark.parametrize(
    "family", sorted(INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES)
)
def test_internal_shard_guard_blocks_loopback_batch_bypass(family: str) -> None:
    """Whole-call admission cannot bypass the executor batch shard gate."""
    admitted = RemoteJobFamily(family)
    manifest = _manifest()
    envelopes = tuple(
        _envelope(family=admitted, unit_index=index, manifest=manifest)
        for index in range(2)
    )
    handled: list[int] = []

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        del unit_seed
        handled.append(envelope.unit_index)
        return envelope.unit_index

    with pytest.raises(ValueError, match="cannot be sharded inside one call"):
        LoopbackExecutor().run_batch(
            envelopes,
            handler,
            worker_manifest=manifest,
        )

    assert handled == []


@pytest.mark.parametrize(
    "family", sorted(INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES)
)
def test_internal_shard_guard_blocks_subprocess_batch_bypass(family: str) -> None:
    """Subprocess dispatch reuses the same batch shard gate as loopback."""
    admitted = RemoteJobFamily(family)
    payload = {"placeholder": True}
    manifest = _payload_manifest(payload)
    envelopes = tuple(
        _envelope(family=admitted, unit_index=index, manifest=manifest)
        for index in range(2)
    )

    with pytest.raises(ValueError, match="cannot be sharded inside one call"):
        SubprocessExecutor(socket.gethostname()).run_batch(
            envelopes,
            worker_manifest=manifest,
            payload=payload,
        )


def test_internal_shard_guard_allows_distinct_whole_calls() -> None:
    """Separate run ids remain valid whole-call placements for sequential families."""
    manifest = _manifest()
    envelopes = tuple(
        _envelope(
            family=RemoteJobFamily.TWO_TIER,
            unit_index=0,
            manifest=manifest,
            run_id=f"two_tier_run_{index}",
        )
        for index in range(2)
    )
    handled: list[str] = []

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> str:
        del unit_seed
        handled.append(envelope.run_id)
        return envelope.run_id

    outcomes = LoopbackExecutor().run_batch(
        envelopes,
        handler,
        worker_manifest=manifest,
    )

    assert len(outcomes) == 2
    assert handled == ["two_tier_run_0", "two_tier_run_1"]


def test_envelope_round_trip_json_is_stable() -> None:
    envelope = _envelope(family=RemoteJobFamily.FIT_RESTART, unit_index=2)
    restored = RemoteJobEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope
    assert envelope_fingerprint(envelope) == envelope_fingerprint(restored)
    round_trip = json.loads(json.dumps(envelope.to_dict()))
    assert RemoteJobEnvelope.from_dict(round_trip) == envelope


def test_loopback_executor_runs_batch_with_index_derived_seeds() -> None:
    manifest = _manifest()
    envelopes = tuple(
        _envelope(
            family=RemoteJobFamily.MC_REPLICATE,
            unit_index=index,
            base_seed=99,
            manifest=manifest,
        )
        for index in range(4)
    )
    seen: list[tuple[int, int, RemoteJobFamily]] = []

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        seen.append((envelope.unit_index, unit_seed, envelope.family))
        return unit_seed + envelope.unit_index

    executor = LoopbackExecutor()
    outcomes = executor.run_batch(
        envelopes,
        handler,
        worker_manifest=manifest,
        requested_device="cpu",
        effective_device="cpu",
    )

    assert len(outcomes) == 4
    assert all(
        outcome.delivery_state is RemoteJobDeliveryState.COMPLETED for outcome in outcomes
    )
    for envelope, outcome in zip(envelopes, outcomes, strict=True):
        assert outcome.unit_seed == envelope.unit_seed()
        assert outcome.result == envelope.unit_seed() + envelope.unit_index
        assert outcome.provenance.requested_device == "cpu"
        assert outcome.provenance.effective_device == "cpu"
        assert outcome.provenance.library_version == manifest.library_version
    assert seen == [
        (index, derive_index_seed(99, index), RemoteJobFamily.MC_REPLICATE)
        for index in range(4)
    ]


def test_cohort_gate_rejects_mismatched_worker_manifest() -> None:
    driver_manifest = _manifest(source_sha256=_SHA_B)
    worker_manifest = _manifest(source_sha256=_SHA_C)
    envelope = _envelope(manifest=driver_manifest)
    executor = LoopbackExecutor()

    with pytest.raises(CohortMismatchError, match="incompatible with envelope cohort"):
        executor.run_batch(
            (envelope,),
            lambda env, seed: seed,
            worker_manifest=worker_manifest,
        )


def test_loopback_executor_records_handler_failure_without_aborting_batch() -> None:
    manifest = _manifest()
    good = _envelope(unit_index=0, manifest=manifest)
    bad = _envelope(unit_index=1, manifest=manifest)

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        if envelope.unit_index == 1:
            raise RuntimeError("simulated non-convergence")
        return unit_seed

    outcomes = LoopbackExecutor().run_batch(
        (good, bad),
        handler,
        worker_manifest=manifest,
    )

    assert outcomes[0].delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcomes[0].result == good.unit_seed()
    assert outcomes[1].delivery_state is RemoteJobDeliveryState.FAILED
    assert outcomes[1].error_message == "simulated non-convergence"
    assert outcomes[1].result is None


@pytest.mark.parametrize("exception_message", ["", "  \t"])
def test_loopback_executor_records_blank_handler_error_without_aborting_batch(
    exception_message: str,
) -> None:
    """An exception without meaningful text still becomes one failed outcome."""
    manifest = _manifest()

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        del envelope, unit_seed
        raise RuntimeError(exception_message)

    outcomes = LoopbackExecutor().run_batch(
        (_envelope(manifest=manifest),),
        handler,
        worker_manifest=manifest,
    )

    assert len(outcomes) == 1
    assert outcomes[0].delivery_state is RemoteJobDeliveryState.FAILED
    assert outcomes[0].error_message == "RuntimeError"
    assert outcomes[0].result is None


def test_loopback_executor_preflights_entire_batch_before_handler_execution() -> None:
    """A later cohort mismatch cannot leave earlier handler side effects."""
    worker_manifest = _manifest(source_sha256=_SHA_B)
    incompatible_manifest = _manifest(source_sha256=_SHA_C)
    handled_indices: list[int] = []

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        del unit_seed
        handled_indices.append(envelope.unit_index)
        return envelope.unit_index

    with pytest.raises(CohortMismatchError, match="incompatible with envelope cohort"):
        LoopbackExecutor().run_batch(
            (
                _envelope(unit_index=0, manifest=worker_manifest),
                _envelope(unit_index=1, manifest=incompatible_manifest),
            ),
            handler,
            worker_manifest=worker_manifest,
        )

    assert handled_indices == []


def test_loopback_executor_returns_outcomes_in_unit_index_order() -> None:
    """Outcome ordering follows the backend contract, not caller input order."""
    manifest = _manifest()
    outcomes = LoopbackExecutor().run_batch(
        tuple(
            _envelope(unit_index=unit_index, manifest=manifest)
            for unit_index in (2, 0, 1)
        ),
        lambda envelope, unit_seed: (envelope.unit_index, unit_seed),
        worker_manifest=manifest,
    )

    assert [outcome.unit_index for outcome in outcomes] == [0, 1, 2]


def test_payload_ref_must_match_manifest_payload_sha256() -> None:
    manifest = _manifest(payload_sha256=_SHA)
    with pytest.raises(ValueError, match="payload_ref must equal manifest.payload_sha256"):
        RemoteJobEnvelope(
            run_id="run_a",
            family=RemoteJobFamily.SCORING_PERSON,
            unit_index=0,
            base_seed=1,
            payload_ref=_SHA_B,
            manifest=manifest,
        )


def test_remote_job_outcome_validates_failed_state() -> None:
    provenance = RemoteWorkerProvenance(
        hostname="host",
        architecture="arm64",
        operating_system="Darwin",
        library_version=LIBRARY_VERSION,
        source_sha256=_SHA_B,
        requested_device="cpu",
        effective_device="cpu",
        wall_clock_seconds=0.01,
        worker_host="host",
        worker_pid=4242,
        cross_host_execution=False,
    )
    with pytest.raises(ValueError, match="failed outcomes require"):
        RemoteJobOutcome(
            run_id="run_a",
            unit_index=0,
            unit_seed=1,
            family=RemoteJobFamily.FIT_RESTART,
            delivery_state=RemoteJobDeliveryState.FAILED,
            result=None,
            error_message=None,
            provenance=provenance,
            input_identity_sha256=_SHA,
            output_identity_sha256=None,
            envelope_fingerprint=_SHA,
            driver_host="host",
            driver_pid=1111,
        )


def test_subprocess_executor_runs_real_simulate_in_child_process() -> None:
    """Criterion 1+2: real library call in a different OS process with explicit host."""
    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(
        family=RemoteJobFamily.MC_REPLICATE,
        unit_index=0,
        base_seed=20260920,
        manifest=manifest,
    )
    driver_pid = os.getpid()
    driver_host = socket.gethostname()
    worker_host = driver_host
    ledger = OutcomeCommitLedger()
    executor = SubprocessExecutor(worker_host, ledger=ledger, driver_host=driver_host)

    outcomes = executor.run_batch((envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.provenance.worker_host == worker_host
    assert outcome.provenance.worker_pid != driver_pid
    assert outcome.provenance.cross_host_execution is False
    assert outcome.provenance.library_version == LIBRARY_VERSION
    assert type(outcome.result) is dict
    assert outcome.result["library_function"] == "fast_mlsirm.simulate"
    assert outcome.input_identity_sha256 == envelope_fingerprint(envelope)
    assert outcome.output_identity_sha256 is not None
    assert ledger.successful_count(envelope_fingerprint(envelope)) == 1


def test_subprocess_executor_retry_does_not_record_second_success() -> None:
    """Criterion 3: replay of the same envelope fingerprint commits at most once."""
    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(
        family=RemoteJobFamily.MC_REPLICATE,
        unit_index=1,
        base_seed=77,
        manifest=manifest,
    )
    fingerprint = envelope_fingerprint(envelope)
    ledger = OutcomeCommitLedger()
    executor = SubprocessExecutor(socket.gethostname(), ledger=ledger)

    first = executor.run_batch((envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD)[0]
    second = executor.run_batch((envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD)[0]

    assert first.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert second.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert second is first
    assert ledger.successful_count(fingerprint) == 1


def test_sqlite_ledger_deduplicates_success_across_executor_processes(tmp_path) -> None:
    """A new driver process must reuse the success left by an exited driver."""
    database = tmp_path / "remote-outcomes.sqlite3"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()

    first_process = context.Process(target=_run_sqlite_executor, args=(str(database), results))
    first_process.start()
    first = _collect_process_results((first_process,), results, timeout=30)[0]

    results = context.Queue()
    resumed_process = context.Process(target=_run_sqlite_executor, args=(str(database), results))
    resumed_process.start()
    assert _collect_process_results((resumed_process,), results, timeout=30)[0] == first
    assert SQLiteOutcomeCommitLedger(database).successful_count(first["envelope_fingerprint"]) == 1


def test_sqlite_ledger_concurrent_writers_return_one_first_success(tmp_path) -> None:
    database = tmp_path / "remote-outcomes.sqlite3"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    writers = [
        context.Process(
            target=_commit_sqlite_candidate,
            args=(str(database), candidate, start, results),
        )
        for candidate in ("a", "b")
    ]
    for writer in writers:
        writer.start()
    start.set()
    committed = _collect_process_results(writers, results, timeout=30)
    assert committed[0] == committed[1]
    assert committed[0]["result"]["candidate"] in {"a", "b"}
    assert SQLiteOutcomeCommitLedger(database).successful_count(
        committed[0]["envelope_fingerprint"]
    ) == 1


def test_sqlite_process_cleanup_reaps_a_stalled_child() -> None:
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    results = context.Queue()
    child = context.Process(target=_stall, args=(stop,))
    child.start()
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="did not return a result"):
        _collect_process_results((child,), results, timeout=0.2)

    assert time.monotonic() - started < 5
    assert not child.is_alive()


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
@pytest.mark.parametrize("occupied", [False, True], ids=["empty", "committed"])
def test_ledger_rejects_another_envelopes_outcome_without_changing_store(
    tmp_path, durable: bool, occupied: bool,
) -> None:
    """A successful result must belong to the supplied envelope commit key."""
    ledger = (
        SQLiteOutcomeCommitLedger(tmp_path / "remote-outcomes.sqlite3")
        if durable else OutcomeCommitLedger()
    )
    expected = _envelope(unit_index=0)
    unrelated = _envelope(unit_index=1)
    outcomes = LoopbackExecutor().run_batch(
        (expected, unrelated),
        lambda envelope, seed: {"unit_index": envelope.unit_index, "seed": seed},
        worker_manifest=expected.manifest,
    )
    key = envelope_fingerprint(expected)
    other_key = envelope_fingerprint(unrelated)
    if occupied:
        ledger.commit_success(key, outcomes[0])

    with pytest.raises(ValueError, match="outcome fingerprint does not match commit key"):
        ledger.commit_success(key, outcomes[1])

    assert ledger.successful_count(key) == int(occupied)
    assert ledger.committed_success(key) == (outcomes[0] if occupied else None)
    assert ledger.successful_count(other_key) == 0
    assert ledger.committed_success(other_key) is None
    assert ledger.commit_success(key, outcomes[0]) == outcomes[0]
    assert ledger.commit_success(key, outcomes[0]) == outcomes[0]
    assert ledger.successful_count(key) == 1
    assert ledger.committed_success(key) == outcomes[0]


@pytest.mark.parametrize("access", ["lookup", "commit", "executor"])
def test_sqlite_ledger_rejects_legacy_miskeyed_success_without_dispatch(
    tmp_path, monkeypatch, access: str,
) -> None:
    """Reject a row the pre-guard ledger could persist under another unit's key."""
    from fast_mlsirm import remote_exec

    payload: dict[str, object] = dict(_MC_PAYLOAD)
    manifest = _payload_manifest(payload)
    expected = _envelope(unit_index=0, manifest=manifest)
    unrelated = _envelope(unit_index=1, manifest=manifest)
    outcomes = LoopbackExecutor().run_batch(
        (expected, unrelated),
        lambda envelope, seed: {"unit_index": envelope.unit_index, "seed": seed},
        worker_manifest=manifest,
    )
    key = envelope_fingerprint(expected)
    database = tmp_path / "legacy-outcomes.sqlite3"
    ledger = SQLiteOutcomeCommitLedger(database)
    stored_json = json.dumps(outcomes[1].to_dict())
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute(
                "INSERT INTO successful_outcomes (fingerprint, outcome_json) VALUES (?, ?)",
                (key, stored_json),
            )
    finally:
        connection.close()
    dispatched = []

    def no_dispatch(*args, **kwargs):
        dispatched.append(True)
        raise AssertionError("invalid stored identity must fail before worker dispatch")

    monkeypatch.setattr(remote_exec, "_invoke_worker_process", no_dispatch)
    try:
        with pytest.raises(ValueError, match="stored outcome fingerprint does not match commit key"):
            if access == "lookup":
                ledger.committed_success(key)
            elif access == "commit":
                ledger.commit_success(key, outcomes[0])
            else:
                SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
                    (expected,), worker_manifest=manifest, payload=_MC_PAYLOAD
                )
    finally:
        connection = sqlite3.connect(database)
        try:
            rows = connection.execute(
                "SELECT fingerprint, outcome_json FROM successful_outcomes"
            ).fetchall()
        finally:
            connection.close()
        assert rows == [(key, stored_json)]
        assert dispatched == []


@pytest.mark.parametrize("access", ["lookup", "commit", "executor"])
@pytest.mark.parametrize("state", ["completed", "failed"])
def test_sqlite_success_readback_requires_completed_state(
    tmp_path, monkeypatch, access: str, state: str,
) -> None:
    """Reject failed success rows without mutation or worker dispatch."""
    from dataclasses import replace
    from fast_mlsirm import remote_exec

    payload: dict[str, object] = dict(_MC_PAYLOAD)
    manifest = _payload_manifest(payload)
    envelope = _envelope(unit_index=0, manifest=manifest)
    valid = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: {"objective": 1.0},
        worker_manifest=manifest,
    )[0]
    stored_outcome = valid if state == "completed" else replace(
        valid, delivery_state=RemoteJobDeliveryState.FAILED, result=None,
        output_identity_sha256=None, error_message="stored worker failure",
    )
    key = envelope_fingerprint(envelope)
    stored_json = json.dumps(stored_outcome.to_dict())
    database = tmp_path / "success-state.sqlite3"
    ledger = SQLiteOutcomeCommitLedger(database)
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute(
                "INSERT INTO successful_outcomes VALUES (?, ?)", (key, stored_json),
            )
    finally:
        connection.close()
    dispatched = []

    def no_dispatch(*args, **kwargs):
        dispatched.append(True)
        raise AssertionError("stored success must be validated before dispatch")

    monkeypatch.setattr(remote_exec, "_invoke_worker_process", no_dispatch)

    def read():
        if access == "lookup":
            return ledger.committed_success(key)
        if access == "commit":
            return ledger.commit_success(key, valid)
        return SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
            (envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD,
        )[0]

    try:
        if state == "completed":
            assert read() == valid
        else:
            with pytest.raises(ValueError, match="stored successful outcome must be completed"):
                read()
    finally:
        connection = sqlite3.connect(database)
        try:
            assert connection.execute("SELECT * FROM successful_outcomes").fetchall() == [
                (key, stored_json)
            ]
        finally:
            connection.close()
        assert dispatched == []


def test_shared_outcome_decoder_preserves_terminal_failure() -> None:
    """Generic decoding must still support failure records used by stream adapters."""
    from fast_mlsirm import remote_exec

    envelope = _envelope()
    failure = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: (_ for _ in ()).throw(ValueError("failure")),
        worker_manifest=envelope.manifest,
    )[0]
    assert failure.delivery_state is RemoteJobDeliveryState.FAILED
    assert remote_exec._outcome_from_dict(
        failure.to_dict(), fingerprint=envelope_fingerprint(envelope),
    ) == failure


@pytest.mark.parametrize("access", ["lookup", "commit", "executor"])
@pytest.mark.parametrize(
    "case", ["valid", "tamper", "nan", "inf", "-inf", "forged_nan", "forged_nested_inf"],
)
def test_sqlite_ledger_validates_stored_result_before_reuse(
    tmp_path, monkeypatch, access: str, case: str,
) -> None:
    """Reject corrupted result bytes without replacing the row or dispatching."""
    from fast_mlsirm import remote_exec

    payload: dict[str, object] = dict(_MC_PAYLOAD)
    manifest = _payload_manifest(payload)
    envelope = _envelope(unit_index=0, manifest=manifest)
    outcome = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: {"objective": 1.0},
        worker_manifest=manifest,
    )[0]
    stored = outcome.to_dict()
    corruptions = {
        "tamper": 2.0, "nan": float("nan"), "inf": float("inf"),
        "-inf": -float("inf"), "forged_nan": float("nan"),
        "forged_nested_inf": {"nested": [float("inf")]},
    }
    if case != "valid":
        stored["result"] = {"objective": corruptions[case]}
        if case.startswith("forged_"):
            stored["output_identity_sha256"] = hashlib.sha256(
                json.dumps(
                    stored["result"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
    stored_json = json.dumps(stored)
    key = envelope_fingerprint(envelope)
    database = tmp_path / "corrupted-result.sqlite3"
    ledger = SQLiteOutcomeCommitLedger(database)
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute(
                "INSERT INTO successful_outcomes VALUES (?, ?)", (key, stored_json),
            )
    finally:
        connection.close()
    dispatched = []

    def no_dispatch(*args, **kwargs):
        dispatched.append(True)
        raise AssertionError("stored result must be checked before worker dispatch")

    monkeypatch.setattr(remote_exec, "_invoke_worker_process", no_dispatch)

    def read():
        if access == "lookup":
            return ledger.committed_success(key)
        if access == "commit":
            return ledger.commit_success(key, outcome)
        return SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
            (envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD,
        )[0]

    try:
        if case == "valid":
            assert read() == outcome
        else:
            with pytest.raises(ValueError, match="stored completed result"):
                read()
    finally:
        connection = sqlite3.connect(database)
        try:
            rows = connection.execute("SELECT * FROM successful_outcomes").fetchall()
        finally:
            connection.close()
        assert rows == [(key, stored_json)]
        assert dispatched == []


@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("case", ["tamper", "forged_nan", "forged_nested_inf"])
def test_sqlite_ledger_rejects_invalid_result_before_new_commit(
    tmp_path, occupied: bool, case: str,
) -> None:
    """A rejected commit must not persist corruption or hide behind a winner."""
    from dataclasses import replace

    envelope = _envelope()
    valid = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: {"objective": 1.0},
        worker_manifest=envelope.manifest,
    )[0]
    result = {
        "tamper": {"objective": 2.0},
        "forged_nan": {"objective": float("nan")},
        "forged_nested_inf": {"objective": {"nested": [float("inf")]}},
    }[case]
    digest = valid.output_identity_sha256
    if case.startswith("forged_"):
        digest = hashlib.sha256(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest()
    candidate = replace(valid, result=result, output_identity_sha256=digest)
    database = tmp_path / "invalid-new-result.sqlite3"
    ledger = SQLiteOutcomeCommitLedger(database)
    key = valid.envelope_fingerprint
    if occupied:
        ledger.commit_success(key, valid)
    connection = sqlite3.connect(database)
    try:
        before = connection.execute("SELECT * FROM successful_outcomes").fetchall()
    finally:
        connection.close()
    try:
        with pytest.raises(ValueError, match="stored completed result"):
            ledger.commit_success(key, candidate)
    finally:
        connection = sqlite3.connect(database)
        try:
            assert connection.execute("SELECT * FROM successful_outcomes").fetchall() == before
        finally:
            connection.close()


@pytest.mark.parametrize("occupied", [False, True], ids=["empty", "committed"])
@pytest.mark.parametrize("case", ["tamper", "forged_nan", "forged_nested_inf"])
def test_memory_ledger_rejects_invalid_result_before_commit(occupied: bool, case: str) -> None:
    """Memory commits enforce the same finite-result digest as durable commits."""
    from dataclasses import replace

    envelope = _envelope()
    valid = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: {"objective": 1.0},
        worker_manifest=envelope.manifest,
    )[0]
    result = {
        "tamper": {"objective": 2.0},
        "forged_nan": {"objective": float("nan")},
        "forged_nested_inf": {"objective": {"nested": [float("inf")]}},
    }[case]
    digest = valid.output_identity_sha256
    if case.startswith("forged_"):
        digest = hashlib.sha256(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest()
    candidate = replace(valid, result=result, output_identity_sha256=digest)
    ledger = OutcomeCommitLedger()
    key = valid.envelope_fingerprint
    if occupied:
        assert ledger.commit_success(key, valid) == valid
    with pytest.raises(ValueError, match="stored completed result"):
        ledger.commit_success(key, candidate)
    assert ledger.successful_count(key) == int(occupied)
    assert ledger.committed_success(key) == (valid if occupied else None)
    assert ledger.commit_success(key, valid) == valid
    assert ledger.successful_count(key) == 1


@pytest.mark.parametrize("access", ["lookup", "commit", "executor"])
@pytest.mark.parametrize("case", ["valid", "tamper", "nan"])
def test_memory_ledger_validates_mutable_cached_result_before_reuse(
    monkeypatch, access: str, case: str,
) -> None:
    """Nested mutation cannot turn a cached success into unchecked output."""
    payload = dict(_MC_PAYLOAD)
    manifest = _payload_manifest(payload)
    envelope = _envelope(unit_index=0, manifest=manifest)
    valid = LoopbackExecutor().run_batch(
        (envelope,), lambda envelope, seed: {"nested": [1.0]},
        worker_manifest=manifest,
    )[0]
    ledger = OutcomeCommitLedger()
    key = valid.envelope_fingerprint
    cached = ledger.commit_success(key, valid)
    if case != "valid":
        cached.result["nested"][0] = 2.0 if case == "tamper" else float("nan")
    dispatched = []

    def no_dispatch(*args, **kwargs):
        dispatched.append(True)
        raise AssertionError("cached result must be checked before worker dispatch")

    from fast_mlsirm import remote_exec
    monkeypatch.setattr(remote_exec, "_invoke_worker_process", no_dispatch)

    def read():
        if access == "lookup":
            return ledger.committed_success(key)
        if access == "commit":
            replacement = LoopbackExecutor().run_batch(
                (envelope,), lambda envelope, seed: {"nested": [1.0]},
                worker_manifest=manifest,
            )[0]
            return ledger.commit_success(key, replacement)
        return SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
            (envelope,), worker_manifest=manifest, payload=payload,
        )[0]

    try:
        if case == "valid":
            assert read() == valid
        else:
            with pytest.raises(ValueError, match="stored completed result"):
                read()
    finally:
        assert ledger.successful_count(key) == 1
        assert ledger._successful[key] is cached
        assert dispatched == []
        if case == "tamper":
            assert cached.result["nested"] == [2.0]
        elif case == "nan":
            import math
            assert math.isnan(cached.result["nested"][0])


def test_sqlite_ledger_closes_connections_after_repeated_calls(tmp_path, monkeypatch) -> None:
    from fast_mlsirm import remote_exec

    opened = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(remote_exec.sqlite3, "connect", tracked_connect)
    ledger = SQLiteOutcomeCommitLedger(tmp_path / "remote-outcomes.sqlite3")
    for _ in range(3):
        assert ledger.successful_count(_SHA) == 0
        assert ledger.committed_success(_SHA) is None
    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(manifest=manifest)
    SubprocessExecutor(socket.gethostname(), ledger=ledger).run_batch(
        (envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD
    )

    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_subprocess_executor_records_input_output_and_version_identity() -> None:
    """Criterion 4: outcome carries input identity, output identity, and package version."""
    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(family=RemoteJobFamily.MC_REPLICATE, unit_index=2, manifest=manifest)
    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,),
        worker_manifest=manifest,
        payload=_MC_PAYLOAD,
    )[0]

    assert outcome.envelope_fingerprint == envelope_fingerprint(envelope)
    assert outcome.input_identity_sha256 == envelope_fingerprint(envelope)
    assert outcome.output_identity_sha256 is not None
    assert outcome.provenance.library_version == manifest.library_version


@pytest.mark.parametrize(
    ("family", "payload", "library_function"),
    [
        (
            RemoteJobFamily.FIT_RESTART,
            {
                "responses": [[0, 1], [1, 0], [1, 1], [0, 0]],
                "factor_id": [0, 0],
                "config": {
                    "model": "MLS2PLM",
                    "latent_dim": 1,
                    "optimizer": "adam",
                    "max_iter": 1,
                    "n_restarts": 1,
                    "backend": "rust",
                    "rust_device": "cpu",
                },
            },
            "fast_mlsirm.fit",
        ),
        (
            RemoteJobFamily.SCORING_PERSON,
            {
                "a": [1.0, 1.2, 0.8],
                "b": [-0.5, 0.0, 0.5],
                "responses": [[1, 1, 0]],
            },
            "fast_mlsirm.score_wle",
        ),
    ],
)
def test_subprocess_executor_runs_real_fit_and_score_calls(
    family: RemoteJobFamily,
    payload: dict[str, object],
    library_function: str,
) -> None:
    """L4 acceptance: subprocess families invoke production fit/score APIs."""
    manifest = _payload_manifest(payload)
    envelope = _envelope(family=family, unit_index=0, manifest=manifest)

    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,), worker_manifest=manifest, payload=payload
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.result["library_function"] == library_function
    assert outcome.input_identity_sha256 == envelope_fingerprint(envelope)
    assert outcome.output_identity_sha256 is not None
    assert outcome.provenance.library_version == manifest.library_version


def test_subprocess_executor_rejects_payload_identity_mismatch() -> None:
    payload = {"a": [1.0], "b": [0.0], "responses": [[1]]}
    manifest = _payload_manifest(payload)
    envelope = _envelope(family=RemoteJobFamily.SCORING_PERSON, manifest=manifest)

    with pytest.raises(CohortMismatchError, match="payload identity"):
        SubprocessExecutor(socket.gethostname()).run_batch(
            (envelope,),
            worker_manifest=manifest,
            payload={**payload, "responses": [[0]]},
        )


def _fake_completed_worker(stdout: str, *, returncode: int = 0):
    class _Completed:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    return _Completed()


def test_subprocess_executor_fails_closed_on_worker_library_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported worker library_version must match the cohort manifest."""
    import fast_mlsirm.remote_exec as remote_exec

    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(family=RemoteJobFamily.MC_REPLICATE, unit_index=0, manifest=manifest)
    worker_payload = {
        "delivery_state": RemoteJobDeliveryState.COMPLETED.value,
        "result": {"ok": True},
        "worker_pid": 4242,
        "hostname": "remote-worker",
        "architecture": "arm64",
        "operating_system": "Darwin",
        "library_version": "0.0.1",
        "wall_clock_seconds": 0.01,
    }
    monkeypatch.setattr(
        remote_exec,
        "_invoke_worker_process",
        lambda *args, **kwargs: _fake_completed_worker(json.dumps(worker_payload)),
    )

    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,),
        worker_manifest=manifest,
        payload=_MC_PAYLOAD,
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "library_version" in (outcome.error_message or "")
    assert outcome.provenance.library_version == manifest.library_version


def test_subprocess_executor_fails_closed_on_unknown_delivery_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing/unknown delivery_state must not become a committed success."""
    import fast_mlsirm.remote_exec as remote_exec

    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(family=RemoteJobFamily.MC_REPLICATE, unit_index=1, manifest=manifest)
    worker_payload = {
        "worker_pid": 4242,
        "hostname": "remote-worker",
        "architecture": "arm64",
        "operating_system": "Darwin",
        "library_version": LIBRARY_VERSION,
        "wall_clock_seconds": 0.01,
    }
    monkeypatch.setattr(
        remote_exec,
        "_invoke_worker_process",
        lambda *args, **kwargs: _fake_completed_worker(json.dumps(worker_payload)),
    )

    outcome = SubprocessExecutor(socket.gethostname(), ledger=OutcomeCommitLedger()).run_batch(
        (envelope,),
        worker_manifest=manifest,
        payload=_MC_PAYLOAD,
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "delivery_state" in (outcome.error_message or "")
    assert outcome.output_identity_sha256 is None


def test_subprocess_executor_fails_closed_on_non_finite_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite wall_clock_seconds must return FAILED without aborting the batch."""
    import fast_mlsirm.remote_exec as remote_exec

    manifest = _payload_manifest(_MC_PAYLOAD)
    envelope = _envelope(family=RemoteJobFamily.MC_REPLICATE, unit_index=2, manifest=manifest)
    worker_payload = {
        "delivery_state": RemoteJobDeliveryState.COMPLETED.value,
        "result": {"ok": True},
        "worker_pid": 4242,
        "hostname": "remote-worker",
        "architecture": "arm64",
        "operating_system": "Darwin",
        "library_version": LIBRARY_VERSION,
        "wall_clock_seconds": float("inf"),
    }
    monkeypatch.setattr(
        remote_exec,
        "_invoke_worker_process",
        lambda *args, **kwargs: _fake_completed_worker(json.dumps(worker_payload)),
    )

    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,),
        worker_manifest=manifest,
        payload=_MC_PAYLOAD,
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.FAILED
    assert "wall_clock_seconds" in (outcome.error_message or "")
