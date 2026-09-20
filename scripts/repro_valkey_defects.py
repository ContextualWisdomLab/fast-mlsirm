#!/usr/bin/env python3
"""Reproduce Valkey transport defects from PR #2073 head against an isolated daemon."""

from __future__ import annotations

import inspect
import json
import os
import sys
import uuid

import redis

from fast_mlsirm.remote_exec import (
    RemoteJobDeliveryState,
    ValkeyStreamsBackend,
    ValkeyStreamsOutcomeStore,
)

# Import test helpers from the suite (script is run from repo root).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_remote_exec import _envelope, _manifest  # noqa: E402


def _completed(unit_index: int, result: object):
    from fast_mlsirm.remote_exec import LoopbackExecutor

    envelope = _envelope(unit_index=unit_index)
    return LoopbackExecutor().run_batch(
        (envelope,), lambda _e, _s: result, worker_manifest=envelope.manifest
    )[0]


def _record(outcome) -> dict[str, str]:
    return {
        "fingerprint": outcome.envelope_fingerprint,
        "outcome": json.dumps(outcome.to_dict(), sort_keys=True, separators=(",", ":")),
    }


def main() -> int:
    url = os.environ.get("FAST_MLSIRM_VALKEY_URL", "redis://127.0.0.1:16380/0")
    client = redis.Redis.from_url(url, decode_responses=True)
    suffix = uuid.uuid4().hex[:8]
    stream = f"fast-mlsirm:repro:outcomes:{suffix}"
    group = f"fast-mlsirm-repro-{suffix}"
    results: list[str] = []

    try:
        outcome = _completed(1, {"attempt": 1})
        fp = outcome.envelope_fingerprint

        # Defect 1: ACK'd outcomes are not visible to a fresh store in the same group.
        store1 = ValkeyStreamsOutcomeStore(
            client, stream=stream, group=group, consumer="driver-a", block_ms=100
        )
        store1.commit_success(fp, outcome)
        got1 = store1.committed_success(fp)
        store2 = ValkeyStreamsOutcomeStore(
            client, stream=stream, group=group, consumer="driver-b", block_ms=100
        )
        got2 = store2.committed_success(fp)
        if got1 is not None and got2 is None:
            results.append("DEFECT1 restart_recovery: fresh consumer cannot read ACK'd outcome")
        elif got1 is not None and got2 is not None:
            results.append("DEFECT1 restart_recovery: PASS")

        # Defect 3: concurrent commit_success should match SQLite first-success contract.
        stream_dup = f"{stream}:dup"
        group_dup = f"{group}-dup"
        first = _completed(2, {"winner": "first"})
        second = _completed(2, {"winner": "second"})
        fp2 = first.envelope_fingerprint
        store_a = ValkeyStreamsOutcomeStore(
            client, stream=stream_dup, group=group_dup, consumer="a", block_ms=50
        )
        store_b = ValkeyStreamsOutcomeStore(
            client, stream=stream_dup, group=group_dup, consumer="b", block_ms=50
        )
        winner_a = store_a.commit_success(fp2, first)
        winner_b = store_b.commit_success(fp2, second)
        if winner_a.result != winner_b.result:
            results.append(
                "DEFECT3 duplicate_commit: divergent winners "
                f"{winner_a.result!r} vs {winner_b.result!r}"
            )
        else:
            results.append("DEFECT3 duplicate_commit: PASS (same winner returned)")

        # Defect 4: run_batch drains only one consume_available batch.
        jobs = f"{stream}:jobs"
        outcomes = f"{stream}:batch-out"
        batch_group = f"{group}-batch"
        envs = [_envelope(unit_index=i) for i in range(3)]
        backend = ValkeyStreamsBackend(
            client,
            jobs_stream=jobs,
            outcomes_stream=outcomes,
            group=batch_group,
            consumer="driver",
            block_ms=50,
        )
        for env in envs:
            done = _completed(env.unit_index, {"idx": env.unit_index})
            ValkeyStreamsOutcomeStore(
                client,
                stream=outcomes,
                group=batch_group,
                consumer="worker",
                block_ms=50,
            ).commit_success(done.envelope_fingerprint, done)
        try:
            backend.run_batch(tuple(envs), worker_manifest=_manifest())
            results.append("DEFECT4 batch_drain: PASS (all outcomes returned)")
        except TimeoutError as exc:
            results.append(f"DEFECT4 batch_drain: {exc}")

        # Defect 5: requested/effective device dropped from ValkeyStreamsBackend.run_batch.
        sig = inspect.signature(ValkeyStreamsBackend.run_batch)
        params = sig.parameters
        if "requested_device" not in params or "effective_device" not in params:
            results.append("DEFECT5 device_params: missing from run_batch signature")
        else:
            source = inspect.getsource(ValkeyStreamsBackend.run_batch)
            if "del requested_device, effective_device" in source:
                results.append(
                    "DEFECT5 device_params: run_batch discards requested/effective device"
                )
            else:
                results.append("DEFECT5 device_params: PASS")

    finally:
        for key in client.scan_iter(f"fast-mlsirm:*{suffix}*"):
            client.delete(key)
        for name in (
            stream,
            f"{stream}:committed",
            f"{stream}:dup",
            f"{stream}:dup:committed",
            f"{stream}:jobs",
            f"{stream}:batch-out",
            f"{stream}:batch-out:committed",
        ):
            client.delete(name)

    print("\n".join(results))
    defects = [line for line in results if "DEFECT" in line and "PASS" not in line]
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
