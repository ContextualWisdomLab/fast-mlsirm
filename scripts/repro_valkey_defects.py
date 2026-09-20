#!/usr/bin/env python3
"""Isolated localhost Valkey daemon evidence for PR #2073 outcome-store contracts.

Requires FAST_MLSIRM_VALKEY_URL pointing at a disposable localhost daemon.
Creates only UUID-suffixed keys and deletes them in finally; never mutates
pre-existing Redis data.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import types
import uuid
from pathlib import Path


def _bootstrap_remote_exec() -> None:
    """Load ``fast_mlsirm.remote_exec`` without requiring the compiled ``_core``."""
    if getattr(sys.modules.get("fast_mlsirm"), "_stub_remote", False):
        return
    root = Path(__file__).resolve().parents[1]
    pkg_dir = root / "python" / "fast_mlsirm"
    pkg = types.ModuleType("fast_mlsirm")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg._stub_remote = True
    sys.modules["fast_mlsirm"] = pkg
    path = pkg_dir / "remote_exec.py"
    spec = importlib.util.spec_from_file_location("fast_mlsirm.remote_exec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fast_mlsirm.remote_exec"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)


_bootstrap_remote_exec()

import redis  # noqa: E402

from fast_mlsirm.remote_exec import (  # noqa: E402
    ValkeyStreamsBackend,
    ValkeyStreamsOutcomeStore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
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


def _cleanup(client: redis.Redis, suffix: str, names: list[str]) -> None:
    for key in client.scan_iter(f"fast-mlsirm:*{suffix}*"):
        client.delete(key)
    for name in names:
        client.delete(name)


def main() -> int:
    url = os.environ.get("FAST_MLSIRM_VALKEY_URL")
    if not url:
        print("FAIL: FAST_MLSIRM_VALKEY_URL is required")
        return 2
    if "127.0.0.1" not in url and "localhost" not in url:
        print("FAIL: refusing non-localhost Valkey URL")
        return 2

    client = redis.Redis.from_url(url, decode_responses=True)
    client.ping()
    suffix = uuid.uuid4().hex[:10]
    results: list[tuple[str, str]] = []
    owned: list[str] = []

    try:
        # --- restart lookup recovery (HSETNX durable hash) ---
        stream = f"fast-mlsirm:ev:{suffix}:restart"
        group = f"fast-mlsirm-ev-{suffix}-restart"
        owned.extend([stream, f"{stream}:committed"])
        outcome = _completed(1, {"attempt": 1})
        fp = outcome.envelope_fingerprint
        store1 = ValkeyStreamsOutcomeStore(
            client, stream=stream, group=group, consumer="driver-a", block_ms=50
        )
        store1.commit_success(fp, outcome)
        store2 = ValkeyStreamsOutcomeStore(
            client, stream=stream, group=group, consumer="driver-b", block_ms=50
        )
        got = store2.committed_success(fp)
        results.append(
            (
                "restart_lookup_recovery",
                "PASS" if got == outcome else f"FAIL got={got!r}",
            )
        )

        # --- first-success atomicity (HSETNX) via concurrent connections/threads ---
        stream_dup = f"fast-mlsirm:ev:{suffix}:hsetnx"
        group_dup = f"fast-mlsirm-ev-{suffix}-hsetnx"
        owned.extend([stream_dup, f"{stream_dup}:committed"])
        first = _completed(2, {"winner": "first"})
        second = _completed(2, {"winner": "second"})
        fp2 = first.envelope_fingerprint
        ValkeyStreamsOutcomeStore(
            client, stream=stream_dup, group=group_dup, consumer="seed", block_ms=50
        )
        barrier = threading.Barrier(2)
        winners: list[dict | None] = [None, None]
        errors: list[BaseException] = []

        def _race(index: int, consumer: str, outcome_obj) -> None:
            try:
                local = redis.Redis.from_url(url, decode_responses=True)
                store = ValkeyStreamsOutcomeStore(
                    local,
                    stream=stream_dup,
                    group=group_dup,
                    consumer=consumer,
                    block_ms=50,
                )
                barrier.wait()
                winner = store.commit_success(outcome_obj.envelope_fingerprint, outcome_obj)
                winners[index] = winner.to_dict()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_race, args=(0, "proc-a", first)),
            threading.Thread(target=_race, args=(1, "proc-b", second)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        durable = client.hget(f"{stream_dup}:committed", fp2)
        durable_result = json.loads(durable)["result"] if durable else None
        same = (
            winners[0] is not None
            and winners[1] is not None
            and winners[0]["result"] == winners[1]["result"]
        )
        results.append(
            (
                "first_success_atomicity_hsetnx",
                "PASS"
                if not errors
                and same
                and durable_result in ({"winner": "first"}, {"winner": "second"})
                else f"FAIL winners={winners!r} durable={durable_result!r} errors={errors!r}",
            )
        )

        # --- concurrent consumers: barrier-synced drains on distinct connections ---
        stream_cc = f"fast-mlsirm:ev:{suffix}:concurrent"
        group_cc = f"fast-mlsirm-ev-{suffix}-concurrent"
        owned.extend([stream_cc, f"{stream_cc}:committed"])
        ValkeyStreamsOutcomeStore(
            client, stream=stream_cc, group=group_cc, consumer="seed", block_ms=50
        )
        outcomes_cc = [_completed(100 + i, {"c": i}) for i in range(16)]
        for oc in outcomes_cc:
            client.xadd(stream_cc, _record(oc))
        barrier_cc = threading.Barrier(2)
        seen: list[set[str]] = [set(), set()]

        def _drain_consumer(index: int, consumer: str) -> None:
            local = redis.Redis.from_url(url, decode_responses=True)
            store = ValkeyStreamsOutcomeStore(
                local,
                stream=stream_cc,
                group=group_cc,
                consumer=consumer,
                batch_size=2,
                block_ms=200,
            )
            barrier_cc.wait()
            seen[index] = set(store.consume_available())

        threads = [
            threading.Thread(target=_drain_consumer, args=(0, "consumer-a")),
            threading.Thread(target=_drain_consumer, args=(1, "consumer-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        all_fps = {oc.envelope_fingerprint for oc in outcomes_cc}
        results.append(
            (
                "concurrent_consumers",
                "PASS"
                if (seen[0] | seen[1]) == all_fps and seen[0] and seen[1]
                else f"FAIL a={len(seen[0])} b={len(seen[1])} union={len(seen[0]|seen[1])}",
            )
        )

        # --- raw stream > batch_size (no pre-HSETNX) + deadline drain ---
        jobs = f"fast-mlsirm:ev:{suffix}:jobs"
        outcomes = f"fast-mlsirm:ev:{suffix}:batch"
        batch_group = f"fast-mlsirm-ev-{suffix}-batch"
        owned.extend([jobs, outcomes, f"{outcomes}:committed"])
        envs = [_envelope(unit_index=i) for i in range(5)]
        ValkeyStreamsOutcomeStore(
            client, stream=outcomes, group=batch_group, consumer="seed", block_ms=50
        )
        for env in envs:
            done = _completed(env.unit_index, {"idx": env.unit_index})
            client.xadd(outcomes, _record(done))
        backend = ValkeyStreamsBackend(
            client,
            jobs_stream=jobs,
            outcomes_stream=outcomes,
            group=batch_group,
            consumer="driver",
            block_ms=50,
            wait_timeout_s=5.0,
            min_idle_ms=0,
        )
        backend._outcomes = ValkeyStreamsOutcomeStore(
            client,
            stream=outcomes,
            group=batch_group,
            consumer="driver",
            batch_size=2,
            block_ms=50,
            min_idle_ms=0,
        )
        try:
            got_batch = backend.run_batch(tuple(envs), worker_manifest=_manifest())
            results.append(
                (
                    "deadline_drain_multi_batch_raw_stream",
                    "PASS" if len(got_batch) == 5 else f"FAIL n={len(got_batch)}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - evidence table wants the error
            results.append(("deadline_drain_multi_batch_raw_stream", f"FAIL {exc}"))

        # --- empty-stream deadline (must not hang on BLOCK 0) ---
        # Watchdog thread fails closed if wait_for_committed exceeds upper bound.
        empty_stream = f"fast-mlsirm:ev:{suffix}:empty"
        empty_group = f"fast-mlsirm-ev-{suffix}-empty"
        owned.extend([empty_stream, f"{empty_stream}:committed"])
        store_empty = ValkeyStreamsOutcomeStore(
            client,
            stream=empty_stream,
            group=empty_group,
            consumer="driver",
            block_ms=100,
        )
        missing_fp = _completed(99, {"x": 1}).envelope_fingerprint
        wait_s = 0.4
        upper_s = 2.0
        done_empty = threading.Event()
        watchdog_fired = threading.Event()
        empty_result: dict[str, object] = {}

        def _empty_watchdog() -> None:
            if not done_empty.wait(timeout=upper_s + 0.5):
                watchdog_fired.set()

        watchdog = threading.Thread(target=_empty_watchdog, daemon=True)
        watchdog.start()
        t0 = time.monotonic()
        deadline = t0 + wait_s
        try:
            found = store_empty.wait_for_committed((missing_fp,), deadline=deadline)
            empty_result["found"] = found
        finally:
            done_empty.set()
        elapsed = time.monotonic() - t0
        watchdog.join(timeout=1.0)
        results.append(
            (
                "empty_stream_deadline",
                "PASS"
                if (
                    not watchdog_fired.is_set()
                    and empty_result.get("found") == {}
                    and wait_s - 0.1 <= elapsed <= upper_s
                )
                else (
                    f"FAIL found={empty_result.get('found')!r} elapsed={elapsed:.3f} "
                    f"watchdog={watchdog_fired.is_set()}"
                ),
            )
        )

        # --- final lookup after drain past deadline (no false timeout) ---
        late_stream = f"fast-mlsirm:ev:{suffix}:late"
        late_group = f"fast-mlsirm-ev-{suffix}-late"
        owned.extend([late_stream, f"{late_stream}:committed"])
        late_outcome = _completed(98, {"late": True})
        ValkeyStreamsOutcomeStore(
            client, stream=late_stream, group=late_group, consumer="seed", block_ms=50
        )
        client.xadd(late_stream, _record(late_outcome))

        class _SlowPersistStore(ValkeyStreamsOutcomeStore):
            def _accept_records(self, records):  # type: ignore[no-untyped-def]
                time.sleep(0.12)
                return super()._accept_records(records)

        store_late = _SlowPersistStore(
            client,
            stream=late_stream,
            group=late_group,
            consumer="driver",
            batch_size=10,
            block_ms=50,
            min_idle_ms=0,
        )
        late_deadline = time.monotonic() + 0.05
        late_found = store_late.wait_for_committed(
            (late_outcome.envelope_fingerprint,), deadline=late_deadline
        )
        results.append(
            (
                "final_lookup_after_deadline_drain",
                "PASS"
                if late_found.get(late_outcome.envelope_fingerprint) == late_outcome
                else f"FAIL found={set(late_found)!r}",
            )
        )

        # --- bounded drain under continuous XADD (deadline must win) ---
        flood_stream = f"fast-mlsirm:ev:{suffix}:flood"
        flood_group = f"fast-mlsirm-ev-{suffix}-flood"
        owned.extend([flood_stream, f"{flood_stream}:committed"])
        ValkeyStreamsOutcomeStore(
            client, stream=flood_stream, group=flood_group, consumer="seed", block_ms=50
        )
        stop_flood = threading.Event()

        def _flood() -> None:
            n = 0
            while not stop_flood.is_set():
                oc = _completed(300 + (n % 50), {"flood": n})
                client.xadd(flood_stream, _record(oc))
                n += 1
                time.sleep(0.005)

        flood_thread = threading.Thread(target=_flood, daemon=True)
        flood_thread.start()
        store_flood = ValkeyStreamsOutcomeStore(
            client,
            stream=flood_stream,
            group=flood_group,
            consumer="driver",
            batch_size=2,
            block_ms=50,
            min_idle_ms=0,
        )
        flood_wait = 0.25
        flood_t0 = time.monotonic()
        store_flood._drain(deadline=flood_t0 + flood_wait)
        flood_elapsed = time.monotonic() - flood_t0
        stop_flood.set()
        flood_thread.join(timeout=2.0)
        results.append(
            (
                "bounded_drain_under_flood",
                "PASS"
                if flood_elapsed <= flood_wait + 0.75
                else f"FAIL elapsed={flood_elapsed:.3f}",
            )
        )

        # --- mid-pending ineligible then eligible (XAUTOCLAIM min_idle) ---
        pend_stream = f"fast-mlsirm:ev:{suffix}:pending"
        pend_group = f"fast-mlsirm-ev-{suffix}-pending"
        owned.extend([pend_stream, f"{pend_stream}:committed"])
        ready = _completed(50, {"idle": "ready"})
        ValkeyStreamsOutcomeStore(
            client, stream=pend_stream, group=pend_group, consumer="seed", block_ms=50
        )
        client.xadd(pend_stream, _record(ready))
        client.xreadgroup(
            pend_group, "consumer-old", {pend_stream: ">"}, count=10, block=10
        )
        fresh_high = ValkeyStreamsOutcomeStore(
            client,
            stream=pend_stream,
            group=pend_group,
            consumer="consumer-new",
            block_ms=50,
            min_idle_ms=60_000,
        )
        before = fresh_high.consume_available()
        fresh_low = ValkeyStreamsOutcomeStore(
            client,
            stream=pend_stream,
            group=pend_group,
            consumer="consumer-new",
            block_ms=50,
            min_idle_ms=0,
        )
        after = fresh_low.consume_available()
        results.append(
            (
                "pending_idle_ineligible_then_eligible",
                "PASS"
                if ready.envelope_fingerprint not in before
                and ready.envelope_fingerprint in after
                else f"FAIL before={set(before)!r} after={set(after)!r}",
            )
        )

        # --- device-field preservation on published job envelopes ---
        jobs_dev = f"fast-mlsirm:ev:{suffix}:jobs-dev"
        out_dev = f"fast-mlsirm:ev:{suffix}:out-dev"
        group_dev = f"fast-mlsirm-ev-{suffix}-dev"
        owned.extend([jobs_dev, out_dev, f"{out_dev}:committed"])
        env = _envelope(unit_index=70)
        done = _completed(70, {"device": True})
        ValkeyStreamsOutcomeStore(
            client, stream=out_dev, group=group_dev, consumer="seed", block_ms=50
        )
        client.xadd(out_dev, _record(done))
        backend_dev = ValkeyStreamsBackend(
            client,
            jobs_stream=jobs_dev,
            outcomes_stream=out_dev,
            group=group_dev,
            consumer="driver",
            block_ms=50,
            wait_timeout_s=3.0,
            min_idle_ms=0,
        )
        backend_dev.run_batch(
            (env,),
            worker_manifest=_manifest(),
            requested_device="gpu",
            effective_device="cpu",
        )
        entries = client.xrange(jobs_dev)
        fields = entries[0][1] if entries else {}
        results.append(
            (
                "device_field_preservation",
                "PASS"
                if fields.get("requested_device") == "gpu"
                and fields.get("effective_device") == "cpu"
                else f"FAIL fields={fields!r}",
            )
        )

    finally:
        _cleanup(client, suffix, owned)

    width = max(len(name) for name, _ in results)
    for name, status in results:
        print(f"{name:<{width}}  {status}")
    failed = [name for name, status in results if not status.startswith("PASS")]
    print(f"summary  {len(results) - len(failed)}/{len(results)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
