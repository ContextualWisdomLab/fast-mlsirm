#!/usr/bin/env python3
"""Real fast_mlsirm SubprocessExecutor + ValkeyStreamsOutcomeStore 3-host evidence.

LABEL: generic remote/Valkey path validation — NOT research model complete.
Uses mc_replicate → fast_mlsirm.simulate with fixed manifest inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis

from fast_mlsirm.remote_exec import (
    SEED_DERIVATION_RULE,
    ExecutionFloatPath,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteRunManifest,
    SubprocessExecutor,
    ValkeyStreamsOutcomeStore,
    envelope_fingerprint,
)

VALKEY_URL = os.environ.get("FAST_MLSIRM_VALKEY_URL", "redis://192.168.68.3:16381/0")
OUT = Path(os.environ.get(
    "HETERO_EVIDENCE_OUT",
    "billing-snapshots/valkey_library_3host_mc_replicate_20260920.json",
))
SOURCE_ROOT = Path(__file__).resolve().parents[1]
LIB_SHA = subprocess.check_output(
    ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
).strip()

# Fixed cohort inputs (not shrunk research fixtures claiming convergence).
PAYLOAD = "a" * 64
SOURCE = hashlib.sha256(f"hetero-lib-{LIB_SHA}".encode()).hexdigest()
NODES = "c" * 64
BASE_SEED = 20260920
RUN_ID = f"hetero_lib_{uuid.uuid4().hex[:8]}"


def manifest() -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version="0.11.4",
        source_sha256=SOURCE,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=PAYLOAD,
        integration_nodes_sha256=NODES,
    )


def envelope(unit_index: int) -> RemoteJobEnvelope:
    return RemoteJobEnvelope(
        run_id=RUN_ID,
        family=RemoteJobFamily.MC_REPLICATE,
        unit_index=unit_index,
        base_seed=BASE_SEED,
        payload_ref=PAYLOAD,
        manifest=manifest(),
    )


def valkey_info(client: redis.Redis) -> dict:
    info = client.info("server")
    return {
        "redis_version": info.get("redis_version"),
        "os": info.get("os"),
        "tcp_port": info.get("tcp_port"),
    }


def run_host(
    *,
    label: str,
    worker_host: str,
    remote_interpreter: str,
    worker_entry: str | None,
    units: list[int],
    client: redis.Redis,
    stream: str,
    group: str,
) -> dict:
    env_backup = os.environ.get("FAST_MLSIRM_WORKER_ENTRY")
    if worker_entry:
        os.environ["FAST_MLSIRM_WORKER_ENTRY"] = worker_entry
    elif "FAST_MLSIRM_WORKER_ENTRY" in os.environ:
        del os.environ["FAST_MLSIRM_WORKER_ENTRY"]

    store = ValkeyStreamsOutcomeStore(
        client,
        stream=stream,
        group=group,
        consumer=f"driver-{label}-{uuid.uuid4().hex[:6]}",
        block_ms=50,
        min_idle_ms=100,
    )
    executor = SubprocessExecutor(
        worker_host,
        remote_interpreter=remote_interpreter,
        ledger=store,
        driver_host=socket.gethostname(),
    )
    envelopes = tuple(envelope(u) for u in units)
    t0 = time.perf_counter()
    outcomes = executor.run_batch(envelopes, worker_manifest=manifest(), requested_device="cpu", effective_device="cpu")
    elapsed = time.perf_counter() - t0

    if env_backup is None:
        os.environ.pop("FAST_MLSIRM_WORKER_ENTRY", None)
    else:
        os.environ["FAST_MLSIRM_WORKER_ENTRY"] = env_backup

    rows = []
    for o in outcomes:
        rows.append(
            {
                "unit_index": o.unit_index,
                "delivery_state": o.delivery_state.value,
                "envelope_fingerprint": o.envelope_fingerprint,
                "result": o.result,
                "error_message": o.error_message,
                "worker_host_cfg": worker_host,
            }
        )
    return {
        "label": label,
        "worker_host": worker_host,
        "remote_interpreter": remote_interpreter,
        "worker_entry": worker_entry,
        "elapsed_s": elapsed,
        "outcomes": rows,
        "completed": sum(1 for r in rows if r["delivery_state"] == "completed"),
        "failed": sum(1 for r in rows if r["delivery_state"] == "failed"),
    }


def deliberate_fail_reclaim(client: redis.Redis, stream: str, group: str) -> dict:
    """Fail once (bad interpreter), then reclaim via Valkey ledger miss + successful rerun."""
    unit = 99
    env = envelope(unit)
    fp = envelope_fingerprint(env)
    store = ValkeyStreamsOutcomeStore(
        client, stream=stream, group=group, consumer=f"retry-{uuid.uuid4().hex[:6]}", block_ms=50
    )
    # Attempt 1: force failure with nonexistent interpreter on local path
    bad = SubprocessExecutor("local-bad", remote_interpreter="/nonexistent/python-hetero", ledger=store)
    try:
        bad_out = bad.run_batch((env,), worker_manifest=manifest())[0]
    except Exception as exc:  # noqa: BLE001 — record failure mode
        bad_out = None
        fail_exc = str(exc)
    else:
        fail_exc = bad_out.error_message
    assert store.committed_success(fp) is None, "failed attempt must not commit success"
    # Attempt 2: real local worker (Air venv) without worker entry
    os.environ.pop("FAST_MLSIRM_WORKER_ENTRY", None)
    good = SubprocessExecutor("local-air", remote_interpreter=os.environ["HETERO_AIR_PYTHON"], ledger=store)
    good_out = good.run_batch((env,), worker_manifest=manifest())[0]
    # Attempt 3: duplicate must hit ledger first-success
    again = good.run_batch((env,), worker_manifest=manifest())[0]
    return {
        "fingerprint": fp,
        "first_attempt_error": fail_exc,
        "first_attempt_state": None if bad_out is None else bad_out.delivery_state.value,
        "reclaim_state": good_out.delivery_state.value,
        "reclaim_result": good_out.result,
        "dedup_same_result": again.result == good_out.result,
        "dedup_same_fingerprint": again.envelope_fingerprint == good_out.envelope_fingerprint,
        "PASS": (
            good_out.delivery_state is RemoteJobDeliveryState.COMPLETED
            and again.result == good_out.result
            and store.committed_success(fp) is not None
        ),
    }


def main() -> int:
    client = redis.Redis.from_url(VALKEY_URL, decode_responses=True)
    client.ping()
    suffix = uuid.uuid4().hex[:10]
    stream = f"fast-mlsirm:libhetero:{suffix}:outcomes"
    group = f"libhetero-{suffix}"

    air_py = os.environ["HETERO_AIR_PYTHON"]
    # Unit partition across hosts (fixed inputs, shared cohort).
    air = run_host(
        label="air",
        worker_host="local-air",
        remote_interpreter=air_py,
        worker_entry=None,
        units=[0, 1, 2],
        client=client,
        stream=stream,
        group=group,
    )
    s1 = run_host(
        label="s1",
        worker_host="seongho@192.168.68.3",
        remote_interpreter="python3",
        worker_entry="/home/seongho/fmls-hetero/scripts/hetero_remote_worker_entry.py",
        units=[3, 4, 5],
        client=client,
        stream=stream,
        group=group,
    )
    m1 = run_host(
        label="m1",
        worker_host="seonghobae@10.6.0.3",
        remote_interpreter="/tmp/fmls-hetero/.venv/bin/python",
        worker_entry="/tmp/fmls-hetero/scripts/hetero_remote_worker_entry.py",
        units=[6, 7, 8],
        client=client,
        stream=stream,
        group=group,
    )
    retry = deliberate_fail_reclaim(client, stream, group)

    # Aggregate equality: same unit_index+seed must match response_sha256 if re-run on Air
    # Cross-host: verify all completed and library_function is simulate
    all_rows = air["outcomes"] + s1["outcomes"] + m1["outcomes"]
    all_completed = all(r["delivery_state"] == "completed" for r in all_rows)
    all_simulate = all(
        (r.get("result") or {}).get("library_function") == "fast_mlsirm.simulate"
        for r in all_rows
    )

    out = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": "generic fast_mlsirm remote/Valkey path validation — NOT research model complete",
        "valkey_url": VALKEY_URL,
        "valkey_server": valkey_info(client),
        "library_git_sha": LIB_SHA,
        "per_host_library_sha_note": "workers report PackageNotFound library_version=0+unknown; source tree SHA is library_git_sha above; Air also has linked _core from fit-score WT",
        "fixed_inputs": {
            "family": "mc_replicate",
            "n_persons": 24,
            "items_per_dim": 4,
            "base_seed": BASE_SEED,
            "run_id": RUN_ID,
            "payload_sha256": PAYLOAD,
            "source_sha256": SOURCE,
            "seed_derivation_rule": SEED_DERIVATION_RULE,
        },
        "hosts": {"air": air, "s1": s1, "m1": m1},
        "fail_reclaim_dedup": retry,
        "checks": {
            "all_nine_completed": all_completed and len(all_rows) == 9,
            "all_library_function_simulate": all_simulate,
            "fail_reclaim_dedup_pass": retry["PASS"],
        },
        "PASS": all_completed and all_simulate and retry["PASS"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"PASS": out["PASS"], "out": str(OUT.resolve()), "checks": out["checks"]}, indent=2))
    client.delete(stream, f"{stream}:committed")
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
