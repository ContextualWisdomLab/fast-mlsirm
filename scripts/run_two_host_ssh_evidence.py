# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Two-host evidence for L4 subprocess/SSH remote execution.

Runs one legal ``mc_replicate`` unit on a second host and checks that the
outcome carries cross-host provenance. It needs a real second machine, so it
is operator evidence rather than a CI test (a skipped pytest node would trip
the fail-closed outcome gate, and loopback SSH cannot satisfy the
``hostname != driver_host`` check).

Usage::

    FAST_MLSIRM_REMOTE_SSH_HOST=user@host \
    FAST_MLSIRM_REMOTE_INTERPRETER=/path/to/.venv/bin/python \
    python scripts/run_two_host_ssh_evidence.py

Exit status: 0 on PASS, 1 on a failed check, 2 when the host is not
configured or not reachable.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    LIBRARY_VERSION = version("fast-mlsirm")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.11.4"

from fast_mlsirm.remote_exec import (
    OutcomeCommitLedger,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteRunManifest,
    SEED_DERIVATION_RULE,
    SubprocessExecutor,
    ExecutionFloatPath,
    envelope_fingerprint,
    payload_identity_sha256,
)

_MC_PAYLOAD = {
    "config": {"n_persons": 24, "n_dims": 1, "items_per_dim": 4, "latent_dim": 1, "gamma": 1.0}
}
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SSH_PROBE_TIMEOUT_SECONDS = 20.0


def _manifest() -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256=_SHA_B,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_identity_sha256(_MC_PAYLOAD),
        integration_nodes_sha256=_SHA_C,
    )


def _envelope() -> RemoteJobEnvelope:
    manifest = _manifest()
    return RemoteJobEnvelope(
        run_id="two_host_probe_001",
        family=RemoteJobFamily.MC_REPLICATE,
        unit_index=0,
        base_seed=20260920,
        payload_ref=manifest.payload_sha256,
        manifest=manifest,
    )


def _ssh_reachable(remote_host: str) -> tuple[bool, str]:
    try:
        probe = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "--",
                remote_host,
                "hostname",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_SSH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"ssh probe timed out after {_SSH_PROBE_TIMEOUT_SECONDS:.0f}s"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()
        return False, detail or f"ssh exited {probe.returncode}"
    return True, probe.stdout.strip()


def main() -> int:
    """Run one legal mc_replicate unit on a second host and check provenance."""
    remote_host = os.environ.get("FAST_MLSIRM_REMOTE_SSH_HOST")
    remote_interpreter = os.environ.get("FAST_MLSIRM_REMOTE_INTERPRETER")
    if not remote_host or not remote_interpreter:
        print(
            "SKIP: set FAST_MLSIRM_REMOTE_SSH_HOST and FAST_MLSIRM_REMOTE_INTERPRETER",
            file=sys.stderr,
        )
        return 2

    reachable, detail = _ssh_reachable(remote_host)
    if not reachable:
        print(f"SKIP: SSH host not reachable ({remote_host}): {detail}", file=sys.stderr)
        return 2

    try:
        import_check = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                "--",
                remote_host,
                shlex.join(
                    [
                        remote_interpreter,
                        "-c",
                        "import fast_mlsirm; print(fast_mlsirm.__version__)",
                    ]
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_SSH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(
            f"SKIP: SSH import probe timed out after {_SSH_PROBE_TIMEOUT_SECONDS:.0f}s "
            f"on {remote_host}",
            file=sys.stderr,
        )
        return 2
    if import_check.returncode != 0:
        print(
            "FAIL: SSH host reachable but fast_mlsirm import failed: "
            f"{(import_check.stderr or import_check.stdout).strip()}",
            file=sys.stderr,
        )
        return 1

    driver_host = socket.gethostname()
    driver_pid = os.getpid()
    envelope = _envelope()
    manifest = envelope.manifest
    executor = SubprocessExecutor(
        remote_host,
        remote_interpreter=remote_interpreter,
        ledger=OutcomeCommitLedger(),
        driver_host=driver_host,
    )
    outcome = executor.run_batch(
        (envelope,), worker_manifest=manifest, payload=_MC_PAYLOAD
    )[0]

    checks = {
        "completed": outcome.delivery_state is RemoteJobDeliveryState.COMPLETED,
        "driver_host": outcome.driver_host == driver_host,
        "driver_pid": outcome.driver_pid == driver_pid,
        "worker_host": outcome.provenance.worker_host == remote_host,
        "worker_pid": type(outcome.provenance.worker_pid) is int
        and outcome.provenance.worker_pid > 0,
        "cross_host_execution": outcome.provenance.cross_host_execution is True,
        "distinct_hostname": outcome.provenance.hostname != driver_host,
        "library_function": isinstance(outcome.result, dict)
        and outcome.result.get("library_function") == "fast_mlsirm.simulate",
        "fingerprint": envelope_fingerprint(envelope) == outcome.envelope_fingerprint,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
