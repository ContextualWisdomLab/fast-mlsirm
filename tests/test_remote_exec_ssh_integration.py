# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional two-host integration probe for L4 subprocess/SSH remote execution."""

from __future__ import annotations

import os
import socket
import subprocess

import pytest
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
)

_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
DEFAULT_REMOTE_SSH_HOST = "seongho@192.168.68.3"
DEFAULT_REMOTE_INTERPRETER = os.environ.get(
    "FAST_MLSIRM_REMOTE_INTERPRETER",
    "/data/orca/workspaces/fmls-2048-remote-exec-s1/.venv/bin/python3.12",
)


def _manifest() -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256=_SHA_B,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=_SHA,
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
    probe = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            remote_host,
            "hostname",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()
        return False, detail or f"ssh exited {probe.returncode}"
    return True, probe.stdout.strip()


@pytest.mark.integration
def test_two_host_mc_replicate_over_ssh_when_reachable() -> None:
    """Run one legal mc_replicate unit on a second host when SSH credentials exist."""
    remote_host = os.environ.get("FAST_MLSIRM_REMOTE_SSH_HOST", DEFAULT_REMOTE_SSH_HOST)
    reachable, detail = _ssh_reachable(remote_host)
    if not reachable:
        pytest.skip(f"SSH host not reachable for two-host probe ({remote_host}): {detail}")

    remote_interpreter = os.environ.get(
        "FAST_MLSIRM_REMOTE_INTERPRETER",
        DEFAULT_REMOTE_INTERPRETER,
    )
    import_check = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            remote_host,
            remote_interpreter,
            "-c",
            "import fast_mlsirm; print(fast_mlsirm.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if import_check.returncode != 0:
        pytest.fail(
            "SSH host reachable but fast_mlsirm import failed: "
            f"{(import_check.stderr or import_check.stdout).strip()}"
        )

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
    outcome = executor.run_batch((envelope,), worker_manifest=manifest)[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.driver_host == driver_host
    assert outcome.driver_pid == driver_pid
    assert outcome.provenance.worker_host == remote_host
    assert outcome.provenance.worker_pid != driver_pid
    assert outcome.provenance.cross_host_execution is True
    assert outcome.provenance.hostname != driver_host
    assert outcome.result["library_function"] == "fast_mlsirm.simulate"
    assert envelope_fingerprint(envelope) == outcome.envelope_fingerprint
