# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Offline contracts for the operator probe; these do not execute two hosts."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import replace

import pytest

from scripts import run_two_host_ssh_evidence as probe
from fast_mlsirm.remote_exec import (
    RemoteJobOutcome,
    local_worker_provenance,
    result_identity_sha256,
)


@pytest.mark.parametrize(
    "interpreter", ["/opt/py 3/bin/python", "/opt/python;unexpected-command"]
)
def test_import_probe_preserves_remote_shell_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys, interpreter: str
) -> None:
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_SSH_HOST", "test-user@test-host")
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_INTERPRETER", interpreter)
    monkeypatch.setattr(probe, "_ssh_reachable", lambda _host: (True, "test-host"))
    commands: list[list[str]] = []

    def reject_import(argv, **kwargs):
        commands.append(argv)
        # OpenSSH joins command arguments before remote shell interpretation.
        command = " ".join(argv[argv.index("--") + 2 :])
        assert shlex.split(command) == [
            interpreter,
            "-c",
            "import fast_mlsirm; print(fast_mlsirm.__version__)",
        ]
        assert kwargs["timeout"] == probe._SSH_PROBE_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="test import rejection")

    monkeypatch.setattr(probe.subprocess, "run", reject_import)
    assert probe.main() == 1
    assert len(commands) == 1
    assert "import failed" in capsys.readouterr().err


def test_failed_worker_returns_failure_instead_of_dereferencing_none(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_SSH_HOST", "test-user@test-host")
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_INTERPRETER", "/test/python")
    monkeypatch.setattr(probe, "_ssh_reachable", lambda _host: (True, "test-host"))
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=probe.LIBRARY_VERSION, stderr=""
        ),
    )
    monkeypatch.setattr(
        probe.SubprocessExecutor,
        "_execute_one",
        lambda executor, envelope, **kwargs: RemoteJobOutcome(
            run_id=envelope.run_id,
            unit_index=envelope.unit_index,
            unit_seed=envelope.unit_seed(),
            family=envelope.family,
            delivery_state=probe.RemoteJobDeliveryState.FAILED,
            result=None,
            error_message="test worker failure",
            provenance=local_worker_provenance(
                envelope.manifest,
                requested_device="cpu",
                effective_device="cpu",
                wall_clock_seconds=0.0,
            ),
            input_identity_sha256=probe.envelope_fingerprint(envelope),
            output_identity_sha256=None,
            envelope_fingerprint=probe.envelope_fingerprint(envelope),
            driver_host=executor._driver_host,
            driver_pid=executor._driver_pid,
        ),
    )
    assert probe.main() == 1
    output = capsys.readouterr().out
    assert "FAIL: completed" in output
    assert "FAIL: library_function" in output


@pytest.mark.parametrize("cross_host", [False, True])
def test_probe_accepts_only_distinct_worker_host_control(
    monkeypatch: pytest.MonkeyPatch, capsys, cross_host: bool
) -> None:
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_SSH_HOST", "test-user@test-host")
    monkeypatch.setenv("FAST_MLSIRM_REMOTE_INTERPRETER", "/test/python")
    monkeypatch.setattr(probe, "_ssh_reachable", lambda _host: (True, "test-host"))
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=probe.LIBRARY_VERSION, stderr=""
        ),
    )

    def completed_control(executor, envelope, **kwargs):
        result = {"library_function": "fast_mlsirm.simulate"}
        provenance = replace(
            local_worker_provenance(
                envelope.manifest,
                requested_device="cpu",
                effective_device="cpu",
                wall_clock_seconds=0.0,
                worker_host=executor.worker_host,
                worker_pid=4242,
                cross_host_execution=cross_host,
            ),
            hostname="test-distinct-worker" if cross_host else executor._driver_host,
        )
        return RemoteJobOutcome(
            run_id=envelope.run_id,
            unit_index=envelope.unit_index,
            unit_seed=envelope.unit_seed(),
            family=envelope.family,
            delivery_state=probe.RemoteJobDeliveryState.COMPLETED,
            result=result,
            error_message=None,
            provenance=provenance,
            input_identity_sha256=probe.envelope_fingerprint(envelope),
            output_identity_sha256=result_identity_sha256(result),
            envelope_fingerprint=probe.envelope_fingerprint(envelope),
            driver_host=executor._driver_host,
            driver_pid=executor._driver_pid,
        )

    monkeypatch.setattr(probe.SubprocessExecutor, "_execute_one", completed_control)
    assert probe.main() == (0 if cross_host else 1)
    output = capsys.readouterr().out
    if cross_host:
        assert "FAIL:" not in output
    else:
        assert "FAIL: cross_host_execution" in output
        assert "FAIL: distinct_hostname" in output
