# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Remote execution backend for ``run_bifactor_bootstrap`` (#2001).

A remote run must reproduce the in-process run replicate by replicate: both
paths call the same resample-and-fit body with the same index-derived seed.
A replicate whose fit raises (for example a resample that misses a response
category) is a result, not a failure. It is counted and never retried, since
redrawing until success would bias the bootstrap sample.
"""

from __future__ import annotations

import socket
import stat
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytest

try:
    LIBRARY_VERSION = version("fast-mlsirm")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.11.4"

from fast_mlsirm.bifactor_bootstrap import (
    RemoteBootstrapBackend,
    run_bifactor_bootstrap,
)
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath,
    OutcomeCommitLedger,
    RemoteRunManifest,
    SEED_DERIVATION_RULE,
    SubprocessExecutor,
)

pytest.importorskip("fast_mlsirm._core")

_N_ITEMS, _N_CAT = 6, 3
_SMAP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)


def _responses(*, rare_category: bool) -> np.ndarray:
    rng = np.random.default_rng(20260930)
    y = rng.integers(0, _N_CAT, size=(60, _N_ITEMS)).astype(np.float64)
    if rare_category:
        # Item 0 uses category 2 exactly once, so many resamples miss it.
        y[:, 0] = rng.integers(0, 2, size=60)
        y[7, 0] = 2
    return y


def _run(y: np.ndarray, *, backend: RemoteBootstrapBackend | None, n_replicates: int = 6):
    return run_bifactor_bootstrap(
        y,
        _SMAP,
        _N_CAT,
        2,
        n_replicates,
        3,
        0.0,
        600.0,
        7,
        7,
        base_seed=20260930,
        ci_level=0.9,
        max_iter=15,
        n_starts=1,
        tol=1e-3,
        n_jobs=1,
        remote_backend=backend,
    )


def _manifest_template() -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256="b" * 64,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256="0" * 64,
        integration_nodes_sha256="c" * 64,
    )


def _backend(executor: SubprocessExecutor) -> RemoteBootstrapBackend:
    return RemoteBootstrapBackend(
        executor=executor, manifest=_manifest_template(), run_id="bootstrap_remote_test"
    )


def _assert_same_replicates(local, remote) -> None:
    assert remote.replicate_ids == local.replicate_ids
    assert remote.converged_replicate_ids == local.converged_replicate_ids
    assert remote.rejected_replicate_ids == local.rejected_replicate_ids
    assert remote.replicate_errors == local.replicate_errors
    for name in (
        "replicate_a_general",
        "replicate_a_specific",
        "replicate_threshold",
        "replicate_loglik",
    ):
        np.testing.assert_array_equal(getattr(remote, name), getattr(local, name), err_msg=name)


def test_remote_backend_reproduces_local_replicates() -> None:
    y = _responses(rare_category=False)
    local = _run(y, backend=None)
    remote = _run(y, backend=_backend(SubprocessExecutor(socket.gethostname())))
    _assert_same_replicates(local, remote)
    assert remote.rejected_replicate_ids == ()


def test_rejected_replicates_are_counted_with_reasons_locally_and_remotely() -> None:
    y = _responses(rare_category=True)
    local = _run(y, backend=None, n_replicates=9)
    assert local.rejected_replicate_ids, "fixture must produce at least one rejected resample"
    assert local.n_rejected == len(local.rejected_replicate_ids)
    for rep in local.rejected_replicate_ids:
        reason = local.replicate_errors[local.replicate_ids.index(rep)]
        assert reason and len(reason) <= 512

    ledger = OutcomeCommitLedger()
    remote = _run(
        y,
        backend=_backend(SubprocessExecutor(socket.gethostname(), ledger=ledger)),
        n_replicates=9,
    )
    _assert_same_replicates(local, remote)


def test_rejected_replicate_is_a_committed_result_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm.remote_exec as remote_exec

    y = _responses(rare_category=True)
    backend = _backend(SubprocessExecutor(socket.gethostname(), ledger=OutcomeCommitLedger()))
    first = _run(y, backend=backend, n_replicates=9)
    assert first.rejected_replicate_ids

    calls: list[object] = []
    real_invoke = remote_exec._invoke_worker_process

    def counting_invoke(*args, **kwargs):
        calls.append(args)
        return real_invoke(*args, **kwargs)

    monkeypatch.setattr(remote_exec, "_invoke_worker_process", counting_invoke)
    second = _run(y, backend=backend, n_replicates=9)
    # Every unit, rejected ones included, replays from the ledger: nothing is redrawn.
    assert calls == []
    assert second.rejected_replicate_ids == first.rejected_replicate_ids


def test_transport_failure_fails_closed(tmp_path: Path) -> None:
    broken = tmp_path / "broken_interpreter"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
    executor = SubprocessExecutor(socket.gethostname(), remote_interpreter=str(broken))
    with pytest.raises(RuntimeError, match="remote bootstrap replicate"):
        _run(_responses(rare_category=False), backend=_backend(executor))
