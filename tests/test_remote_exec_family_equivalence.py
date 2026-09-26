# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Family-complete remote worker evidence and local↔subprocess equivalence (#2072)."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import socket

import numpy as np
import pytest
from importlib.metadata import PackageNotFoundError, version

try:
    LIBRARY_VERSION = version("fast-mlsirm")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.11.4"

from fast_mlsirm.remote_exec import (
    CohortMismatchError,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteRunManifest,
    SEED_DERIVATION_RULE,
    ExecutionFloatPath,
    SubprocessExecutor,
    envelope_fingerprint,
    result_identity_sha256,
)
from fast_mlsirm.remote_worker import execute_envelope

_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _manifest(*, payload_sha256: str = _SHA) -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version=LIBRARY_VERSION,
        source_sha256=_SHA_B,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_sha256,
        integration_nodes_sha256=_SHA_C,
    )


def _payload_manifest(payload: dict[str, object]) -> RemoteRunManifest:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _manifest(payload_sha256=hashlib.sha256(encoded).hexdigest())


def _envelope(
    *,
    family: RemoteJobFamily,
    payload_manifest: RemoteRunManifest,
    unit_index: int = 0,
    base_seed: int = 20260920,
) -> RemoteJobEnvelope:
    return RemoteJobEnvelope(
        run_id=f"family_equiv_{family.value}",
        family=family,
        unit_index=unit_index,
        base_seed=base_seed,
        payload_ref=payload_manifest.payload_sha256,
        manifest=payload_manifest,
    )


def _fit_payload(*, max_iter: int = 1) -> dict[str, object]:
    return {
        "responses": [[0, 1], [1, 0], [1, 1], [0, 0]],
        "factor_id": [0, 0],
        "config": {
            "model": "MLS2PLM",
            "latent_dim": 1,
            "optimizer": "adam",
            "max_iter": max_iter,
            "n_restarts": 1,
            "backend": "rust",
            "rust_device": "cpu",
        },
    }


def _se_derivatives_payload() -> dict[str, object]:
    return {
        "a_general": [1.4, -1.1, 1.0, 1.2, 0.9, 1.1],
        "a_specific": [1.0, 0.9, 1.1, 1.0, 0.8, 0.9],
        "threshold": [
            [1.2, 0.0, -1.2],
            [1.0, -0.1, -1.3],
            [1.3, 0.2, -1.0],
            [1.1, 0.1, -1.1],
            [0.9, -0.2, -1.4],
            [1.2, 0.0, -1.2],
        ],
        "responses": [
            [0, 1, 2, 3, 0, 1],
            [1, 2, 3, 0, 1, 2],
            [2, 3, 0, 1, 2, 3],
            [3, 0, 1, 2, 3, 0],
        ],
        "specific_map": [0, 0, 0, 1, 1, 1],
        "n_cat": 4,
        "n_specific": 2,
        "q_general": 7,
        "q_specific": 7,
        "fd_step": 1e-5,
    }


def _regression_contrasts_payload() -> dict[str, object]:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float64)
    y = np.array([0.5, 1.2, 1.9, 2.4], dtype=np.float64)
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "contrast_vector": [0.0, 1.0],
        "df": 2.0,
        "hc": "HC3",
    }


def _fipc_payload() -> dict[str, object]:
    rng = np.random.default_rng(20260920)
    n_items, n_cat, n_anchor = 6, 3, 4
    y_ref = rng.integers(0, n_cat, size=(120, n_items), dtype=np.int64)
    y_foc = rng.integers(0, n_cat, size=(80, n_items), dtype=np.int64)
    from fast_mlsirm.polytomous import fit_polytomous

    ref = fit_polytomous(y_ref, n_cat=n_cat, model="grm", q_theta=21, max_iter=80, tol=1e-4)
    anchor = np.zeros(n_items, dtype=bool)
    anchor[:n_anchor] = True
    return {
        "responses": y_foc.tolist(),
        "n_cat": n_cat,
        "anchor": anchor.tolist(),
        "anchor_slope": ref.slope.tolist(),
        "anchor_cat_params": ref.cat_params.tolist(),
        "q_theta": 21,
        "max_iter": 80,
        "tol": 1e-4,
    }


def _two_tier_payload() -> dict[str, object]:
    rng = np.random.default_rng(20260920)
    n_persons, n_items, n_cat = 40, 4, 3
    y = np.zeros((n_persons, n_items), dtype=np.int64)
    for item in range(n_items):
        y[:, item] = np.arange(n_persons) % n_cat
        y[:n_cat, item] = np.arange(n_cat)
    return {
        "responses": y.tolist(),
        "primary_map": [[True, False], [True, False], [False, True], [False, True]],
        "specific_map": [0, 0, 0, 0],
        "n_cat": n_cat,
        "n_primary": 2,
        "n_specific": 1,
        "q_primary": 7,
        "q_specific": 7,
        "max_iter": 40,
        "tol": 1e-4,
        "n_starts": 1,
    }


def _scoring_person_payload() -> dict[str, object]:
    return {
        "a": [1.0, 1.2, 0.8],
        "b": [-0.5, 0.0, 0.5],
        "responses": [[1, 1, 0]],
    }


# Lazy factories keep collection green when fast_mlsirm._core is unavailable
# (FIPC payload construction calls fit_polytomous at import time otherwise).
FAMILY_PAYLOADS: tuple[
    tuple[RemoteJobFamily, Callable[[], dict[str, object] | None]],
    ...,
] = (
    (RemoteJobFamily.MC_REPLICATE, lambda: None),
    (RemoteJobFamily.FIT_RESTART, lambda: _fit_payload(max_iter=1)),
    (RemoteJobFamily.SCORING_PERSON, _scoring_person_payload),
    (RemoteJobFamily.EM_M_STEP, lambda: _fit_payload(max_iter=1)),
    (RemoteJobFamily.SE_DERIVATIVES, _se_derivatives_payload),
    (RemoteJobFamily.REGRESSION_CONTRASTS, _regression_contrasts_payload),
    (RemoteJobFamily.FIPC, _fipc_payload),
    (RemoteJobFamily.TWO_TIER, _two_tier_payload),
)


def test_remote_job_family_enum_is_family_complete() -> None:
    """Every policy enum member must map to a worker handler."""
    handled = {member for member, _ in FAMILY_PAYLOADS}
    assert handled == set(RemoteJobFamily)


@pytest.mark.parametrize(("family", "payload_factory"), FAMILY_PAYLOADS)
def test_execute_envelope_handles_every_family(
    family: RemoteJobFamily,
    payload_factory,
) -> None:
    pytest.importorskip("fast_mlsirm._core")
    payload = payload_factory()
    manifest = _manifest() if payload is None else _payload_manifest(payload)
    envelope = _envelope(family=family, payload_manifest=manifest)
    result = execute_envelope(envelope, payload)
    assert result["family"] == family.value
    assert type(result["library_function"]) is str


@pytest.mark.parametrize(("family", "payload_factory"), FAMILY_PAYLOADS)
def test_local_and_subprocess_outcomes_match_for_family(
    family: RemoteJobFamily,
    payload_factory,
) -> None:
    """Local in-process execution and SubprocessExecutor agree on output identity."""
    pytest.importorskip("fast_mlsirm._core")
    payload = payload_factory()
    manifest = _manifest() if payload is None else _payload_manifest(payload)
    envelope = _envelope(family=family, payload_manifest=manifest)
    local_result = execute_envelope(envelope, payload)
    local_identity = result_identity_sha256(local_result)

    outcome = SubprocessExecutor(socket.gethostname()).run_batch(
        (envelope,),
        worker_manifest=manifest,
        payload=payload,
    )[0]

    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.output_identity_sha256 == local_identity
    assert outcome.input_identity_sha256 == envelope_fingerprint(envelope)
    assert outcome.provenance.library_version == manifest.library_version
    assert outcome.result == local_result


def test_cohort_manifest_mismatch_fails_closed_before_dispatch() -> None:
    """Mismatched library_version in the worker manifest must not dispatch."""
    payload = _fit_payload()
    driver_manifest = _payload_manifest(payload)
    worker_manifest = RemoteRunManifest(
        schema_version="1.0",
        library_version="0.0.0",
        source_sha256=_SHA_B,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=driver_manifest.payload_sha256,
        integration_nodes_sha256=_SHA_C,
    )
    envelope = _envelope(
        family=RemoteJobFamily.FIT_RESTART,
        payload_manifest=driver_manifest,
    )
    with pytest.raises(CohortMismatchError, match="incompatible with envelope cohort"):
        SubprocessExecutor(socket.gethostname()).run_batch(
            (envelope,),
            worker_manifest=worker_manifest,
            payload=payload,
        )
