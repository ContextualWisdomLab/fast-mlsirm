# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Subprocess worker entry point for legal L4 remote numerical families."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time

import numpy as np

from importlib.metadata import PackageNotFoundError, version

from ._fit_public import fit
from .config import FitConfig, MLS2PLMConfig
from .remote_exec import (
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    payload_identity_sha256,
    result_identity_sha256,
)
from .simulation import simulate
from .wle import score_wle


def _library_version() -> str:
    try:
        return version("fast-mlsirm")
    except PackageNotFoundError:
        return "0+unknown"


def execute_mc_replicate(unit_seed: int) -> dict[str, object]:
    """Run one Monte Carlo replicate via ``simulate`` with ``unit_seed``."""
    config = MLS2PLMConfig(
        n_persons=24,
        n_dims=1,
        items_per_dim=4,
        latent_dim=1,
        gamma=1.0,
        seed=unit_seed,
    )
    data = simulate(config)
    response_sha256 = hashlib.sha256(data.Y.tobytes()).hexdigest()
    return {
        "family": RemoteJobFamily.MC_REPLICATE.value,
        "response_sha256": response_sha256,
        "n_persons": int(data.Y.shape[0]),
        "n_items": int(data.Y.shape[1]),
        "library_function": "fast_mlsirm.simulate",
    }


def execute_fit_restart(payload: dict[str, object], unit_seed: int) -> dict[str, object]:
    """Run one production ``fit`` restart from a JSON payload."""
    config_values = dict(payload.get("config", {}))
    config_values["seed"] = unit_seed
    result = fit(
        np.asarray(payload["responses"], dtype=np.uint8),
        np.asarray(payload["factor_id"], dtype=np.int64),
        FitConfig(**config_values),
    )
    parameter_bytes = b"".join(
        np.asarray(value).tobytes()
        for value in (
            result.params.theta,
            result.params.alpha,
            result.params.b,
            result.params.xi,
            result.params.zeta,
        )
    )
    return {
        "family": RemoteJobFamily.FIT_RESTART.value,
        "library_function": "fast_mlsirm.fit",
        "objective": float(result.objective),
        "convergence_status": result.convergence_status,
        "n_iter": int(result.n_iter),
        "parameter_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
    }


def execute_scoring_person(payload: dict[str, object]) -> dict[str, object]:
    """Run one production ``score_wle`` person shard from a JSON payload."""
    result = score_wle(
        np.asarray(payload["a"], dtype=np.float64),
        np.asarray(payload["b"], dtype=np.float64),
        np.asarray(payload["responses"], dtype=np.float64),
    )
    return {
        "family": RemoteJobFamily.SCORING_PERSON.value,
        "library_function": "fast_mlsirm.score_wle",
        "theta": result["theta"].tolist(),
        "se": result["se"].tolist(),
        "boundary": result["boundary"].tolist(),
    }


def execute_envelope(
    envelope: RemoteJobEnvelope, payload: object = None
) -> dict[str, object]:
    """Dispatch one envelope to the production library function for its family."""
    unit_seed = envelope.unit_seed()
    if envelope.family is RemoteJobFamily.MC_REPLICATE:
        return execute_mc_replicate(unit_seed)
    if type(payload) is not dict:
        raise ValueError(f"payload is required for {envelope.family.value}")
    if payload_identity_sha256(payload) != envelope.manifest.payload_sha256:
        raise ValueError("payload identity does not match envelope manifest")
    if envelope.family is RemoteJobFamily.FIT_RESTART:
        return execute_fit_restart(payload, unit_seed)
    if envelope.family is RemoteJobFamily.SCORING_PERSON:
        return execute_scoring_person(payload)
    raise ValueError(f"unsupported remote family {envelope.family.value!r}")


def main(argv: list[str] | None = None) -> int:
    del argv
    started = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(json.dumps({"delivery_state": RemoteJobDeliveryState.FAILED.value, "error_message": str(exc)}))
        return 1

    if type(request) is not dict or "envelope" not in request:
        print(
            json.dumps(
                {
                    "delivery_state": RemoteJobDeliveryState.FAILED.value,
                    "error_message": "worker request must include envelope",
                }
            )
        )
        return 1

    try:
        envelope = RemoteJobEnvelope.from_dict(request["envelope"])
        result = execute_envelope(envelope, request.get("payload"))
    except Exception as exc:  # worker failures are serialized, not raised to driver
        print(
            json.dumps(
                {
                    "delivery_state": RemoteJobDeliveryState.FAILED.value,
                    "error_message": str(exc) or type(exc).__name__,
                    "worker_pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "architecture": platform.machine(),
                    "operating_system": platform.system(),
                    "library_version": _library_version(),
                    "wall_clock_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "delivery_state": RemoteJobDeliveryState.COMPLETED.value,
                "result": result,
                "output_identity_sha256": result_identity_sha256(result),
                "worker_pid": os.getpid(),
                "hostname": socket.gethostname(),
                "architecture": platform.machine(),
                "operating_system": platform.system(),
                "library_version": _library_version(),
                "wall_clock_seconds": elapsed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
