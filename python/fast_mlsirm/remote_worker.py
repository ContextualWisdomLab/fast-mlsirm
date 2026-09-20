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
from .bifactor_grm import bifactor_oakes_se
from .config import FitConfig, MLS2PLMConfig
from .fipc_group_score import execute_fipc_group_person_score_payload
from .polytomous import fit_poly_fipc
from .regression import contrast, fit_ols_hc
from .remote_exec import (
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    payload_identity_sha256,
    result_identity_sha256,
)
from .simulation import simulate
from .two_tier_grm import fit_two_tier_grm
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


def _fit_record(result: object, *, family: RemoteJobFamily) -> dict[str, object]:
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
        "family": family.value,
        "library_function": "fast_mlsirm.fit",
        "objective": float(result.objective),
        "convergence_status": result.convergence_status,
        "n_iter": int(result.n_iter),
        "parameter_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
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
    return _fit_record(result, family=RemoteJobFamily.FIT_RESTART)


def execute_em_m_step(payload: dict[str, object], unit_seed: int) -> dict[str, object]:
    """Run one EM/JMLE iteration via ``fit`` with caller-owned ``max_iter=1``."""
    config_values = dict(payload.get("config", {}))
    config_values["seed"] = unit_seed
    config_values["max_iter"] = 1
    result = fit(
        np.asarray(payload["responses"], dtype=np.uint8),
        np.asarray(payload["factor_id"], dtype=np.int64),
        FitConfig(**config_values),
    )
    return _fit_record(result, family=RemoteJobFamily.EM_M_STEP)


def execute_se_derivatives(payload: dict[str, object]) -> dict[str, object]:
    """Run ``bifactor_oakes_se`` for one whole-call SE/derivative family unit."""
    res = bifactor_oakes_se(
        np.asarray(payload["a_general"], dtype=np.float64),
        np.asarray(payload["a_specific"], dtype=np.float64),
        np.asarray(payload["threshold"], dtype=np.float64),
        np.asarray(payload["responses"], dtype=np.int64),
        np.asarray(payload["specific_map"], dtype=np.int64),
        int(payload["n_cat"]),
        int(payload["n_specific"]),
        q_general=int(payload["q_general"]),
        q_specific=int(payload["q_specific"]),
        fd_step=float(payload["fd_step"]),
    )
    information_sha256 = hashlib.sha256(res.information.tobytes()).hexdigest()
    se_sha256 = (
        hashlib.sha256(res.se.tobytes()).hexdigest()
        if res.se is not None
        else None
    )
    return {
        "family": RemoteJobFamily.SE_DERIVATIVES.value,
        "library_function": "fast_mlsirm.bifactor_grm.bifactor_oakes_se",
        "positive_definite": bool(res.positive_definite),
        "label_count": len(res.labels),
        "information_sha256": information_sha256,
        "se_sha256": se_sha256,
        "non_pd_reason": res.non_pd_reason,
    }


def execute_regression_contrasts(payload: dict[str, object]) -> dict[str, object]:
    """Run ``fit_ols_hc`` + ``contrast`` for one regression/contrast family unit."""
    x = np.asarray(payload["x"], dtype=np.float64)
    y = np.asarray(payload["y"], dtype=np.float64)
    fit_result = fit_ols_hc(x, y, hc=str(payload.get("hc", "HC3")))
    contrast_result = contrast(
        fit_result["beta"],
        fit_result["vcov"],
        np.asarray(payload["contrast_vector"], dtype=np.float64),
        df=float(payload["df"]),
    )
    return {
        "family": RemoteJobFamily.REGRESSION_CONTRASTS.value,
        "library_function": "fast_mlsirm.regression.contrast",
        "estimate": contrast_result["estimate"],
        "se": contrast_result["se"],
        "wald_chi2": contrast_result["wald_chi2"],
        "p_chi2": contrast_result["p_chi2"],
        "beta_sha256": hashlib.sha256(fit_result["beta"].tobytes()).hexdigest(),
    }


def execute_fipc(payload: dict[str, object]) -> dict[str, object]:
    """Run ``fit_poly_fipc`` for one whole-call FIPC family unit."""
    fit = fit_poly_fipc(
        np.asarray(payload["responses"], dtype=np.int64),
        int(payload["n_cat"]),
        np.asarray(payload["anchor"], dtype=bool),
        np.asarray(payload["anchor_slope"], dtype=np.float64),
        np.asarray(payload["anchor_cat_params"], dtype=np.float64),
        q_theta=int(payload["q_theta"]),
        max_iter=int(payload["max_iter"]),
        tol=float(payload["tol"]),
    )
    parameter_bytes = b"".join(
        np.asarray(value).tobytes()
        for value in (fit.slope, fit.cat_params, np.array([fit.mu, fit.sigma]))
    )
    return {
        "family": RemoteJobFamily.FIPC.value,
        "library_function": "fast_mlsirm.polytomous.fit_poly_fipc",
        "converged": bool(fit.converged),
        "mu": float(fit.mu),
        "sigma": float(fit.sigma),
        "parameter_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
    }


def execute_fipc_group_person_score(payload: dict[str, object]) -> dict[str, object]:
    """Run FIPC group person EAP + expected-raw + optional reference moments.

    Kim (2006) FIPC identification: anchors pin the bank to the reference
    metric; focal ``N(mu, sigma^2)`` is free. Not the #2077 two-tier
    ``E[T|theta_f]`` nuisance-integrated curve API.
    """
    return execute_fipc_group_person_score_payload(payload)


def execute_two_tier(payload: dict[str, object], unit_seed: int) -> dict[str, object]:
    """Run ``fit_two_tier_grm`` for one whole-call two-tier family unit."""
    fit = fit_two_tier_grm(
        np.asarray(payload["responses"], dtype=np.int64),
        np.asarray(payload["primary_map"], dtype=bool),
        np.asarray(payload["specific_map"], dtype=np.int64),
        int(payload["n_cat"]),
        int(payload["n_primary"]),
        int(payload["n_specific"]),
        int(payload["q_primary"]),
        int(payload["q_specific"]),
        int(payload["max_iter"]),
        float(payload["tol"]),
        int(payload["n_starts"]),
        unit_seed,
    )
    parameter_bytes = b"".join(
        np.asarray(value).tobytes()
        for value in (fit.a_primary, fit.a_specific, fit.threshold, fit.phi)
    )
    return {
        "family": RemoteJobFamily.TWO_TIER.value,
        "library_function": "fast_mlsirm.two_tier_grm.fit_two_tier_grm",
        "converged": bool(fit.converged),
        "final_loglik_change": float(fit.final_loglik_change),
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
    if envelope.family is RemoteJobFamily.EM_M_STEP:
        return execute_em_m_step(payload, unit_seed)
    if envelope.family is RemoteJobFamily.SCORING_PERSON:
        return execute_scoring_person(payload)
    if envelope.family is RemoteJobFamily.SE_DERIVATIVES:
        return execute_se_derivatives(payload)
    if envelope.family is RemoteJobFamily.REGRESSION_CONTRASTS:
        return execute_regression_contrasts(payload)
    if envelope.family is RemoteJobFamily.FIPC:
        return execute_fipc(payload)
    if envelope.family is RemoteJobFamily.FIPC_GROUP_PERSON_SCORE:
        return execute_fipc_group_person_score(payload)
    if envelope.family is RemoteJobFamily.TWO_TIER:
        return execute_two_tier(payload, unit_seed)
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
