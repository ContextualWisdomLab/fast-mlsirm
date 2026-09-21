"""S1 consumer verification for the two-tier GRM FIPC binding.

This is a consumer diagnostic, not an optimizer fixture.  In particular, a
failing row-order check is reported with its numerical discrepancy and is not
made green by changing the tolerance or by changing the fitting path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from fast_mlsirm import _core


N_PERSONS = 180
N_ITEMS = 6
N_PRIMARY = 2
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260921
ROW_ORDER_ATOL = 1e-10
ROW_ORDER_RTOL = 1e-10
PRIMARY_MAP = np.array(
    [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=bool
)
SPECIFIC_MAP = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
ANCHOR = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
TRUE_A_P = np.array([[1.4, 0], [1.1, 0], [1.0, 0], [0, 1.2], [0, 0.9], [0, 1.1]])
TRUE_A_S = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 0.9])
TRUE_D = np.array(
    [[1.2, 0.0, -1.2], [1.0, -0.1, -1.3], [1.3, 0.2, -1.0],
     [1.1, 0.1, -1.1], [0.9, -0.2, -1.4], [1.2, 0.0, -1.2]]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    """Return the checked-out consumer source revision when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if value else None


def _provenance() -> dict[str, object]:
    extension = Path(_core.__file__).resolve()
    # Do not infer build_source_sha from the consumer checkout: a stale shared
    # wheel can be loaded by a fresh checkout.  The build pipeline must pass it
    # explicitly, and the distinction remains visible in the JSON receipt.
    build_source_sha = os.environ.get("FIPC_BUILD_SOURCE_SHA") or None
    consumer_sha = os.environ.get("FIPC_CONSUMER_SHA") or _git_sha()
    return {
        "sha": consumer_sha,
        "build_source_sha": build_source_sha,
        "build_source_sha_present": build_source_sha is not None,
        "loaded_extension": str(extension),
        "loaded_extension_sha256": _sha256(extension),
    }


def simulate(seed: int, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z0, z1 = rng.normal(size=(2, N_PERSONS))
    theta = np.column_stack((mean[0] + scale[0] * z0, mean[1] + scale[1] * z1))
    theta_s = rng.normal(size=(N_PERSONS, N_SPECIFIC))
    y = np.empty((N_PERSONS, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = TRUE_A_P[i] @ theta.T + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i])))
        probs = np.concatenate((1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]), axis=1)
        y[:, i] = (rng.random(N_PERSONS)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y


def call_fipc(y: np.ndarray, fixed: dict[str, np.ndarray], **kwargs):
    observed = np.ones(y.size, dtype=bool)
    return _core.fit_two_tier_grm_fipc(
        np.asarray(y, dtype=np.int64).reshape(-1), observed,
        PRIMARY_MAP.reshape(-1), SPECIFIC_MAP,
        N_PERSONS, N_ITEMS, N_PRIMARY, N_SPECIFIC, N_CAT, ANCHOR,
        fixed["a_primary"], fixed["a_specific"], fixed["threshold"],
        kwargs.get("q_primary", 7), kwargs.get("q_specific", 7),
        kwargs.get("max_iter", 100), kwargs.get("tol", 1e-5),
        kwargs.get("newton_iter", 5), kwargs.get("ridge", 1e-8),
        kwargs.get("estimate_specific_vars", False),
    )


def expected_raw(theta: np.ndarray, fit: dict) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(31)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    out = np.zeros(theta.shape[0])
    for i in range(N_ITEMS):
        base = theta @ np.asarray(fit["a_primary"])[i].reshape(N_PRIMARY)
        if np.asarray(fit["a_specific"])[i] != 0:
            nuisance = np.zeros(theta.shape[0])
            for node, weight in zip(nodes, weights):
                nuisance += weight * np.sum(
                    1.0 / (1.0 + np.exp(-(base[:, None] + np.asarray(fit["a_specific"])[i] * node + np.asarray(fit["threshold"])[i]))), axis=1
                )
            out += nuisance
        else:
            cum = 1.0 / (1.0 + np.exp(-(base[:, None] + np.asarray(fit["threshold"])[i])))
            out += cum.sum(axis=1)
    return out


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-source-sha",
        default=os.environ.get("FIPC_BUILD_SOURCE_SHA"),
        help="source revision used to build the loaded extension (or FIPC_BUILD_SOURCE_SHA)",
    )
    parser.add_argument(
        "--consumer-sha",
        default=os.environ.get("FIPC_CONSUMER_SHA") or _git_sha(),
        help="consumer checkout revision (or FIPC_CONSUMER_SHA; defaults to git HEAD)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    results: dict[str, object] = {
        "sha": args.consumer_sha,
        "build_source_sha": args.build_source_sha,
        "build_source_sha_present": args.build_source_sha is not None,
    }
    extension = Path(_core.__file__).resolve()
    results["loaded_extension"] = str(extension)
    results["loaded_extension_sha256"] = _sha256(extension)
    reference_y = simulate(SEED, np.zeros(N_PRIMARY), np.ones(N_PRIMARY))
    try:
        reference = _core.fit_two_tier_grm(
            reference_y.reshape(-1), np.ones(reference_y.size, dtype=bool),
            PRIMARY_MAP.reshape(-1), SPECIFIC_MAP, N_PERSONS, N_ITEMS, N_PRIMARY,
            N_SPECIFIC, N_CAT, 7, 7, 100, 1e-5, 1, SEED,
            e_step_n_chunks=1, e_step_n_threads=1,
        )
        fixed = {k: np.asarray(reference[k], dtype=np.float64) for k in ("a_primary", "a_specific", "threshold")}
        focal_y = simulate(SEED + 1, np.array([0.65, -0.35]), np.array([1.25, 0.8]))
        fit = call_fipc(focal_y, fixed)
        results["responses_fit_eap_expected_raw"] = bool(fit["converged"] and np.isfinite(fit["theta_p_eap"]).all() and np.isfinite(expected_raw(np.asarray(fit["theta_p_eap"]), fit)).all())
        results["non_unit_focal_prior"] = bool(np.max(np.abs(np.asarray(fit["primary_mean"])) > 0.1) and np.max(np.asarray(fit["primary_sd"])) > 1.01)
        perm = np.random.default_rng(17).permutation(N_PERSONS)
        perm_fit = call_fipc(focal_y[perm], fixed)
        row_error = np.asarray(fit["theta_p_eap"])[perm] - np.asarray(perm_fit["theta_p_eap"])
        results["row_order_max_abs"] = float(np.max(np.abs(row_error)))
        results["row_order_rmse"] = float(np.sqrt(np.mean(np.square(row_error))))
        results["row_order"] = bool(np.allclose(row_error, 0.0, atol=ROW_ORDER_ATOL, rtol=ROW_ORDER_RTOL))
        results["row_order_diagnosis"] = "pass" if results["row_order"] else "refit_order_dependence"
        results["anchor_rows_fixed"] = bool(np.array_equal(np.asarray(fit["a_primary"])[ANCHOR], fixed["a_primary"].reshape(N_ITEMS, N_PRIMARY)[ANCHOR]) and np.array_equal(np.asarray(fit["threshold"])[ANCHOR], fixed["threshold"].reshape(N_ITEMS, N_CAT - 1)[ANCHOR]))
        fail = call_fipc(focal_y, fixed, max_iter=1)
        results["convergence_failure"] = bool(not fail["converged"] and fail["termination_reason"] == "max_iter_reached")
        results["orthogonal_specific_prior_fixed"] = bool(np.array_equal(np.asarray(fit["specific_sd"]), np.ones(N_SPECIFIC)))
    except Exception as exc:
        results["fipc_fit_error"] = f"{type(exc).__name__}: {exc}"
        for key in ("responses_fit_eap_expected_raw", "non_unit_focal_prior", "row_order", "anchor_rows_fixed", "convergence_failure", "orthogonal_specific_prior_fixed"):
            results[key] = False
        results["row_order_max_abs"] = None
        results["row_order_rmse"] = None
        results["row_order_diagnosis"] = "fit_error"
    try:
        bad_map = PRIMARY_MAP.copy(); bad_map[:, 1] = False
        _core.fit_two_tier_grm_fipc(focal_y.reshape(-1), np.ones(focal_y.size, dtype=bool), bad_map.reshape(-1), SPECIFIC_MAP, N_PERSONS, N_ITEMS, N_PRIMARY, N_SPECIFIC, N_CAT, ANCHOR, fixed["a_primary"], fixed["a_specific"], fixed["threshold"], 7, 7, 10, 1e-5, 5, 1e-8, False)
        results["wrong_model_reject"] = False
    except Exception as exc:
        results["wrong_model_reject"] = "primary" in str(exc) or "map" in str(exc)
    results["expected_raw_range"] = None
    results["all_pass"] = all(v is True for k, v in results.items() if k not in {"sha", "build_source_sha", "build_source_sha_present", "loaded_extension", "loaded_extension_sha256", "expected_raw_range", "row_order_max_abs", "row_order_rmse", "row_order_diagnosis"})
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
