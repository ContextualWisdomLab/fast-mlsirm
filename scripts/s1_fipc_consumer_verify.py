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
# Two anchored items per primary so the anchored primary-loading submatrix is
# full column rank; items 2 and 5 stay free.  Items 0 and 4 also anchor each
# specific block (Cai, 2010, two-tier structure).
ANCHOR = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
FOCAL_MEAN = np.array([0.65, -0.35])
FOCAL_SD = np.array([1.25, 0.8])
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
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if value else None


def anchor_identification(
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    anchor: np.ndarray,
    estimate_specific_vars: bool,
) -> dict[str, object]:
    """Check that the fixed items pin every focal latent moment being estimated.

    Under FIPC the focal-group metric comes only from the fixed items (Kim,
    2006).  The focal primary mean enters the anchored items through
    ``A_anchor @ mu``, so ``mu`` is identified only when the anchored
    primary-loading pattern has full column rank.  When specific variances are
    estimated, each specific block likewise needs an anchored item.
    """
    rank = int(np.linalg.matrix_rank(np.asarray(primary_map, dtype=float)[anchor]))
    specific_ok = not estimate_specific_vars or set(np.unique(specific_map)) <= set(np.unique(specific_map[anchor]))
    return {
        "anchored_primary_rank": rank,
        "anchor_identified": bool(rank == primary_map.shape[1] and specific_ok),
    }


def _shaped(fit: dict, n_persons: int) -> dict[str, np.ndarray]:
    """Reshape finite PyO3 outputs into item- and person-major arrays."""
    shaped = {
        "theta": np.asarray(fit["theta_p_eap"], dtype=np.float64).reshape(n_persons, N_PRIMARY),
        "a_primary": np.asarray(fit["a_primary"], dtype=np.float64).reshape(N_ITEMS, N_PRIMARY),
        "a_specific": np.asarray(fit["a_specific"], dtype=np.float64).reshape(N_ITEMS),
        "threshold": np.asarray(fit["threshold"], dtype=np.float64).reshape(N_ITEMS, N_CAT - 1),
        "primary_mean": np.asarray(fit["primary_mean"], dtype=np.float64),
        "primary_sd": np.asarray(fit["primary_sd"], dtype=np.float64),
    }
    if any(shaped[key].shape != (N_PRIMARY,) for key in ("primary_mean", "primary_sd")):
        raise ValueError("Focal primary moments must have shape (n_primary,).")
    if np.any(shaped["primary_sd"] <= 0):
        raise ValueError("Focal primary SDs must be positive.")
    if any(not np.isfinite(value).all() for value in shaped.values()):
        raise ValueError("Fit item parameters and EAP outputs must be finite.")
    return shaped


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


def call_fipc(
    y: np.ndarray, fixed: dict[str, np.ndarray], device: str = "cpu", *,
    q_primary: int, q_specific: int, **kwargs,
):
    observed = np.ones(y.size, dtype=bool)
    # Pass device only when asked, so bindings without the argument still work.
    extra = {} if device == "cpu" else {"device": device}
    fit = _core.fit_two_tier_grm_fipc(
        np.asarray(y, dtype=np.int64).reshape(-1), observed,
        kwargs.get("primary_map", PRIMARY_MAP).reshape(-1), SPECIFIC_MAP,
        N_PERSONS, N_ITEMS, N_PRIMARY, N_SPECIFIC, N_CAT, ANCHOR,
        fixed["a_primary"], fixed["a_specific"], fixed["threshold"],
        q_primary, q_specific,
        kwargs.get("max_iter", 100), kwargs.get("tol", 1e-5),
        kwargs.get("newton_iter", 5), kwargs.get("ridge", 1e-8),
        kwargs.get("estimate_specific_vars", False),
        **extra,
    )
    # Optional native diagnostics may not be emitted by older bindings, but
    # present numeric outputs cannot be discarded to certify a finite fit.
    for field in (
        "theta_p_sd", "primary_cov", "loglik_trace", "fixed_loglik_trace",
        "fixed_primary_first_moment_trace", "fixed_primary_second_moment_trace",
        "fixed_specific_second_moment_trace", "prior_mean_trace",
        "prior_covariance_trace", "prior_specific_sd_trace",
        "final_loglik_change", "final_param_change",
    ):
        if field in fit and not np.isfinite(np.asarray(fit[field], dtype=np.float64)).all():
            raise ValueError(f"Native diagnostic {field} must be finite.")
    # Check raw call metadata before aggregation can hide malformed values.
    used = fit.get("gpu_execution_used")
    backend = fit.get("gpu_backend")
    if used is not None and type(used) is not bool:
        raise ValueError("GPU execution metadata must be a boolean or null.")
    if backend is not None and not isinstance(backend, str):
        raise ValueError("GPU backend metadata must be a string or null.")
    return fit


def expected_raw(theta: np.ndarray, fit: dict, *, q_specific: int) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(q_specific)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    out = np.zeros(theta.shape[0])
    for i in range(N_ITEMS):
        base = theta @ fit["a_primary"][i]
        if fit["a_specific"][i] != 0:
            nuisance = np.zeros(theta.shape[0])
            for node, weight in zip(nodes, weights):
                nuisance += weight * np.sum(
                    1.0 / (1.0 + np.exp(-(base[:, None] + fit["a_specific"][i] * node + fit["threshold"][i]))), axis=1
                )
            out += nuisance
        else:
            cum = 1.0 / (1.0 + np.exp(-(base[:, None] + fit["threshold"][i])))
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
        "--device", choices=("cpu", "gpu", "auto"), default="cpu",
        help="FIPC execution device; a gpu receipt fails if the binding falls back to CPU",
    )
    parser.add_argument(
        "--consumer-sha",
        default=os.environ.get("FIPC_CONSUMER_SHA") or _git_sha(),
        help="consumer checkout revision (or FIPC_CONSUMER_SHA; defaults to git HEAD)",
    )
    parser.add_argument("--q-primary", type=int, required=True)
    parser.add_argument("--q-specific", type=int, required=True)
    parser.add_argument("--q-expected-raw", type=int, required=True)
    return parser.parse_args()


GATES = (
    "responses_fit_eap_expected_raw", "non_unit_focal_prior", "row_order",
    "anchor_rows_fixed", "convergence_failure", "orthogonal_specific_prior_fixed",
    "wrong_model_reject",
)


def _fipc_gates(
    results: dict[str, object], device: str, *,
    q_primary: int, q_specific: int, q_expected_raw: int,
) -> None:
    reference_y = simulate(SEED, np.zeros(N_PRIMARY), np.ones(N_PRIMARY))
    reference = _core.fit_two_tier_grm(
        reference_y.reshape(-1), np.ones(reference_y.size, dtype=bool),
        PRIMARY_MAP.reshape(-1), SPECIFIC_MAP, N_PERSONS, N_ITEMS, N_PRIMARY,
        N_SPECIFIC, N_CAT, q_primary, q_specific, 100, 1e-5, 1, SEED,
    )
    if reference.get("converged") is not True:
        raise ValueError("Reference calibration must report convergence before focal fitting.")
    fixed = {k: np.asarray(reference[k], dtype=np.float64) for k in ("a_primary", "a_specific", "threshold")}
    focal_y = simulate(SEED + 1, FOCAL_MEAN, FOCAL_SD)
    fit = call_fipc(focal_y, fixed, device, q_primary=q_primary, q_specific=q_specific)
    results["gpu_execution_used"] = fit.get("gpu_execution_used")
    results["gpu_backend"] = fit.get("gpu_backend")
    shaped = _shaped(fit, N_PERSONS)
    fit_converged = fit.get("converged") is True
    results["responses_fit_eap_expected_raw"] = bool(fit_converged and np.isfinite(shaped["theta"]).all() and np.isfinite(expected_raw(shaped["theta"], shaped, q_specific=q_expected_raw)).all())
    # One seeded replicate: these are recovery errors, not Monte Carlo bias.
    results["focal_primary_mean_error"] = (np.asarray(fit["primary_mean"], dtype=np.float64) - FOCAL_MEAN).tolist()
    results["focal_primary_sd_error"] = (np.asarray(fit["primary_sd"], dtype=np.float64) - FOCAL_SD).tolist()
    free = ~ANCHOR
    free_error = np.concatenate((
        (shaped["a_primary"] - TRUE_A_P)[free][PRIMARY_MAP[free]],
        (shaped["threshold"] - TRUE_D)[free].reshape(-1),
    ))
    results["free_item_param_rmse"] = float(np.sqrt(np.mean(np.square(free_error))))
    # The focal prior must move off the unit reference in location and in
    # scale; a scale change can go either way (the fixture's truth is 1.25, 0.8).
    results["non_unit_focal_prior"] = bool(np.max(np.abs(np.asarray(fit["primary_mean"]))) > 0.1 and np.max(np.abs(np.asarray(fit["primary_sd"]) - 1.0)) > 0.01)
    perm = np.random.default_rng(17).permutation(N_PERSONS)
    perm_fit = call_fipc(focal_y[perm], fixed, device, q_primary=q_primary, q_specific=q_specific)
    perm_shaped = _shaped(perm_fit, N_PERSONS)
    row_error = shaped["theta"][perm] - perm_shaped["theta"]
    results["row_order_max_abs"] = float(np.max(np.abs(row_error)))
    results["row_order_rmse"] = float(np.sqrt(np.mean(np.square(row_error))))
    refits_converged = fit.get("converged") is True and perm_fit.get("converged") is True
    results["row_order"] = bool(refits_converged and np.allclose(row_error, 0.0, atol=ROW_ORDER_ATOL, rtol=ROW_ORDER_RTOL))
    if not refits_converged:
        results["row_order_diagnosis"] = "fit_error"
        results["fipc_fit_error"] = "Both row-order fits must report convergence."
    else:
        results["row_order_diagnosis"] = "pass" if results["row_order"] else "refit_order_dependence"
    fail = call_fipc(focal_y, fixed, device, q_primary=q_primary, q_specific=q_specific, max_iter=1)
    # A budget-limited negative must still return valid finite item/EAP arrays;
    # metadata alone cannot distinguish nonconvergence from corrupt output.
    fail_shaped = _shaped(fail, N_PERSONS)
    results["anchor_rows_fixed"] = all(
        np.array_equal(candidate[key][ANCHOR], fixed[key].reshape(candidate[key].shape)[ANCHOR])
        for candidate in (shaped, perm_shaped, fail_shaped)
        for key in ("a_primary", "a_specific", "threshold")
    )
    results["convergence_failure"] = bool(
        fail.get("converged") is False
        and fail.get("termination_reason") in {"max_iter_reached", "step_limited"}
    )
    if device in {"gpu", "auto"}:
        # GPU metadata describes all valid focal/refit/budget calls, including
        # auto's allowed CPU fallback. The reference calibration has no device
        # argument; the invalid-map probe rejects before numerical execution.
        results["gpu_execution_used"] = all(
            result.get("gpu_execution_used") is True for result in (fit, perm_fit, fail)
        )
        if not results["gpu_execution_used"]:
            results["gpu_backend"] = None
    results["orthogonal_specific_prior_fixed"] = all(
        np.array_equal(np.asarray(candidate["specific_sd"]), np.ones(N_SPECIFIC))
        for candidate in (fit, perm_fit, fail)
    )
    bad_map = PRIMARY_MAP.copy()
    bad_map[:, 1] = False
    try:
        call_fipc(focal_y, fixed, device, q_primary=q_primary, q_specific=q_specific, primary_map=bad_map, max_iter=10)
        results["wrong_model_reject"] = False
    except ValueError as exc:
        results["wrong_model_reject"] = str(exc).startswith(
            "primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required"
        )


def build_receipt(
    consumer_sha: str | None, build_source_sha: str | None, device: str = "cpu", *,
    q_primary: int, q_specific: int, q_expected_raw: int,
) -> dict[str, object]:
    if not isinstance(device, str) or device not in {"cpu", "gpu", "auto"}:
        raise ValueError("Device must be one of cpu, gpu, auto")
    for revision, label in ((consumer_sha, "Consumer revision"), (build_source_sha, "Build source revision")):
        if revision is not None and (
            not isinstance(revision, str) or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise ValueError(f"{label} must contain 40 lowercase hexadecimal characters")
    extension = Path(_core.__file__).resolve()
    results: dict[str, object] = {
        "sha": consumer_sha,
        "build_source_sha": build_source_sha,
        "build_source_sha_present": build_source_sha is not None,
        # build_source_sha is a caller label, not bound to the binary digest.
        # Only an immutable release/attestation manifest can bind the two.
        "build_source_sha_authority": "caller_asserted",
        "evidence_class": "synthetic_diagnostic",
        "device_requested": device,
        "gpu_execution_used": None,
        "gpu_backend": None,
        "loaded_extension": str(extension),
        "loaded_extension_sha256": _sha256(extension),
        "expected_raw_range": None,
        "row_order_max_abs": None,
        "row_order_rmse": None,
        "focal_primary_mean_error": None,
        "focal_primary_sd_error": None,
        "free_item_param_rmse": None,
    }
    results.update(dict.fromkeys(GATES, False))
    results.update(anchor_identification(PRIMARY_MAP, SPECIFIC_MAP, ANCHOR, estimate_specific_vars=False))
    # Row-order drift is uninterpretable when the focal metric is not
    # identified, so fail closed before fitting anything.
    if not results["anchor_identified"]:
        results["row_order_diagnosis"] = "not_identified"
    # The FIPC binding ships separately from this receipt; say so rather than
    # reporting a TypeError as a failed fit.
    elif not hasattr(_core, "fit_two_tier_grm_fipc"):
        results["row_order_diagnosis"] = "binding_unavailable"
    else:
        try:
            _fipc_gates(results, device, q_primary=q_primary, q_specific=q_specific, q_expected_raw=q_expected_raw)
        except Exception as exc:
            results.update(dict.fromkeys(GATES, False))
            results["gpu_execution_used"] = None
            results["gpu_backend"] = None
            results["row_order_max_abs"] = None
            results["row_order_rmse"] = None
            results["focal_primary_mean_error"] = None
            results["focal_primary_sd_error"] = None
            results["free_item_param_rmse"] = None
            results["row_order_diagnosis"] = "fit_error"
            results["fipc_fit_error"] = f"{type(exc).__name__}: {exc}"
    if results["gpu_backend"] is not None and not isinstance(results["gpu_backend"], str):
        results["gpu_backend"] = None
        results["gpu_execution_used"] = None
        results.update(dict.fromkeys(GATES, False))
        results["row_order_diagnosis"] = "fit_error"
        results["fipc_fit_error"] = "GPU backend metadata must be a string or null."
    invalid_metrics = False
    for key in (
        "row_order_max_abs", "row_order_rmse", "focal_primary_mean_error",
        "focal_primary_sd_error", "free_item_param_rmse",
    ):
        if results[key] is not None and not np.isfinite(results[key]).all():
            results[key] = None
            invalid_metrics = True
    if invalid_metrics:
        results.update(dict.fromkeys(GATES, False))
        results["row_order_diagnosis"] = "fit_error"
        results["fipc_fit_error"] = "The fit did not produce valid numerical results."
    gpu_ok = device != "gpu" or results["gpu_execution_used"] is True
    results["all_pass"] = bool(results["anchor_identified"]) and gpu_ok and all(results[key] is True for key in GATES)
    return results


def main() -> None:
    args = _arguments()
    try:
        receipt = build_receipt(
            args.consumer_sha, args.build_source_sha, args.device,
            q_primary=args.q_primary, q_specific=args.q_specific,
            q_expected_raw=args.q_expected_raw,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
