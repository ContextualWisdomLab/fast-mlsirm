"""Validate a saved FIPC artifact before an E research consumer may consume it.

This is deliberately a no-fit preflight. It checks immutable provenance,
input/row identity, focal-prior gates, GPU receipt, and the expected-raw
contract. It never invokes the native fitter or mutates an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = (
    "source_sha",
    "loaded_core",
    "loaded_core_sha256",
    "input_reference_sha256",
    "input_focal_sha256",
    "permutation_sha256",
    "focal_gpu",
    "permuted_gpu",
    "acceptance",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_equal(result: dict[str, Any], path: str, expected: str | None, failures: list[str]) -> None:
    value = result.get(path)
    if expected is not None and value != expected:
        failures.append(f"{path}: expected {expected}, got {value}")


def validate(artifact: Path, expected_source: str | None, expected_core: str | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text())
    failures: list[str] = []
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in payload]
    failures.extend(f"missing top-level field: {key}" for key in missing)
    _check_equal(payload, "source_sha", expected_source, failures)
    _check_equal(payload, "loaded_core_sha256", expected_core, failures)

    acceptance = payload.get("acceptance", {})
    for key in ("responses_fit_eap_expected_raw", "non_unit_focal_prior", "row_order", "gpu_required", "all_pass"):
        if acceptance.get(key) is not True:
            failures.append(f"acceptance.{key} is not true")

    focal = payload.get("focal_gpu", {})
    permuted = payload.get("permuted_gpu", {})
    for name, result in (("focal_gpu", focal), ("permuted_gpu", permuted)):
        for key in ("converged", "termination_reason", "gpu_execution_used", "gpu_backend", "gpu_device_name"):
            if key not in result:
                failures.append(f"{name}.{key} missing")
        if result.get("converged") is not True or result.get("termination_reason") != "tolerance_met":
            failures.append(f"{name}: not converged with tolerance_met")
        if result.get("gpu_execution_used") is not True:
            failures.append(f"{name}: GPU execution was not used")
        if result.get("cpu_fallback_reason") is not None:
            failures.append(f"{name}: CPU fallback recorded: {result['cpu_fallback_reason']}")

    if payload.get("row_order_max_abs_delta") != 0.0:
        failures.append("row_order_max_abs_delta is not exactly zero")
    if "row_identity" not in payload:
        failures.append("row_identity missing; permutation_sha256 alone is insufficient")
    if "expected_raw" not in focal:
        failures.append("focal_gpu.expected_raw missing; no research expected-raw contract is present")

    return {
        "artifact": str(artifact),
        "artifact_sha256": _digest(artifact),
        "source_sha": payload.get("source_sha"),
        "loaded_core_sha256": payload.get("loaded_core_sha256"),
        "gpu_backend": focal.get("gpu_backend"),
        "gpu_device_name": focal.get("gpu_device_name"),
        "row_order_max_abs_delta": payload.get("row_order_max_abs_delta"),
        "expected_raw_present": "expected_raw" in focal,
        "research_consumption_ready": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-source")
    parser.add_argument("--expected-core")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.artifact, args.expected_source, args.expected_core)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["research_consumption_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
