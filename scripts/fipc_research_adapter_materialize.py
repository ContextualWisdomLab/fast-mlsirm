"""Materialize research-consumer-owned FIPC sidecars onto an immutable fit artifact.

This command never fits or invents research values. The research consumer
must provide row identity and expected-raw sidecars; this tool only validates
their hashes/contracts and writes a new derived artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _float64_le(values: list[float]) -> bytes:
    return b"".join(struct.pack("<d", float(value)) for value in values)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def materialize(artifact_path: Path, row_path: Path, expected_raw_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = _load(artifact_path)
    row = _load(row_path)
    expected = _load(expected_raw_path)
    failures: list[str] = []

    if row.get("schema") != "fipc-row-identity/v1":
        failures.append("row_identity.schema must be fipc-row-identity/v1")
    row_ids = row.get("row_ids")
    permutation = row.get("permutation")
    if not isinstance(row_ids, list) or not all(isinstance(value, str) for value in row_ids):
        failures.append("row_identity.row_ids must be a string array")
    if not isinstance(permutation, list) or not all(isinstance(value, int) for value in permutation):
        failures.append("row_identity.permutation must be an integer array")
    if isinstance(permutation, list):
        permutation_bytes = b"".join(struct.pack("<q", value) for value in permutation)
        if row.get("permutation_sha256") != _sha256_bytes(permutation_bytes):
            failures.append("row_identity.permutation_sha256 mismatch")
    if isinstance(row_ids, list) and isinstance(permutation, list):
        if len(row_ids) != len(permutation) or sorted(permutation) != list(range(len(permutation))):
            failures.append("row_identity must contain a complete permutation of row_ids")
    if row.get("source_input_sha256") != artifact.get("input_focal_sha256"):
        failures.append("row_identity.source_input_sha256 does not match artifact input_focal_sha256")

    if expected.get("schema") != "fipc-expected-raw/v1":
        failures.append("expected_raw.schema must be fipc-expected-raw/v1")
    values = expected.get("values")
    if not isinstance(values, list) or not all(isinstance(value, (int, float)) for value in values):
        failures.append("expected_raw.values must be a finite numeric array")
    if isinstance(values, list):
        if any(not isinstance(value, (int, float)) or not (-float("inf") < float(value) < float("inf")) for value in values):
            failures.append("expected_raw.values must be finite")
        if expected.get("values_sha256") != _sha256_bytes(_float64_le(values)):
            failures.append("expected_raw.values_sha256 mismatch")
    if expected.get("source_input_sha256") != artifact.get("input_focal_sha256"):
        failures.append("expected_raw.source_input_sha256 does not match artifact input_focal_sha256")
    if expected.get("parameter_hashes") != {
        key: artifact.get("focal_gpu", {}).get(key + "_sha256")
        for key in ("a_primary", "a_specific", "threshold", "theta_p_eap")
    }:
        failures.append("expected_raw.parameter_hashes do not match focal_gpu output hashes")
    if not isinstance(expected.get("formula_id"), str) or not expected["formula_id"]:
        failures.append("expected_raw.formula_id is required")

    if failures:
        raise ValueError("; ".join(failures))

    derived = dict(artifact)
    derived["row_identity"] = row
    derived["focal_gpu"] = dict(artifact["focal_gpu"])
    derived["focal_gpu"]["expected_raw"] = expected
    output_path.write_text(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "input_artifact_sha256": _sha256_bytes(artifact_path.read_bytes()),
        "output_artifact_sha256": _sha256_bytes(output_path.read_bytes()),
        "row_identity_sha256": _sha256_bytes(_canonical_json(row)),
        "expected_raw_sha256": _sha256_bytes(_canonical_json(expected)),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("row_identity", type=Path)
    parser.add_argument("expected_raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.artifact, args.row_identity, args.expected_raw, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
