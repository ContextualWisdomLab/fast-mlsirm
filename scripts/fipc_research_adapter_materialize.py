"""Materialize research-consumer-owned FIPC sidecars onto an immutable fit artifact.

This command never fits or invents research values. The research consumer
must provide row identity and expected-raw sidecars; this tool only validates
their hashes/contracts and writes a new derived artifact. The same
``sidecar_failures`` contract is re-run by the preflight on the derived
artifact, so a post-materialization edit cannot pass as consumer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts._bounded_json import parse_json_bounded
except ModuleNotFoundError:
    from _bounded_json import parse_json_bounded


PARAMETER_KEYS = ("a_primary", "a_specific", "threshold", "theta_p_eap")
ADAPTER_KEY = "research_adapter"
_INT64 = (-(2**63), 2**63 - 1)


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of ``value`` as sorted, compact, ASCII JSON."""
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_int(value: Any) -> bool:
    return type(value) is int and _INT64[0] <= value <= _INT64[1]


def _is_finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _row_failures(artifact: dict[str, Any], row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["row_identity must be a JSON object"]
    failures = []
    if row.get("schema") != "fipc-row-identity/v1":
        failures.append("row_identity.schema must be fipc-row-identity/v1")
    row_ids = row.get("row_ids")
    permutation = row.get("permutation")
    ids_ok = isinstance(row_ids, list) and all(isinstance(value, str) for value in row_ids)
    perm_ok = isinstance(permutation, list) and all(_is_int(value) for value in permutation)
    if not ids_ok:
        failures.append("row_identity.row_ids must be a string array")
    if not perm_ok:
        failures.append("row_identity.permutation must be an int64 array")
    else:
        encoded = b"".join(struct.pack("<q", value) for value in permutation)
        if row.get("permutation_sha256") != _sha256(encoded):
            failures.append("row_identity.permutation_sha256 mismatch")
    if ids_ok and perm_ok and (
        len(row_ids) != len(permutation) or sorted(permutation) != list(range(len(permutation)))
    ):
        failures.append("row_identity must contain a complete permutation of row_ids")
    if row.get("source_input_sha256") != artifact.get("input_focal_sha256"):
        failures.append("row_identity.source_input_sha256 does not match artifact input_focal_sha256")
    return failures


def _expected_failures(artifact: dict[str, Any], expected: Any) -> list[str]:
    if not isinstance(expected, dict):
        return ["expected_raw must be a JSON object"]
    failures = []
    if expected.get("schema") != "fipc-expected-raw/v1":
        failures.append("expected_raw.schema must be fipc-expected-raw/v1")
    values = expected.get("values")
    if not isinstance(values, list) or not all(_is_finite(value) for value in values):
        failures.append("expected_raw.values must be a finite numeric array")
    else:
        encoded = b"".join(struct.pack("<d", float(value)) for value in values)
        if expected.get("values_sha256") != _sha256(encoded):
            failures.append("expected_raw.values_sha256 mismatch")
    if expected.get("source_input_sha256") != artifact.get("input_focal_sha256"):
        failures.append("expected_raw.source_input_sha256 does not match artifact input_focal_sha256")
    focal = artifact.get("focal_gpu")
    focal = focal if isinstance(focal, dict) else {}
    output_hashes = {key: focal.get(key + "_sha256") for key in PARAMETER_KEYS}
    if None in output_hashes.values() or expected.get("parameter_hashes") != output_hashes:
        failures.append("expected_raw.parameter_hashes do not match focal_gpu output hashes")
    if not isinstance(expected.get("formula_id"), str) or not expected["formula_id"]:
        failures.append("expected_raw.formula_id is required")
    return failures


def sidecar_failures(artifact: dict[str, Any], row: Any, expected: Any) -> list[str]:
    """Return every violation of the row-identity and expected-raw sidecar contract."""
    return _row_failures(artifact, row) + _expected_failures(artifact, expected)


def _refuse_input_alias(inputs: list[os.stat_result], output_path: Path) -> None:
    try:
        output_status = output_path.stat()
    except FileNotFoundError:
        return
    identities = {(status.st_dev, status.st_ino) for status in inputs}
    if (output_status.st_dev, output_status.st_ino) in identities:
        raise ValueError(f"{output_path}: output names an input artifact or sidecar; refusing to overwrite it")


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load(path: Path) -> tuple[dict[str, Any], bytes, os.stat_result]:
    with path.open("rb") as handle:
        status = os.fstat(handle.fileno())
        content = handle.read()
    value = parse_json_bounded(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value, content, status


def materialize(artifact_path: Path, row_path: Path, expected_raw_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate the sidecars and atomically write a new derived artifact."""
    artifact, artifact_bytes, artifact_status = _load(artifact_path)
    row, _, row_status = _load(row_path)
    expected, _, expected_status = _load(expected_raw_path)
    inputs = [artifact_status, row_status, expected_status]
    _refuse_input_alias(inputs, output_path)
    failures = sidecar_failures(artifact, row, expected)
    if ADAPTER_KEY in artifact or "row_identity" in artifact:
        failures.append("artifact already carries research sidecars; materialize from the original fit artifact")
    if failures:
        raise ValueError("; ".join(failures))

    receipt = {
        "input_artifact_sha256": _sha256(artifact_bytes),
        "row_identity_sha256": canonical_sha256(row),
        "expected_raw_sha256": canonical_sha256(expected),
    }
    derived = dict(artifact)
    derived["row_identity"] = row
    derived["focal_gpu"] = {**artifact["focal_gpu"], "expected_raw": expected}
    derived[ADAPTER_KEY] = receipt
    content = (json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _refuse_input_alias(inputs, output_path)
    _atomic_write(output_path, content)
    return {**receipt, "output_artifact_sha256": _sha256(content), "output": str(output_path)}


def main() -> int:
    """Run the materializer from the command line."""
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
