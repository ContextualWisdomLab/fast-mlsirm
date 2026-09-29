"""Input immutability and tamper detection for the FIPC research adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
materializer = importlib.import_module("scripts.fipc_research_adapter_materialize")
preflight = importlib.import_module("scripts.fipc_research_adapter_preflight")

_SOURCE = "a" * 40
_CORE = "c" * 64
_INPUT = hashlib.sha256(b"focal-input").hexdigest()
_PARAMS = ("a_primary", "a_specific", "threshold", "theta_p_eap")


def _gpu_result(**extra: object) -> dict[str, object]:
    """Return a converged GPU fit receipt."""
    return {
        "converged": True,
        "termination_reason": "tolerance_met",
        "gpu_execution_used": True,
        "gpu_backend": "metal",
        "gpu_device_name": "synthetic",
        "cpu_fallback_reason": None,
        **extra,
    }


def _sidecars() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return a consistent artifact, row-identity sidecar, and expected-raw sidecar."""
    hashes = {key: hashlib.sha256(key.encode()).hexdigest() for key in _PARAMS}
    artifact = {
        "source_sha": _SOURCE,
        "loaded_core": "fast_mlsirm._core",
        "loaded_core_sha256": _CORE,
        "input_reference_sha256": hashlib.sha256(b"reference").hexdigest(),
        "input_focal_sha256": _INPUT,
        "permutation_sha256": "0" * 64,
        "focal_gpu": _gpu_result(**{key + "_sha256": value for key, value in hashes.items()}),
        "permuted_gpu": _gpu_result(),
        "acceptance": {
            key: True
            for key in (
                "responses_fit_eap_expected_raw",
                "non_unit_focal_prior",
                "row_order",
                "gpu_required",
                "all_pass",
            )
        },
        "row_order_max_abs_delta": 0.0,
    }
    permutation = [2, 0, 1]
    row = {
        "schema": "fipc-row-identity/v1",
        "row_ids": ["r1", "r2", "r3"],
        "permutation": permutation,
        "permutation_sha256": hashlib.sha256(
            b"".join(struct.pack("<q", value) for value in permutation)
        ).hexdigest(),
        "source_input_sha256": _INPUT,
    }
    values = [1.5, 0.25, 3.0]
    expected = {
        "schema": "fipc-expected-raw/v1",
        "values": values,
        "values_sha256": hashlib.sha256(
            b"".join(struct.pack("<d", value) for value in values)
        ).hexdigest(),
        "source_input_sha256": _INPUT,
        "parameter_hashes": hashes,
        "formula_id": "consumer/expected-raw/v1",
    }
    return artifact, row, expected


def _write(path: Path, value: object) -> Path:
    """Write ``value`` as JSON and return ``path``."""
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """Write the three consistent inputs to disk."""
    artifact, row, expected = _sidecars()
    return {
        "artifact": _write(tmp_path / "artifact.json", artifact),
        "row": _write(tmp_path / "row.json", row),
        "expected": _write(tmp_path / "expected.json", expected),
    }


def _derived(paths: dict[str, Path], tmp_path: Path) -> Path:
    """Materialize the valid inputs and return the derived artifact path."""
    output = tmp_path / "derived.json"
    materializer.materialize(paths["artifact"], paths["row"], paths["expected"], output)
    return output


def test_valid_sidecars_materialize_and_pass_preflight_with_pinned_hashes(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    """A consistent bundle is ready, and the receipt pins artifact and sidecar hashes."""
    derived = _derived(paths, tmp_path)
    receipt = preflight.validate(derived, _SOURCE, _CORE)
    assert receipt["failures"] == []
    assert receipt["research_consumption_ready"] is True
    assert receipt["artifact_sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
    payload = json.loads(derived.read_text())
    assert receipt["row_identity_sha256"] == materializer.canonical_sha256(payload["row_identity"])
    assert receipt["expected_raw_sha256"] == materializer.canonical_sha256(
        payload["focal_gpu"]["expected_raw"]
    )


@pytest.mark.parametrize("alias", ["same", "dotted", "symlink", "hardlink"])
def test_materialize_refuses_to_overwrite_the_input_artifact(
    paths: dict[str, Path], alias: str
) -> None:
    """Every path that names the input artifact is refused, leaving its bytes intact."""
    artifact = paths["artifact"]
    before = artifact.read_bytes()
    if alias == "same":
        output = artifact
    elif alias == "dotted":
        output = artifact.parent / "." / artifact.name
    elif alias == "symlink":
        output = artifact.parent / "link.json"
        output.symlink_to(artifact)
    else:
        output = artifact.parent / "hard.json"
        os.link(artifact, output)
    with pytest.raises(ValueError, match="input artifact"):
        materializer.materialize(artifact, paths["row"], paths["expected"], output)
    assert artifact.read_bytes() == before


def _tamper(field: str, payload: dict) -> None:
    """Change exactly one sidecar field in a derived artifact."""
    row = payload["row_identity"]
    expected = payload["focal_gpu"]["expected_raw"]
    if field == "row_id":
        row["row_ids"][0] = "forged"
    elif field == "permutation_sha256":
        row["permutation_sha256"] = "f" * 64
    elif field == "permutation_incomplete":
        row["permutation"][0] = row["permutation"][1]
    elif field == "row_source":
        row["source_input_sha256"] = "e" * 64
    elif field == "expected_value":
        expected["values"][0] += 1.0
    elif field == "expected_nonfinite":
        expected["values"][0] = float("nan")
    elif field == "parameter_hash":
        expected["parameter_hashes"]["theta_p_eap"] = "d" * 64
    elif field == "expected_source":
        expected["source_input_sha256"] = "e" * 64
    else:
        expected["formula_id"] = ""


_TAMPERS = (
    "row_id",
    "permutation_sha256",
    "permutation_incomplete",
    "row_source",
    "expected_value",
    "expected_nonfinite",
    "parameter_hash",
    "expected_source",
    "formula_id",
)


@pytest.mark.parametrize("field", _TAMPERS)
def test_preflight_rejects_post_materialization_tampering(
    paths: dict[str, Path], tmp_path: Path, field: str
) -> None:
    """A single-field edit to a materialized sidecar fails the preflight closed."""
    derived = _derived(paths, tmp_path)
    payload = json.loads(derived.read_text())
    _tamper(field, payload)
    derived.write_text(json.dumps(payload))
    receipt = preflight.validate(derived, _SOURCE, _CORE)
    assert receipt["research_consumption_ready"] is False
    assert receipt["failures"]


def test_row_id_tamper_is_caught_by_a_row_identity_digest(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    """Row IDs carry no self-hash, so the sidecar digest must be pinned at materialization."""
    derived = _derived(paths, tmp_path)
    payload = json.loads(derived.read_text())
    payload["row_identity"]["row_ids"][0] = "forged"
    derived.write_text(json.dumps(payload))
    failures = preflight.validate(derived, _SOURCE, _CORE)["failures"]
    assert any("row_identity" in failure for failure in failures)
