"""Interoperability contracts for governed JSON artifact writers."""

from __future__ import annotations

import json
import math

import pytest

from fast_mlsirm.io import save_dimensionality_diagnostics
from fast_mlsirm.types import DimensionalityDiagnostics


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_dimensionality_artifact_rejects_nonfinite_json_number(
    tmp_path, nonfinite: float
) -> None:
    """Governed artifacts reject every non-finite JSON numeric extension."""
    output_dir = tmp_path / "diagnostics"
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "criteria": [{"bic": nonfinite}]}],
        best={"n_dims": 2.0, "bic": 123.5},
    )

    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        save_dimensionality_diagnostics(diagnostics, output_dir)

    assert not (output_dir / "dimension_diagnostics.json").exists()


def test_dimensionality_artifact_emits_strict_finite_json(tmp_path) -> None:
    """Finite extreme values remain valid, strict, round-trippable JSON."""
    output_dir = tmp_path / "diagnostics"
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "bic": 1.7976931348623157e308}],
        best={"n_dims": 2.0, "bic": -1.7976931348623157e308},
    )

    save_dimensionality_diagnostics(diagnostics, output_dir)

    artifact = output_dir / "dimension_diagnostics.json"
    text = artifact.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    decoded = json.loads(text)
    assert decoded["candidates"][0]["bic"] == 1.7976931348623157e308
    assert decoded["best"]["bic"] == -1.7976931348623157e308
