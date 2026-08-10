"""Fail-first interoperability contracts for governed JSON artifact writers."""

from __future__ import annotations

import math

import pytest

from fast_mlsirm.io import save_dimensionality_diagnostics
from fast_mlsirm.types import DimensionalityDiagnostics


def test_dimensionality_artifact_rejects_nonfinite_json_number(tmp_path) -> None:
    """A governed artifact must not publish Python's non-standard NaN token."""
    output_dir = tmp_path / "diagnostics"
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "bic": math.nan}],
        best={"n_dims": 2.0, "bic": 123.5},
    )

    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        save_dimensionality_diagnostics(diagnostics, output_dir)

    assert not (output_dir / "dimension_diagnostics.json").exists()
