"""Regression coverage for parallel-analysis public data conversion errors."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.parallel_analysis import parallel_analysis


def test_non_numeric_data_conversion_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize NumPy conversion failure to a package ValueError before Rust lookup."""
    core_calls: list[str] = []

    def forbidden_core():
        core_calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    data = np.array([[object()]], dtype=object)

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []
