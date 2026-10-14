"""Fail-first tests for parallel-analysis complete-data admission ordering."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm.parallel_analysis import parallel_analysis


def _reject_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make native-core discovery observable and forbidden for invalid data."""
    import fast_mlsirm.fitstats as fitstats

    calls: list[bool] = []

    def discover_core() -> object:
        calls.append(True)
        raise AssertionError("native core discovered before finite-data validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    return calls


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_exact_ndarray_fails_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    """Known-invalid exact NumPy evidence is rejected before contiguous marshalling."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    discovery_calls = _reject_core_discovery(monkeypatch)
    dense_calls: list[bool] = []

    def reject_dense_conversion(raw: np.ndarray) -> np.ndarray:
        dense_calls.append(True)
        raise AssertionError("dense float64 conversion executed before finite-data validation")

    monkeypatch.setattr(module, "_lossless_float64_matrix", reject_dense_conversion)
    data = np.array(
        [[0.1, 1.0], [0.4, bad_value], [0.9, -0.2]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=r"^data must be finite"):
        parallel_analysis(data, n_iterations=1)

    assert dense_calls == []
    assert discovery_calls == []


@pytest.mark.parametrize("bad_value", [float("nan"), np.float32(np.inf)])
def test_nonfinite_builtin_leaf_fails_before_numpy_materialization_or_core(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: object,
) -> None:
    """Trusted scalar leaves still replay complete-data finiteness before NumPy."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    discovery_calls = _reject_core_discovery(monkeypatch)
    asarray_calls: list[bool] = []
    original_asarray = module.np.asarray

    def reject_asarray(*args: object, **kwargs: object) -> np.ndarray:
        asarray_calls.append(True)
        raise AssertionError("np.asarray executed before finite-data validation")

    monkeypatch.setattr(module.np, "asarray", reject_asarray)
    data = [[0.1, 1.0], [0.4, bad_value], [0.9, -0.2]]

    try:
        with pytest.raises(ValueError, match=r"^data must be finite"):
            parallel_analysis(data, n_iterations=1)
    finally:
        monkeypatch.setattr(module.np, "asarray", original_asarray)

    assert asarray_calls == []
    assert discovery_calls == []
