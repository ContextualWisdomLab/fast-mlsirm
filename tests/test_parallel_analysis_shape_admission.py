"""Fail-first tests for parallel-analysis minimum-shape admission ordering."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm.parallel_analysis import parallel_analysis


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((20_000_000, 1), r"^parallel analysis needs n_items >= 2$"),
        ((2, 10_000_000), r"^parallel analysis needs n_persons >= 3$"),
    ],
)
def test_undersized_exact_ndarray_fails_before_value_scan_dense_or_core(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    message: str,
) -> None:
    """Known minimum-shape violations do not scan/copy a 20M-cell broadcast view."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    fitstats = importlib.import_module("fast_mlsirm.fitstats")
    value_scan_calls: list[bool] = []
    dense_calls: list[bool] = []
    discovery_calls: list[bool] = []

    def reject_value_scan(_value: np.ndarray) -> None:
        value_scan_calls.append(True)
        raise AssertionError("value scan executed before minimum-shape validation")

    def reject_dense_conversion(_raw: np.ndarray) -> np.ndarray:
        dense_calls.append(True)
        raise AssertionError("dense conversion executed before minimum-shape validation")

    def reject_core_discovery() -> object:
        discovery_calls.append(True)
        raise AssertionError("native core discovered before minimum-shape validation")

    monkeypatch.setattr(module, "_validate_real_array_storage", reject_value_scan)
    monkeypatch.setattr(module, "_lossless_float64_matrix", reject_dense_conversion)
    monkeypatch.setattr(fitstats, "_core_module", reject_core_discovery)

    data = np.broadcast_to(np.array(0.0, dtype=np.float64), shape)

    with pytest.raises(ValueError, match=message):
        parallel_analysis(data, n_iterations=1)

    assert value_scan_calls == []
    assert dense_calls == []
    assert discovery_calls == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([[object(), object()], [object(), object()]], r"^parallel analysis needs n_persons >= 3$"),
        ([[object()], [object()], [object()]], r"^parallel analysis needs n_items >= 2$"),
    ],
)
def test_undersized_builtin_shape_fails_before_scalar_admission(
    monkeypatch: pytest.MonkeyPatch,
    data: list[list[object]],
    message: str,
) -> None:
    """Built-in row metadata establishes an invalid design before scalar validation."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    scalar_calls: list[bool] = []

    def reject_scalar(_value: object) -> None:
        scalar_calls.append(True)
        raise AssertionError("scalar admission executed before minimum-shape validation")

    monkeypatch.setattr(module, "_validate_trusted_real_scalar", reject_scalar)

    with pytest.raises(ValueError, match=message):
        parallel_analysis(data, n_iterations=1)

    assert scalar_calls == []


def test_minimum_valid_three_by_two_shape_still_reaches_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite 3x2 matrix remains inside the public Rust-owned numeric path."""
    fitstats = importlib.import_module("fast_mlsirm.fitstats")
    calls: list[tuple[int, int]] = []

    class Core:
        @staticmethod
        def parallel_analysis(
            data: np.ndarray,
            n_persons: int,
            n_items: int,
            n_iterations: int,
            centile: int,
            seed: int,
        ) -> dict[str, object]:
            del data, n_iterations, centile, seed
            calls.append((n_persons, n_items))
            return {
                "retained": 0,
                "eigenvalues": [1.0, 1.0],
                "random_eigenvalues": [1.0, 1.0],
                "bias": [0.0, 0.0],
                "adjusted_eigenvalues": [1.0, 1.0],
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    result = parallel_analysis(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        n_iterations=1,
    )

    assert calls == [(3, 2)]
    assert result.retained == 0
