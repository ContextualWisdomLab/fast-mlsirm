"""Regression coverage for parallel-analysis public data conversion errors."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.parallel_analysis import parallel_analysis


class _FloatBomb:
    """Object-array element that records any attempted real-number coercion."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("OBJECT_ELEMENT_CONVERSION_MUST_NOT_RUN")


def _forbid_native_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace Rust capability discovery with a sentinel and return its call log."""
    core_calls: list[str] = []

    def forbidden_core():
        core_calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    return core_calls


def test_non_numeric_data_conversion_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize invalid storage to a package ValueError before Rust lookup."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = np.array([[object()]], dtype=object)

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_complex_data_is_rejected_before_lossy_projection_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not discard imaginary observed evidence before Horn retention."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="data must be real-valued"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_object_storage_is_rejected_without_element_numeric_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject object storage before caller-defined ``__float__`` can execute."""
    core_calls = _forbid_native_discovery(monkeypatch)
    bomb = _FloatBomb()
    data = np.array([[bomb, bomb], [bomb, bomb]], dtype=object)

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert bomb.calls == 0
    assert core_calls == []
