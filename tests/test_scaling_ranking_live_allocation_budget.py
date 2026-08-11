"""Fail-first allocation contracts for LSR ranking CSR materialization."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fast_mlsirm.scaling as scaling


def _element_count(shape: object) -> int:
    """Return the number of elements requested by a NumPy allocation shape."""
    if isinstance(shape, (int, np.integer)):
        return int(shape)
    return math.prod(int(value) for value in shape)  # type: ignore[arg-type]


def test_tiny_budget_does_not_allocate_oversized_uint64_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal uint64 capacities must not exceed the declared CSR payload ceiling."""
    budget = 32  # two item ids + two start offsets at eight bytes each
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", budget)
    real_empty = np.empty
    requested_uint64_bytes: list[int] = []

    def recording_empty(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        dtype = kwargs.get("dtype", args[0] if args else np.float64)
        if np.dtype(dtype) == np.dtype(np.uint64):
            requested_uint64_bytes.append(_element_count(shape) * np.dtype(np.uint64).itemsize)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(scaling.np, "empty", recording_empty)

    flat, starts, n = scaling._rankings_to_csr("probe", [[0, 1]], 2)

    assert n == 2
    assert flat.nbytes + starts.nbytes == budget
    assert requested_uint64_bytes
    assert max(requested_uint64_bytes) <= budget


def test_ranking_copy_does_not_create_unbudgeted_uint64_list_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming validation must not add a list→uint64 temporary beside live CSR arrays."""
    real_asarray = np.asarray
    list_uint64_conversions: list[int] = []

    def recording_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        dtype = kwargs.get("dtype", args[0] if args else None)
        if isinstance(value, list) and dtype is not None and np.dtype(dtype) == np.dtype(np.uint64):
            list_uint64_conversions.append(len(value))
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(scaling.np, "asarray", recording_asarray)

    flat, starts, n = scaling._rankings_to_csr("probe", ([0, 1], [1, 2]), 3)

    assert n == 3
    assert np.array_equal(flat, np.array([0, 1, 1, 2], dtype=np.uint64))
    assert np.array_equal(starts, np.array([0, 2, 4], dtype=np.uint64))
    assert list_uint64_conversions == []
