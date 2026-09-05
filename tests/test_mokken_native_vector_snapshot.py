"""Mutation-safety regressions for Mokken native vector marshalling."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from fast_mlsirm import mokken


def test_native_float_vector_snapshots_before_numpy_conversion(monkeypatch) -> None:
    """Provider mutation at the float-vector seam cannot redefine evidence."""
    source = [0.4, 0.4]
    original_asarray = np.asarray
    seam_calls = 0

    def mutating_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal seam_calls
        seam_calls += 1
        source[0] = 0.9
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(mokken.np, "asarray", mutating_asarray)

    result = mokken._native_float_vector(source, 2)

    assert seam_calls == 1
    assert source == [0.9, 0.4]
    assert result.tolist() == [0.4, 0.4]


def test_native_scale_vector_snapshots_before_numpy_conversion(monkeypatch) -> None:
    """Provider mutation at the AISP-vector seam cannot redefine evidence."""
    source = [1, 1]
    original_asarray = np.asarray
    seam_calls = 0

    def mutating_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal seam_calls
        seam_calls += 1
        source[:] = [0, 0]
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(mokken.np, "asarray", mutating_asarray)

    result = mokken._native_scale_vector(source, 2)

    assert seam_calls == 1
    assert source == [0, 0]
    assert result.tolist() == [1, 1]


def _resize_after_first_length_observation(
    monkeypatch: pytest.MonkeyPatch,
    source: list[object],
    appended: object,
) -> None:
    """Grow one exact list after its admitted length has been observed once."""
    observed_source = False

    def resizing_len(value: object) -> int:
        nonlocal observed_source
        size = builtins.len(value)
        if value is source and not observed_source:
            observed_source = True
            source.append(appended)
        return size

    monkeypatch.setattr(mokken, "len", resizing_len, raising=False)


def test_native_float_vector_replays_snapshot_cardinality_after_copy(monkeypatch) -> None:
    """Growth between source-length admission and copy must fail closed."""
    source: list[object] = [0.4, 0.4]
    _resize_after_first_length_observation(monkeypatch, source, 0.8)

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken._native_float_vector(source, 2)

    assert source == [0.4, 0.4, 0.8]


def test_native_scale_vector_replays_snapshot_cardinality_after_copy(monkeypatch) -> None:
    """AISP growth between source-length admission and copy must fail closed."""
    source: list[object] = [1, 1]
    _resize_after_first_length_observation(monkeypatch, source, 0)

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken._native_scale_vector(source, 2)

    assert source == [1, 1, 0]
