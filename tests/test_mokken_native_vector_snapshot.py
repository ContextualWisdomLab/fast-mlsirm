"""Mutation-safety regressions for Mokken native vector marshalling."""

from __future__ import annotations

import numpy as np

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
