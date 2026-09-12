"""Pre-copy admission regressions for nested ndarray response rows."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


def _reject_outer_snapshot(*args: object, **kwargs: object) -> object:
    """Fail if an already-inadmissible nested row reaches outer-list copying."""
    raise AssertionError("inadmissible nested ndarray reached outer-list snapshot")


def test_nested_ndarray_rejects_unsupported_dtype_before_outer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert row dtype metadata must reject before duplicating the outer list."""
    responses = [
        np.array([b"0", b"1"], dtype="S32"),
        np.array([b"1", b"0"], dtype="S32"),
    ]
    monkeypatch.setattr(mokken, "islice", _reject_outer_snapshot)

    with pytest.raises(ValueError, match=r"responses must be a numeric array"):
        mokken._validated_scores(responses)


def test_nested_ndarray_rejects_complex_before_outer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert complex storage must reject before duplicating the outer list."""
    responses = [
        np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
    ]
    monkeypatch.setattr(mokken, "islice", _reject_outer_snapshot)

    with pytest.raises(ValueError, match=r"responses must be real-valued"):
        mokken._validated_scores(responses)


def test_nested_ndarray_rejects_storage_budget_before_outer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert row bytes over the snapshot ceiling must reject before list copying."""
    responses = [
        np.array([0, 1], dtype=np.int64),
        np.array([1, 0], dtype=np.int64),
    ]
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES", 8)
    monkeypatch.setattr(mokken, "islice", _reject_outer_snapshot)

    with pytest.raises(ValueError, match=r"responses exceed .* logical cells"):
        mokken._validated_scores(responses)
