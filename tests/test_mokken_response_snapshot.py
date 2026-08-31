"""Ownership regressions for Mokken response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


def test_top_level_int64_ndarray_scores_are_package_owned() -> None:
    """Admitted Rust-bound scores must not alias caller-owned response storage."""
    responses = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    scores, n_persons, n_items = mokken._validated_scores(responses)

    assert (n_persons, n_items) == (3, 2)
    assert scores.dtype == np.int64
    assert scores.flags.c_contiguous
    assert not np.shares_memory(scores, responses)
    assert scores.tolist() == [0, 1, 1, 0, 1, 1]

    responses[0, 0] = 1
    responses[2, 1] = 0

    assert scores.tolist() == [0, 1, 1, 0, 1, 1]


def test_top_level_ndarray_snapshot_replays_cell_budget_after_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy-time growth must keep the package-owned response resource error."""
    responses = np.array([[0, 1], [1, 0]], dtype=np.int64)
    original_array = np.array

    def _grow_then_copy(value: object, *args: object, **kwargs: object) -> np.ndarray:
        if value is responses:
            responses.resize((3, 2), refcheck=False)
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_CELLS", 4)
    monkeypatch.setattr(mokken.np, "array", _grow_then_copy)

    with pytest.raises(ValueError, match=r"responses exceed 4 logical cells"):
        mokken._validated_scores(responses)


def test_top_level_ndarray_rejects_unsupported_dtype_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert unsupported dtype metadata must reject before duplicating storage."""
    responses = np.empty((3, 2), dtype="S32")

    def _unexpected_copy(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("unsupported response storage was copied")

    monkeypatch.setattr(mokken.np, "array", _unexpected_copy)

    with pytest.raises(ValueError, match=r"responses must be a numeric array"):
        mokken._validated_scores(responses)


def test_top_level_ndarray_rejects_invalid_rank_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert rank metadata must reject before allocating package-owned storage."""
    responses = np.zeros(4, dtype=np.int64)

    def _unexpected_copy(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("invalid-rank response storage was copied")

    monkeypatch.setattr(mokken.np, "array", _unexpected_copy)

    with pytest.raises(ValueError, match=r"responses must be a 2-D persons x items array"):
        mokken._validated_scores(responses)
