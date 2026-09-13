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


def test_top_level_ndarray_snapshot_replays_admitted_shape_after_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-cardinality reshape at the copy seam must not redefine matrix identity."""
    responses = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    original_array = np.array

    def _reshape_then_copy(value: object, *args: object, **kwargs: object) -> np.ndarray:
        if value is responses:
            responses.shape = (2, 3)
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(mokken.np, "array", _reshape_then_copy)

    with pytest.raises(ValueError, match=r"responses must be a 2-D persons x items array"):
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


def test_builtin_sequence_is_sealed_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller mutation at the dense-conversion seam must not redefine evidence."""
    responses = [[0, 1], [1, 0], [1, 1]]
    original_asarray = np.asarray
    mutated = False

    def _mutate_caller_then_convert(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray:
        nonlocal mutated
        if not mutated:
            responses[1][0] = 0
            mutated = True
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(mokken.np, "asarray", _mutate_caller_then_convert)

    scores, n_persons, n_items = mokken._validated_scores(responses)

    assert mutated
    assert (n_persons, n_items) == (3, 2)
    assert scores.tolist() == [0, 1, 1, 0, 1, 1]


def test_builtin_sequence_replays_cell_budget_after_preflight_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row growth after its first length observation must not bypass the cell cap."""
    responses = [[0, 1], [1, 0]]
    original_len = len
    grown_rows: set[int] = set()

    def _grow_row_after_observation(value: object) -> int:
        observed = original_len(value)  # type: ignore[arg-type]
        if any(value is row for row in responses) and id(value) not in grown_rows:
            grown_rows.add(id(value))
            value.append(0)  # type: ignore[union-attr]
        return observed

    def _unexpected_materialization(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("over-budget mutated sequence reached NumPy materialization")

    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_CELLS", 4)
    monkeypatch.setattr(mokken, "len", _grow_row_after_observation, raising=False)
    monkeypatch.setattr(mokken.np, "asarray", _unexpected_materialization)

    with pytest.raises(ValueError, match=r"responses exceed 4 logical cells"):
        mokken._validated_scores(responses)


def test_builtin_row_growth_resource_error_precedes_live_scalar_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-preflight growth must be re-bounded before mutable-row value scans."""
    responses: list[list[object]] = [[0, 1], [1, 0]]
    target_row = responses[0]
    original_len = len
    mutated = False

    def _grow_after_first_row_length(value: object) -> int:
        nonlocal mutated
        observed = original_len(value)  # type: ignore[arg-type]
        if value is target_row and not mutated:
            target_row.extend([0, object()])
            mutated = True
        return observed

    def _unexpected_materialization(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("over-budget mutated row reached NumPy materialization")

    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_CELLS", 4)
    monkeypatch.setattr(mokken, "len", _grow_after_first_row_length, raising=False)
    monkeypatch.setattr(mokken.np, "asarray", _unexpected_materialization)

    with pytest.raises(ValueError, match=r"responses exceed 4 logical cells"):
        mokken._validated_scores(responses)

    assert mutated


def test_nested_ndarray_snapshot_replays_storage_kind_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package-owned row copy must re-prove numeric storage before densifying."""
    first_row = np.array([0, 1], dtype=np.int64)
    responses = [first_row, np.array([1, 0], dtype=np.int64)]
    original_array = np.array

    def _replace_first_row_copy(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray:
        if value is first_row:
            return original_array([object(), object()], dtype=object)
        return original_array(value, *args, **kwargs)

    def _unexpected_materialization(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("unsupported package snapshot reached dense materialization")

    monkeypatch.setattr(mokken.np, "array", _replace_first_row_copy)
    monkeypatch.setattr(mokken.np, "asarray", _unexpected_materialization)

    with pytest.raises(ValueError, match=r"responses must be a numeric array"):
        mokken._validated_scores(responses)
