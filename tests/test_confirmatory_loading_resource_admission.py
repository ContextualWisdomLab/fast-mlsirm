"""Resource-admission regression tests for confirmatory loading patterns."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

import fast_mlsirm.models as models
from fast_mlsirm.irt_contract import MAX_IRT_RESPONSE_CELLS


_RESOURCE_ERROR = "confirmatory loading_pattern exceeds the supported cell budget"


def _unexpected_full_scan(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("oversized loading_pattern reached an O(n) value scan")


def _unexpected_scalar_normalization(_value: object) -> int:
    raise AssertionError("oversized loading_pattern reached scalar normalization")


def _track_fromiter_chunks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counts: list[int] = []
    original = np.fromiter

    def tracked(
        iterable: Iterable[object],
        dtype: object,
        count: int = -1,
    ) -> np.ndarray:
        counts.append(count)
        return original(iterable, dtype=dtype, count=count)

    monkeypatch.setattr(models, "_CONFIRMATORY_SERIALIZATION_CHUNK_CELLS", 2, raising=False)
    monkeypatch.setattr(models.np, "fromiter", tracked)
    return counts


def test_broadcast_ndarray_is_rejected_before_full_value_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = np.broadcast_to(
        np.array([[1.0]], dtype=np.float64),
        (1, MAX_IRT_RESPONSE_CELLS + 1),
    )
    monkeypatch.setattr(models.np, "isfinite", _unexpected_full_scan)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models.ConfirmatoryModel(huge)


def test_sequence_preflight_rejects_row_fanout_before_row_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models._confirmatory_sequence_width([object(), [1], [1], [1], [1]])


def test_sequence_preflight_rejects_known_oversize_before_later_row_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models._confirmatory_sequence_width([[1, 0, 1], object()])


def test_exact_sequence_is_rejected_before_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)
    monkeypatch.setattr(models, "_confirmatory_scalar", _unexpected_scalar_normalization)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models.ConfirmatoryModel([[1, 0, 1], [0, 1, 0]])


def test_replay_rejects_rebound_oversize_before_binary_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models.ConfirmatoryModel(np.array([[1]], dtype=np.int64))
    rebound = np.ones((2, 3), dtype=np.int64)
    rebound.setflags(write=False)
    object.__setattr__(model, "loading_pattern", rebound)

    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)
    monkeypatch.setattr(models.np, "all", _unexpected_full_scan)

    with pytest.raises(ValueError, match="confirmatory model loading_pattern is not canonical"):
        _ = model.n_dims


def test_replay_rejects_foreign_backing_storage() -> None:
    model = models.ConfirmatoryModel(np.array([[1], [0]], dtype=np.int64))
    backing = np.array([[1], [0]], dtype=np.int64)
    rebound = backing.view()
    rebound.setflags(write=False)
    assert rebound.flags.c_contiguous
    assert not rebound.flags.owndata
    assert backing.flags.writeable
    object.__setattr__(model, "loading_pattern", rebound)

    with pytest.raises(ValueError, match="confirmatory model loading_pattern is not canonical"):
        _ = model.n_dims


def test_constructor_canonical_storage_cannot_be_made_writeable() -> None:
    model = models.ConfirmatoryModel([[1, 0], [0, 1]])

    assert not model.loading_pattern.flags.writeable
    with pytest.raises(ValueError):
        model.loading_pattern.setflags(write=True)
    assert np.array_equal(model.loading_pattern, np.array([[1, 0], [0, 1]], dtype=np.int64))


def test_ndarray_immutable_materialization_uses_bounded_conversion_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    counts = _track_fromiter_chunks(monkeypatch)

    model = models.ConfirmatoryModel(source)

    assert counts == [2, 2, 2]
    assert np.array_equal(model.loading_pattern, source.astype(np.int64))


def test_ndarray_mutation_after_validation_is_rechecked_during_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([[1.0, 0.0]], dtype=np.float64)
    original_fromiter = np.fromiter
    mutated = False

    def mutating_fromiter(
        iterable: Iterable[object],
        dtype: object,
        count: int = -1,
    ) -> np.ndarray:
        nonlocal mutated
        if not mutated:
            source[0, 0] = 0.5
            mutated = True
        return original_fromiter(iterable, dtype=dtype, count=count)

    monkeypatch.setattr(models.np, "fromiter", mutating_fromiter)

    with pytest.raises(
        ValueError,
        match="confirmatory loading_pattern entries must be finite and exactly 0 or 1",
    ):
        models.ConfirmatoryModel(source)

    assert mutated


def test_ndarray_row_mutation_after_row_validation_is_rechecked_during_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = np.array([1.0, 0.0], dtype=np.float64)
    original_validate_row = models._confirmatory_ndarray_row
    mutated = False

    def mutating_validate_row(value: np.ndarray, width: int) -> np.ndarray:
        nonlocal mutated
        validated = original_validate_row(value, width)
        if value is row and not mutated:
            row[0] = 0.5
            mutated = True
        return validated

    monkeypatch.setattr(models, "_confirmatory_ndarray_row", mutating_validate_row)

    with pytest.raises(
        ValueError,
        match="confirmatory loading_pattern entries must be finite and exactly 0 or 1",
    ):
        models.ConfirmatoryModel([row])

    assert mutated


def test_sequence_immutable_materialization_uses_bounded_conversion_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _track_fromiter_chunks(monkeypatch)

    model = models.ConfirmatoryModel([[1, 0, 1], [0, 1, 0]])

    assert counts == [2, 2, 2]
    assert np.array_equal(
        model.loading_pattern,
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int64),
    )


def test_sequence_scalar_normalization_streams_into_conversion_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_scalar = models._confirmatory_scalar
    original_fromiter = np.fromiter

    def tracked_scalar(value: object) -> int:
        normalized = original_scalar(value)
        events.append("scalar")
        return normalized

    def tracked_fromiter(
        iterable: Iterable[object],
        dtype: object,
        count: int = -1,
    ) -> np.ndarray:
        chunk = original_fromiter(iterable, dtype=dtype, count=count)
        events.append("chunk")
        return chunk

    monkeypatch.setattr(models, "_CONFIRMATORY_SERIALIZATION_CHUNK_CELLS", 2, raising=False)
    monkeypatch.setattr(models, "_confirmatory_scalar", tracked_scalar)
    monkeypatch.setattr(models.np, "fromiter", tracked_fromiter)

    model = models.ConfirmatoryModel([[1, 0], [0, 1]])

    assert events == ["scalar", "scalar", "chunk", "scalar", "scalar", "chunk"]
    assert np.array_equal(model.loading_pattern, np.array([[1, 0], [0, 1]], dtype=np.int64))


def test_confirmatory_loading_cell_budget_boundary_remains_admissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)

    model = models.ConfirmatoryModel([[1, 0], [0, 1]])

    assert model.n_dims == 2
    assert np.array_equal(model.loading_pattern, np.array([[1, 0], [0, 1]], dtype=np.int64))
