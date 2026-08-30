"""Temporal-coherence evidence for sequence-backed 2PL response snapshots."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.twopl as twopl


def test_outer_list_rebinding_after_snapshot_cannot_replace_later_row(monkeypatch) -> None:
    """A later top-level row rebind cannot splice a newer row into one matrix."""

    first = [0, 1]
    second = [1, 0]
    responses = [first, second]
    original_response_scalar = twopl._response_scalar
    rebound = False

    def rebinding_response_scalar(value: object) -> float:
        nonlocal rebound
        if not rebound:
            rebound = True
            responses[1] = [0, 0]
        return original_response_scalar(value)

    monkeypatch.setattr(twopl, "_response_scalar", rebinding_response_scalar)

    result = twopl._trusted_response_matrix(responses)

    assert rebound is True
    assert np.array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert responses[1] == [0, 0]


def test_mutable_list_row_is_snapshotted_before_scalar_traversal(monkeypatch) -> None:
    """A same-cardinality list mutation cannot create mixed-time row evidence."""

    row = [0, 1]
    original_response_scalar = twopl._response_scalar
    mutated = False

    def mutating_response_scalar(value: object) -> float:
        nonlocal mutated
        if not mutated:
            mutated = True
            row[:] = [1, 0]
        return original_response_scalar(value)

    monkeypatch.setattr(twopl, "_response_scalar", mutating_response_scalar)

    result = twopl._trusted_response_matrix([row])

    assert mutated is True
    assert np.array_equal(result, np.array([[0.0, 1.0]]))
    assert row == [1, 0]


def test_later_list_row_is_snapshotted_before_earlier_scalar_traversal(monkeypatch) -> None:
    """Earlier scalar work cannot splice later live-row mutation into one matrix."""

    first = [0, 1]
    second = [1, 0]
    responses = [first, second]
    original_response_scalar = twopl._response_scalar
    mutated = False

    def mutating_response_scalar(value: object) -> float:
        nonlocal mutated
        if not mutated:
            mutated = True
            second[:] = [0, 0]
        return original_response_scalar(value)

    monkeypatch.setattr(twopl, "_response_scalar", mutating_response_scalar)

    result = twopl._trusted_response_matrix(responses)

    assert mutated is True
    assert np.array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert second == [0, 0]


def test_top_level_ndarray_admission_does_not_alias_caller_storage() -> None:
    """An already contiguous float64 input must still become package-owned evidence."""

    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    admitted = twopl._trusted_response_matrix(responses)

    assert admitted is not responses
    responses[:, :] = 0.0
    assert np.array_equal(admitted, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_top_level_ndarray_snapshot_replays_value_semantics(monkeypatch) -> None:
    """A mutation immediately before the package snapshot must be validated."""

    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    original_array = twopl.np.array
    mutated = False

    def mutate_before_copy(value: object, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal mutated
        if value is responses and kwargs.get("copy") is True:
            responses[0, 0] = 2.0
            mutated = True
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(twopl.np, "array", mutate_before_copy)

    with pytest.raises(ValueError, match="dichotomous responses must be 0, 1, or NaN"):
        twopl._trusted_response_matrix(responses)

    assert mutated is True
