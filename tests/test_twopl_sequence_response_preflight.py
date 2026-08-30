"""Fail-fast structural admission for sequence-backed 2PL response evidence."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import fast_mlsirm.twopl as twopl


def test_sequence_response_cell_budget_precedes_scalar_validation(monkeypatch) -> None:
    """An inert whole-matrix overflow wins before any response cell is traversed."""

    monkeypatch.setattr(twopl, "MAX_IRT_RESPONSE_CELLS", 4)
    original_response_scalar = twopl._response_scalar
    traversed: list[object] = []

    def recording_response_scalar(value: object) -> float:
        traversed.append(value)
        return original_response_scalar(value)

    monkeypatch.setattr(twopl, "_response_scalar", recording_response_scalar)

    with pytest.raises(ValueError, match=r"responses must contain at most 4 cells"):
        twopl._trusted_response_matrix([[0, 1], [1, 0], [0, 1]])

    assert traversed == []


def test_overbudget_list_does_not_snapshot_all_row_references(monkeypatch) -> None:
    """A proven overflow must not first duplicate the whole top-level list."""

    monkeypatch.setattr(twopl, "MAX_IRT_RESPONSE_CELLS", 4)
    row = [0, 1]
    responses = [row, row, row]
    baseline_references = sys.getrefcount(row)
    observed_references: list[int] = []
    original_minimum = twopl._require_minimum_item_count

    def recording_minimum(n_items: int) -> None:
        observed_references.append(sys.getrefcount(row))
        original_minimum(n_items)

    monkeypatch.setattr(twopl, "_require_minimum_item_count", recording_minimum)

    with pytest.raises(ValueError, match=r"responses must contain at most 4 cells"):
        twopl._trusted_response_matrix(responses)

    assert observed_references
    assert observed_references[0] < baseline_references + len(responses)


def test_sequence_rectangularity_precedes_scalar_validation() -> None:
    """Exact row lengths establish a rectangularity failure before cell traversal."""

    with pytest.raises(ValueError, match=r"responses must be a 2-D persons x items array"):
        twopl._trusted_response_matrix([[2, 0], [0, 1, 0]])


@pytest.mark.parametrize(
    ("row", "diagnostic"),
    [
        (np.array([1 + 0j, 0 + 0j]), r"responses must be real-valued"),
        (np.array(["0", "1"], dtype=object), r"responses must be a real numeric matrix"),
    ],
)
def test_sequence_ndarray_dtype_rejects_before_dense_allocation(
    monkeypatch, row: np.ndarray, diagnostic: str
) -> None:
    """Inert ndarray dtype rejection must not require the float64 destination."""

    def fail_allocation(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("dense response allocation must not run")

    monkeypatch.setattr(twopl.np, "empty", fail_allocation)

    with pytest.raises(ValueError, match=diagnostic):
        twopl._trusted_response_matrix([row])


def test_sequence_preflight_preserves_exact_cell_budget_boundary(monkeypatch) -> None:
    """The exact logical-cell ceiling remains admissible for ordinary evidence."""

    monkeypatch.setattr(twopl, "MAX_IRT_RESPONSE_CELLS", 4)

    result = twopl._trusted_response_matrix([[0, 1], np.array([1, 0], dtype=np.int8)])

    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert np.array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))
