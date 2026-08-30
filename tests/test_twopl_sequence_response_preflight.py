"""Fail-fast structural admission for sequence-backed 2PL response evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.twopl as twopl


def test_sequence_response_cell_budget_precedes_scalar_validation(monkeypatch) -> None:
    """An inert whole-matrix overflow wins before any earlier invalid cell is read."""

    monkeypatch.setattr(twopl, "MAX_IRT_RESPONSE_CELLS", 4)

    with pytest.raises(ValueError, match=r"responses must contain at most 4 cells"):
        twopl._trusted_response_matrix([[2, 0], [0, 1], [1, 0]])


def test_sequence_rectangularity_precedes_scalar_validation() -> None:
    """Exact row lengths establish a rectangularity failure before cell traversal."""

    with pytest.raises(ValueError, match=r"responses must be a 2-D persons x items array"):
        twopl._trusted_response_matrix([[2, 0], [0, 1, 0]])


def test_sequence_preflight_preserves_exact_cell_budget_boundary(monkeypatch) -> None:
    """The exact logical-cell ceiling remains admissible for ordinary evidence."""

    monkeypatch.setattr(twopl, "MAX_IRT_RESPONSE_CELLS", 4)

    result = twopl._trusted_response_matrix([[0, 1], np.array([1, 0], dtype=np.int8)])

    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert np.array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))
