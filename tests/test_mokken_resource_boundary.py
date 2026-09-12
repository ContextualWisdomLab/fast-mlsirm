"""Resource-boundary regressions for Mokken response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


_MAX_CELLS = 20_000_000


def _forbid_numpy_materialization(*args: object, **kwargs: object) -> np.ndarray:
    """Fail if resource admission reaches NumPy materialization."""
    raise AssertionError("NumPy materialization executed before resource rejection")


def test_mokken_rejects_oversized_exact_array_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-allocation broadcast view is bounded from logical shape alone."""
    oversized = np.broadcast_to(
        np.array([[0]], dtype=np.int8),
        (_MAX_CELLS + 1, 1),
    )
    assert type(oversized) is np.ndarray
    assert oversized.size == _MAX_CELLS + 1

    monkeypatch.setattr(mokken.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        mokken.mokken_analysis(oversized)


def test_mokken_rejects_oversized_exact_numpy_row_before_sequence_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested exact NumPy rows are bounded before NumPy stacks the matrix."""
    oversized_row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (_MAX_CELLS + 1,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(mokken.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        mokken.mokken_analysis(responses)


def test_mokken_rejects_quadratic_item_matrix_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The item-pair matrix has its own budget below the response-cell limit."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_MATRIX_CELLS", 4)

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="item-matrix cells"):
        mokken.mokken_analysis(np.zeros((3, 3), dtype=np.int8))


def test_mokken_preflights_quadratic_item_budget_before_score_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rectangular built-in width rejects impossible pair work before values."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_MATRIX_CELLS", 4)
    responses = [[object(), object(), object()] for _ in range(3)]

    monkeypatch.setattr(mokken.np, "asarray", _forbid_numpy_materialization)

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="item-matrix cells"):
        mokken.mokken_analysis(responses)


def test_mokken_quadratic_item_budget_keeps_exact_boundary_admissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rectangular built-in matrix at the pair-work boundary stays admitted."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_MATRIX_CELLS", 4)
    responses = [[0, 1], [1, 0], [0, 1]]

    assert mokken._trusted_score_source(responses) is responses
