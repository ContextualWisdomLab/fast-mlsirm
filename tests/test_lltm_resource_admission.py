"""Resource-ordering regressions for the public LLTM scientific matrices."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import lltm


def _unexpected_core() -> object:
    raise AssertionError("compiled core discovered before LLTM resource admission")


def test_response_cell_budget_precedes_dense_marshalling(monkeypatch):
    """Oversized exact response shapes must fail before float64 allocation or Rust."""

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_CELLS", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.broadcast_to(np.array([[0.0]], dtype=np.float64), (3, 1))
    q_design = np.array([[1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="responses.*resource limit"):
        lltm.fit_lltm(responses, q_design)


def test_design_cell_budget_precedes_dense_marshalling(monkeypatch):
    """Oversized exact explanatory-design shapes must fail before float64 allocation or Rust."""

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_CELLS", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    q_design = np.broadcast_to(np.array([[1.0]], dtype=np.float64), (3, 1))

    with pytest.raises(ValueError, match="q_design.*resource limit"):
        lltm.fit_lltm(responses, q_design)


def test_builtin_rectangular_shape_uses_same_cell_budget(monkeypatch):
    """Built-in matrices should be bounded from row-count/width metadata before conversion."""

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_CELLS", 3, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses.*resource limit"):
        lltm.fit_lltm([[0.0, 1.0], [1.0, 0.0]], [[0.0], [1.0]])
