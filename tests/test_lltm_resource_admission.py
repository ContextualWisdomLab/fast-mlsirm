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
    responses = np.array([[0.0, 1.0]], dtype=np.float64)
    q_design = np.broadcast_to(np.array([[1.0]], dtype=np.float64), (2, 2))

    with pytest.raises(ValueError, match="q_design.*resource limit"):
        lltm.fit_lltm(responses, q_design)


def test_builtin_rectangular_shape_uses_same_cell_budget(monkeypatch):
    """Built-in matrices should be bounded from row-count/width metadata before conversion."""

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_CELLS", 3, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses.*resource limit"):
        lltm.fit_lltm([[0.0, 1.0], [1.0, 0.0]], [[0.0], [1.0]])


def test_zero_cell_row_fanout_has_independent_structural_budget(monkeypatch):
    """Empty-row fan-out must fail before NumPy materialization or native discovery."""

    rows = [[], [], []]

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("NumPy materialization reached before LLTM structural admission")

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_NODES", 2, raising=False)
    monkeypatch.setattr(lltm.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses.*structural resource limit"):
        lltm.fit_lltm(rows, [[1.0]])


def test_valid_builtin_matrix_at_structural_boundary_reaches_rust(monkeypatch):
    """A non-empty matrix at the structural ceiling must preserve native marshalling."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_lltm(self, *args):
            captured["args"] = args
            return {
                "eta": [0.0],
                "intercept": 0.0,
                "b": [0.0],
                "theta": [0.0],
                "loglik_trace": [0.0],
                "n_iter": 1,
                "converged": True,
                "n_parameters": 1,
                "loglik_rasch": 0.0,
                "lr_stat": 0.0,
                "lr_df": 0,
                "lr_p": float("nan"),
            }

    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_CELLS", 1, raising=False)
    monkeypatch.setattr(lltm, "_MAX_LLTM_MATRIX_NODES", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())

    fitted = lltm.fit_lltm([[0.0]], [[1.0]], compute_lr=False, max_iter=1, tol=0.0)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.0]))
    np.testing.assert_array_equal(args[1], np.array([True]))
    np.testing.assert_array_equal(args[2], np.array([1.0]))
    assert args[3:6] == (1, 1, 1)
    assert fitted.n_iter == 1
