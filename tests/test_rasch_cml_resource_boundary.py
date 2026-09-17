"""Resource-boundary regressions for Rasch CML response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.rasch_cml as rasch_cml


_MAX_CELLS = 20_000_000


def _forbid_numpy_materialization(*args: object, **kwargs: object) -> np.ndarray:
    """Fail if oversized evidence reaches NumPy materialization."""
    raise AssertionError("NumPy materialization executed before resource rejection")


def test_rasch_cml_rejects_oversized_exact_array_before_float64_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-allocation broadcast matrix is bounded from logical shape alone."""
    oversized = np.broadcast_to(
        np.array([[0]], dtype=np.int8),
        (_MAX_CELLS + 1, 1),
    )
    assert type(oversized) is np.ndarray
    assert oversized.size == _MAX_CELLS + 1

    monkeypatch.setattr(rasch_cml.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        rasch_cml.fit_rasch_cml(oversized)


def test_rasch_cml_rejects_oversized_numpy_row_before_sequence_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact NumPy row inside a built-in matrix is bounded before stacking."""
    oversized_row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (_MAX_CELLS + 1,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(rasch_cml.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        rasch_cml.fit_rasch_cml(responses)


def test_rasch_cml_rejects_zero_cell_structural_fanout_before_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty row fan-out consumes the structural envelope before NumPy work."""
    monkeypatch.setattr(
        rasch_cml,
        "_MAX_RASCH_RESPONSE_STRUCTURAL_NODES",
        2,
        raising=False,
    )
    monkeypatch.setattr(rasch_cml.np, "asarray", _forbid_numpy_materialization)
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("compiled core discovered before structural rejection")
        ),
    )

    with pytest.raises(ValueError, match="responses exceed 2 structural nodes"):
        rasch_cml.fit_rasch_cml([[], [], []])


def test_rasch_cml_structural_boundary_preserves_valid_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty matrix at the reduced structural boundary still reaches Rust."""
    monkeypatch.setattr(
        rasch_cml,
        "_MAX_RASCH_RESPONSE_STRUCTURAL_NODES",
        6,
        raising=False,
    )
    captured: dict[str, object] = {}

    class _Core:
        def fit_rasch_cml(
            self,
            yy: np.ndarray,
            n_persons: int,
            n_items: int,
            max_iter: int,
            tol: float,
        ) -> dict[str, object]:
            captured["yy"] = np.array(yy, copy=True)
            captured["shape"] = (n_persons, n_items)
            return {
                "beta": [0.0, 0.0],
                "se": [1.0, 1.0],
                "loglik": -1.0,
                "n_iter": 1,
                "converged": True,
                "n_used": 2,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = rasch_cml.fit_rasch_cml([[0, 1], [1, 0]])

    assert result["converged"] is True
    assert captured["shape"] == (2, 2)
    np.testing.assert_array_equal(captured["yy"], np.array([0, 1, 1, 0], dtype=np.int64))
