"""Resource regressions for Mokken built-in response traversal."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


def test_mokken_rejects_empty_row_fanout_before_numpy_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell row fan-out consumes a bounded structural budget."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES", 2, raising=False)

    def _unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("NumPy materialization occurred")

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(mokken.np, "asarray", _unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match=r"responses exceed .* structural nodes"):
        mokken.mokken_analysis([[], [], []])


def test_mokken_preserves_valid_matrix_at_structural_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid 2x1 matrix at the 2N structural boundary still reaches Rust."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES", 4, raising=False)
    captured: dict[str, object] = {}

    class _Core:
        def mokken_coef_h(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
        ) -> dict[str, object]:
            captured.update(
                responses=responses.copy(),
                n_persons=n_persons,
                n_items=n_items,
            )
            return {
                "hij": [float("nan")],
                "hi": [float("nan")],
                "h": float("nan"),
                "zij": [float("nan")],
                "zi": [float("nan")],
                "z": float("nan"),
            }

        def mokken_aisp(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            lower_bound: float,
            alpha: float,
        ) -> list[int]:
            return [0]

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = mokken.mokken_analysis([[0], [1]])

    assert captured["n_persons"] == 2
    assert captured["n_items"] == 1
    assert isinstance(captured["responses"], np.ndarray)
    assert captured["responses"].tolist() == [0, 1]
    assert result.scale.tolist() == [0]
