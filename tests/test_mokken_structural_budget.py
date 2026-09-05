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


def test_mokken_rejects_zero_width_matrix_before_snapshot_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible zero-item matrix fails before proportional row snapshots."""

    def _unexpected_snapshot(source: object) -> tuple[object, ...]:
        raise AssertionError("row snapshot allocation occurred")

    monkeypatch.setattr(mokken, "_snapshot_builtin_score_source", _unexpected_snapshot)

    with pytest.raises(ValueError, match="mokken requires at least 2 items"):
        mokken.mokken_analysis([(), (), ()])


def test_mokken_preserves_valid_matrix_at_structural_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid 3x2 matrix at its exact P+N structural boundary reaches Rust."""
    monkeypatch.setattr(mokken, "_MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES", 9, raising=False)
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
                "hij": [float("nan"), 0.4, 0.4, float("nan")],
                "hi": [0.4, 0.4],
                "h": 0.4,
                "zij": [float("nan"), 1.25, 1.25, float("nan")],
                "zi": [1.25, 1.25],
                "z": 1.25,
            }

        def mokken_aisp(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            lower_bound: float,
            alpha: float,
        ) -> list[int]:
            return [1, 1]

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = mokken.mokken_analysis([[0, 0], [0, 1], [1, 1]])

    assert captured["n_persons"] == 3
    assert captured["n_items"] == 2
    assert isinstance(captured["responses"], np.ndarray)
    assert captured["responses"].tolist() == [0, 0, 0, 1, 1, 1]
    assert result.scale.tolist() == [1, 1]
