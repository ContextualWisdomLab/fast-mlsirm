"""Native-result boundary regressions for Mokken analysis."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


_RESPONSES = [[0, 1], [1, 0], [0, 1]]


class _ArrayTrap:
    """Record any attempt to invoke an arbitrary NumPy conversion protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls += 1
        raise AssertionError("hostile native __array__ callback executed")


class _Core:
    def __init__(self, coefficient_result: object, scale: object) -> None:
        self.coefficient_result = coefficient_result
        self.scale = scale

    def mokken_coef_h(
        self,
        responses: np.ndarray,
        n_persons: int,
        n_items: int,
    ) -> object:
        return self.coefficient_result

    def mokken_aisp(
        self,
        responses: np.ndarray,
        n_persons: int,
        n_items: int,
        lower_bound: float,
        alpha: float,
    ) -> object:
        return self.scale


def _valid_coefficients() -> dict[str, object]:
    return {
        "hij": [float("nan"), 0.4, 0.4, float("nan")],
        "hi": [0.4, 0.4],
        "h": 0.4,
        "zij": [float("nan"), 1.25, 1.25, float("nan")],
        "zi": [1.25, 1.25],
        "z": 1.25,
    }


def test_mokken_rejects_hostile_native_vector_without_array_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact Rust-shaped list carriers may reach NumPy marshalling."""
    trap = _ArrayTrap()
    payload = _valid_coefficients()
    payload["hij"] = trap
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(payload, [1, 1]))

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)

    assert trap.calls == 0


def test_mokken_rejects_native_vector_cardinality_before_reshape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native vector lengths are replayed before public reshaping."""
    payload = _valid_coefficients()
    payload["hij"] = [float("nan"), 0.4, 0.4]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(payload, [1, 1]))

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)


def test_mokken_accepts_current_rust_shaped_native_result_with_nan_diagonal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carrier hardening preserves legitimate undefined diagonal statistics."""
    payload = _valid_coefficients()
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(payload, [1, 1]))

    result = mokken.mokken_analysis(_RESPONSES)

    assert result.hij.shape == (2, 2)
    assert np.isnan(result.hij[0, 0])
    assert result.hij[0, 1] == pytest.approx(0.4)
    assert result.hi.tolist() == pytest.approx([0.4, 0.4])
    assert result.h == pytest.approx(0.4)
    assert result.zij.shape == (2, 2)
    assert np.isnan(result.zij[0, 0])
    assert result.zi.tolist() == pytest.approx([1.25, 1.25])
    assert result.z == pytest.approx(1.25)
    assert result.scale.tolist() == [1, 1]
