"""Trust-boundary regressions for the public Mokken analysis wrapper."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


class _HostileReal(float):
    """Real subclass whose semantic callbacks must never run."""

    def __float__(self) -> float:
        raise AssertionError("real conversion callback executed")

    def __le__(self, other: object) -> bool:
        raise AssertionError("real comparison callback executed")

    def __lt__(self, other: object) -> bool:
        raise AssertionError("real comparison callback executed")

    def __ge__(self, other: object) -> bool:
        raise AssertionError("real comparison callback executed")

    def __gt__(self, other: object) -> bool:
        raise AssertionError("real comparison callback executed")


class _HostileNumber:
    """Object-array element whose conversion must never execute."""

    def __float__(self) -> float:
        raise AssertionError("response numeric callback executed")


def _forbid_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if Rust capability discovery occurs before Python admission."""

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)


def test_mokken_rejects_hostile_lower_bound_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AISP lower-bound semantics reject subclasses before callbacks/native work."""
    _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="lower_bound must be a real number"):
        mokken.mokken_analysis(
            np.array([[0, 1], [1, 0]], dtype=np.int8),
            lower_bound=_HostileReal(0.3),
        )


def test_mokken_rejects_hostile_alpha_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AISP alpha semantics reject subclasses before callbacks/native work."""
    _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="alpha must be a real number"):
        mokken.mokken_analysis(
            np.array([[0, 1], [1, 0]], dtype=np.int8),
            alpha=_HostileReal(0.05),
        )


def test_mokken_rejects_complex_responses_before_real_narrowing_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imaginary response evidence cannot disappear through float64 coercion."""
    _forbid_core(monkeypatch)
    responses = np.array(
        [[0.0 + 1.0j, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        mokken.mokken_analysis(responses)


def test_mokken_rejects_object_responses_without_element_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object storage cannot execute per-element numeric conversion callbacks."""
    _forbid_core(monkeypatch)
    responses = np.array(
        [[_HostileNumber(), _HostileNumber()], [_HostileNumber(), _HostileNumber()]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        mokken.mokken_analysis(responses)


@pytest.mark.parametrize("overflow", [np.uint64(1 << 63), np.uint64((1 << 64) - 1)])
def test_mokken_rejects_signed_int64_narrowing_overflow_before_core(
    monkeypatch: pytest.MonkeyPatch,
    overflow: np.uint64,
) -> None:
    """Unsigned category identities cannot wrap into signed Rust scores."""
    _forbid_core(monkeypatch)
    responses = np.array([[0, overflow], [1, 0]], dtype=np.uint64)

    with pytest.raises(ValueError, match="responses exceed signed int64 range"):
        mokken.mokken_analysis(responses)


def test_mokken_preserves_concrete_numpy_scalar_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete NumPy controls normalize to inert built-in floats for Rust."""
    captured: dict[str, object] = {}

    class _Core:
        def mokken_coef_h(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
        ) -> dict[str, object]:
            captured.update(
                responses=responses,
                n_persons=n_persons,
                n_items=n_items,
            )
            return {
                "hij": [float("nan"), 0.4, 0.4, float("nan")],
                "hi": [0.4, 0.4],
                "h": 0.4,
                "zij": [float("nan"), 1.0, 1.0, float("nan")],
                "zi": [1.0, 1.0],
                "z": 1.0,
            }

        def mokken_aisp(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            lower_bound: float,
            alpha: float,
        ) -> list[int]:
            captured.update(lower_bound=lower_bound, alpha=alpha)
            return [1, 1]

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = mokken.mokken_analysis(
        np.array([[0, 1], [1, 0]], dtype=np.int16),
        lower_bound=np.float32(0.3),
        alpha=np.float64(0.05),
    )

    assert type(captured["lower_bound"]) is float
    assert type(captured["alpha"]) is float
    assert result.scale.tolist() == [1, 1]
