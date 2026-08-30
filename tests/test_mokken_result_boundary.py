"""Native-result boundary regressions for Mokken analysis."""

from __future__ import annotations

import gc

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


class _StringKeyTrap(str):
    """Record any package-side equality callback from a stored result key."""

    def __new__(cls, value: str) -> _StringKeyTrap:
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        self.calls += 1
        raise AssertionError("hostile native key comparison callback executed")


class _Core:
    def __init__(self, coefficient_result: object, scale: object) -> None:
        self.coefficient_result = coefficient_result
        self.scale = scale
        self.aisp_calls = 0

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
        self.aisp_calls += 1
        return self.scale


class _LifetimeCore:
    """Expose whether the temporary coefficient envelope survives into AISP."""

    def __init__(self) -> None:
        self._hij = [float("nan"), 0.4, 0.4, float("nan")]
        self._hi = [0.4, 0.4]
        self._zij = [float("nan"), 1.25, 1.25, float("nan")]
        self._zi = [1.25, 1.25]

    def mokken_coef_h(
        self,
        responses: np.ndarray,
        n_persons: int,
        n_items: int,
    ) -> object:
        return {
            "hij": self._hij,
            "hi": self._hi,
            "h": 0.4,
            "zij": self._zij,
            "zi": self._zi,
            "z": 1.25,
        }

    def mokken_aisp(
        self,
        responses: np.ndarray,
        n_persons: int,
        n_items: int,
        lower_bound: float,
        alpha: float,
    ) -> object:
        retained_envelopes = [
            referrer
            for referrer in gc.get_referrers(self._hij)
            if type(referrer) is dict and referrer.get("hij") is self._hij
        ]
        if retained_envelopes:
            raise AssertionError("coefficient result envelope retained into AISP")
        return [1, 1]


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


def test_mokken_rejects_callback_bearing_native_key_without_key_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required-field lookup never executes a provider-owned key protocol."""
    trap = _StringKeyTrap("hij")
    payload: dict[object, object] = {
        trap: [float("nan"), 0.4, 0.4, float("nan")],
        "hi": [0.4, 0.4],
        "h": 0.4,
        "zij": [float("nan"), 1.25, 1.25, float("nan")],
        "zi": [1.25, 1.25],
        "z": 1.25,
    }
    trap.calls = 0
    core = _Core(payload, [1, 1])
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)

    assert trap.calls == 0
    assert core.aisp_calls == 0


def test_mokken_rejects_extra_native_coefficient_field_before_aisp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coefficient envelope is the exact current six-key Rust contract."""
    payload = _valid_coefficients()
    payload["unexpected"] = 1.0
    core = _Core(payload, [1, 1])
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)

    assert core.aisp_calls == 0


def test_mokken_rejects_native_vector_cardinality_before_reshape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native vector lengths are replayed before public reshaping."""
    payload = _valid_coefficients()
    payload["hij"] = [float("nan"), 0.4, 0.4]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(payload, [1, 1]))

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)


def test_mokken_rejects_invalid_coefficients_before_aisp_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid coefficient envelope fails before the second native routine."""
    payload = _valid_coefficients()
    payload["hij"] = [float("nan"), 0.4, 0.4]
    core = _Core(payload, [1, 1])
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)

    assert core.aisp_calls == 0


@pytest.mark.parametrize("scale", ([-1, 1], [1 << 32, 1]))
def test_mokken_rejects_scale_values_outside_native_u32_domain(
    monkeypatch: pytest.MonkeyPatch,
    scale: list[int],
) -> None:
    """AISP labels replay the concrete Rust ``Vec<u32>`` scalar domain."""
    payload = _valid_coefficients()
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(payload, scale))

    with pytest.raises(ValueError, match="invalid Mokken Rust result payload"):
        mokken.mokken_analysis(_RESPONSES)


def test_mokken_releases_native_coefficient_envelope_before_aisp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package-owned arrays replace the temporary list envelope before AISP."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _LifetimeCore())

    result = mokken.mokken_analysis(_RESPONSES)

    assert result.hij.shape == (2, 2)
    assert result.scale.tolist() == [1, 1]


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
