"""Callback-safety regressions for exposure-control scalar marshalling."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import exposure


class _HostileInt(int):
    """Integer subclass that records forbidden coercion callbacks."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass that must never be normalized."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("NumPy integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


class _HostileFloat(float):
    """Floating subclass that must be rejected before integer coercion."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("float integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


@pytest.mark.parametrize("hostile_type", [_HostileInt, _HostileNumpyInt, _HostileFloat])
def test_as_int_rejects_scalar_subclasses_without_callbacks(hostile_type) -> None:
    """Shared integer validation must establish exact scalar identity first."""

    hostile_type.callbacks.clear()
    with pytest.raises(ValueError, match="test_length must be an integer"):
        exposure._as_int("test_length", hostile_type(4), minimum=1, maximum=8)

    assert hostile_type.callbacks == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int8(2), 2),
        (np.uint64(3), 3),
        (np.float32(4.0), 4),
        (np.float64(5.0), 5),
        (6.0, 6),
    ],
)
def test_as_int_preserves_genuine_supported_numpy_and_builtin_scalars(
    value: object,
    expected: int,
) -> None:
    """Exact genuine numeric scalars preserve the established integral contract."""

    assert exposure._as_int("test_length", value, minimum=1, maximum=8) == expected


def test_sympson_hetter_rejects_hostile_control_before_core_use() -> None:
    """A public exposure boundary must fail closed before result-affecting dispatch."""

    _HostileInt.callbacks.clear()
    with pytest.raises(ValueError, match="test_length must be an integer"):
        exposure.sympson_hetter(
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            test_length=_HostileInt(2),
            n_simulees=4,
            max_iter=1,
            q_theta=5,
        )

    assert _HostileInt.callbacks == []


def test_a_stratified_rejects_hostile_control_without_callbacks() -> None:
    """Shared validation semantics also govern a second CAT public boundary."""

    _HostileNumpyInt.callbacks.clear()
    with pytest.raises(ValueError, match="n_strata must be an integer"):
        exposure.a_stratified(
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            n_strata=_HostileNumpyInt(2),
            test_length=2,
            n_simulees=4,
            q_theta=5,
        )

    assert _HostileNumpyInt.callbacks == []
