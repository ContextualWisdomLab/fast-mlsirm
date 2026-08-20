"""Fail-closed trust-boundary tests for bifactor scoreability controls."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import pytest

scoreability = importlib.import_module("fast_mlsirm.bifactor_scoreability")


@dataclass
class _CallbackCounter:
    """Count caller-controlled callbacks attempted before trusted validation."""

    calls: int = 0

    def hit(self) -> NoReturn:
        """Record one callback and fail immediately."""
        self.calls += 1
        raise AssertionError("caller callback executed")


class _DataProbe:
    """Array-like whose materialization callbacks must remain untouched."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    @property
    def shape(self):
        self._counter.hit()

    def __array__(self, *_args, **_kwargs):
        self._counter.hit()


class _IndexProvider:
    """Arbitrary integer protocol provider that is not a trusted control type."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __index__(self) -> int:
        self._counter.hit()

    def __int__(self) -> int:
        self._counter.hit()


class _FloatProvider:
    """Arbitrary float protocol provider that is not a trusted control type."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __float__(self) -> float:
        self._counter.hit()


def _hostile_int(counter: _CallbackCounter) -> int:
    """Return an int subclass whose conversion hooks must never execute."""

    class HostileInt(int):
        def __int__(self):
            counter.hit()

        def __index__(self):
            counter.hit()

        def __repr__(self):
            counter.hit()

    return HostileInt(0)


def _hostile_float(counter: _CallbackCounter) -> float:
    """Return a float subclass whose conversion hooks must never execute."""

    class HostileFloat(float):
        def __float__(self):
            counter.hit()

        def __repr__(self):
            counter.hit()

    return HostileFloat(0.0)


def _call_standardized(data, *, general_factor=0, zero_tolerance=0.0):
    return scoreability.bifactor_scoreability(
        data,
        data,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )


def _call_logit(data, *, general_factor=0, zero_tolerance=0.0):
    return scoreability.bifactor_scoreability_from_logit_slopes(
        data,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )


def _call_valid_standardized(*, general_factor=0, zero_tolerance=0.0):
    loadings = np.asarray([[0.60, 0.20], [0.70, 0.30]], dtype=np.float64)
    uniquenesses = 1.0 - np.square(loadings).sum(axis=1)
    return scoreability.bifactor_scoreability(
        loadings,
        uniquenesses,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )


def _call_valid_logit(*, general_factor=0, zero_tolerance=0.0):
    slopes = np.asarray([[1.0, 0.2], [1.1, 0.3]], dtype=np.float64)
    return scoreability.bifactor_scoreability_from_logit_slopes(
        slopes,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )


@pytest.mark.parametrize("entrypoint", [_call_standardized, _call_logit])
@pytest.mark.parametrize("kind", ["subclass", "provider", "bool", "numpy_bool"])
def test_general_factor_rejects_untrusted_types_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
    kind: str,
) -> None:
    """General-factor controls fail before caller data or native discovery."""
    counter = _CallbackCounter()
    bad_value: object
    if kind == "subclass":
        bad_value = _hostile_int(counter)
    elif kind == "provider":
        bad_value = _IndexProvider(counter)
    elif kind == "bool":
        bad_value = True
    else:
        bad_value = np.bool_(True)
    monkeypatch.setattr(
        scoreability,
        "bifactor_core",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )

    with pytest.raises(ValueError, match="general_factor must be an integer"):
        entrypoint(_DataProbe(counter), general_factor=bad_value)

    assert counter.calls == 0


@pytest.mark.parametrize("entrypoint", [_call_standardized, _call_logit])
@pytest.mark.parametrize(
    "kind",
    ["float_subclass", "int_subclass", "provider", "bool", "numpy_bool"],
)
def test_zero_tolerance_rejects_untrusted_types_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
    kind: str,
) -> None:
    """Tolerance controls fail before caller data or native discovery."""
    counter = _CallbackCounter()
    bad_value: object
    if kind == "float_subclass":
        bad_value = _hostile_float(counter)
    elif kind == "int_subclass":
        bad_value = _hostile_int(counter)
    elif kind == "provider":
        bad_value = _FloatProvider(counter)
    elif kind == "bool":
        bad_value = False
    else:
        bad_value = np.bool_(False)
    monkeypatch.setattr(
        scoreability,
        "bifactor_core",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )

    with pytest.raises(ValueError, match="zero_tolerance must be a real number"):
        entrypoint(_DataProbe(counter), zero_tolerance=bad_value)

    assert counter.calls == 0


def test_trusted_numpy_controls_are_normalized_to_builtin_scalars(monkeypatch):
    """Concrete NumPy controls retain compatibility without leaking subclasses."""
    seen: list[tuple[int, float]] = []

    class FakeCore:
        def bifactor_indices_from_logit_slopes(
            self,
            slopes: np.ndarray,
            general_factor: int,
            zero_tolerance: float,
        ):
            assert slopes.dtype == np.float64
            assert type(general_factor) is int
            assert type(zero_tolerance) is float
            seen.append((general_factor, zero_tolerance))
            raise RuntimeError("normalized controls observed")

    monkeypatch.setattr(scoreability, "bifactor_core", lambda: FakeCore())
    slopes = np.asarray([[1.0, 0.2], [1.1, 0.3]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="normalized controls observed"):
        scoreability.bifactor_scoreability_from_logit_slopes(
            slopes,
            general_factor=np.int32(0),
            zero_tolerance=np.float32(0.0),
        )

    assert seen == [(0, 0.0)]


@pytest.mark.parametrize("entrypoint", [_call_valid_standardized, _call_valid_logit])
def test_validly_typed_general_factor_domain_errors_remain_rust_owned(entrypoint):
    """Trusted index values retain the compiled core's semantic range contract."""
    with pytest.raises(ValueError, match="general_factor must be in 0[.][.]2"):
        entrypoint(general_factor=np.int32(2))


@pytest.mark.parametrize("entrypoint", [_call_valid_standardized, _call_valid_logit])
@pytest.mark.parametrize("zero_tolerance", [np.float32(-0.1), np.float64(np.inf)])
def test_validly_typed_zero_tolerance_domain_errors_remain_rust_owned(
    entrypoint,
    zero_tolerance,
):
    """Trusted tolerance values retain the compiled core's semantic domain contract."""
    with pytest.raises(ValueError, match="zero_tolerance must be finite and non-negative"):
        entrypoint(zero_tolerance=zero_tolerance)
