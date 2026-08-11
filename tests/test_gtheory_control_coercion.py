"""Fail-first safety contracts for G-theory public numeric controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


class _HostileInt:
    """Integer-like probe whose conversion hooks must not run in validation."""

    def __init__(self) -> None:
        self.int_calls = 0
        self.repr_calls = 0

    def __int__(self) -> int:
        self.int_calls += 1
        raise RuntimeError("GTHEORY_INT_COERCION_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("GTHEORY_INT_REPR_SENTINEL")


class _HostileFloat:
    """Real-like probe whose conversion hooks must not run in validation."""

    def __init__(self) -> None:
        self.float_calls = 0
        self.repr_calls = 0

    def __float__(self) -> float:
        self.float_calls += 1
        raise RuntimeError("GTHEORY_FLOAT_COERCION_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("GTHEORY_FLOAT_REPR_SENTINEL")


class _NoNumericCore:
    """Core stub proving invalid controls fail before Rust numerical work."""

    def gtheory_pi(self, *args, **kwargs):
        raise AssertionError("invalid G-theory controls reached Rust")

    def gtheory_pio(self, *args, **kwargs):
        raise AssertionError("invalid G-theory controls reached Rust")

    def phi_lambda(self, *args, **kwargs):
        raise AssertionError("invalid Phi(lambda) controls reached Rust")


def _pi_data() -> np.ndarray:
    """Return the smallest non-degenerate one-facet score matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _pio_data() -> np.ndarray:
    """Return the smallest non-degenerate two-facet score tensor."""
    return np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )


def test_pi_rejects_hostile_dstudy_size_before_integer_coercion(monkeypatch) -> None:
    """One-facet D-study sizes must be validated without caller ``__int__``."""
    hostile = _HostileInt()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[hostile])

    assert hostile.int_calls == 0
    assert hostile.repr_calls == 0


def test_pio_rejects_hostile_dstudy_size_before_integer_coercion(monkeypatch) -> None:
    """Two-facet D-study sizes must reject arbitrary integer-like objects."""
    hostile = _HostileInt()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_pio_data(), n_prime=[(hostile, 2)])

    assert hostile.int_calls == 0
    assert hostile.repr_calls == 0


def test_phi_lambda_rejects_hostile_cut_before_float_coercion(monkeypatch) -> None:
    """Mastery cut validation must not execute caller-defined ``__float__``."""
    hostile = _HostileFloat()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), hostile, n_i_prime=[2])

    assert hostile.float_calls == 0
    assert hostile.repr_calls == 0


def test_dstudy_controls_preserve_builtin_and_numpy_scalar_types(monkeypatch) -> None:
    """Trusted Python/NumPy scalar controls remain accepted after hardening."""
    seen: dict[str, object] = {}

    class _Core:
        def gtheory_pi(self, data, n_p, n_i, primes):
            seen["primes"] = primes
            return {
                "df": [1.0, 1.0, 1.0],
                "ss": [1.0, 1.0, 1.0],
                "ms": [1.0, 1.0, 1.0],
                "var_raw": [1.0, 1.0, 1.0],
                "var": [1.0, 1.0, 1.0],
                "d_study": [],
            }

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _Core())
    gtheory.gtheory_pi(_pi_data(), n_i_prime=[2, np.int64(3)])

    assert seen["primes"] == [2, 3]
