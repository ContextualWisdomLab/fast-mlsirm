"""Fail-closed callback-boundary tests for testlet public controls."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.testlet import fit_testlet


class _HostileInt(int):
    """Integer subclass that records forbidden conversion callbacks."""

    calls = 0

    def __int__(self):
        type(self).calls += 1
        raise AssertionError("hostile __int__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile __repr__ executed")


class _HostileNpInt(np.int64):
    """NumPy integer subclass that records forbidden conversion callbacks."""

    calls = 0

    def __int__(self):
        type(self).calls += 1
        raise AssertionError("hostile numpy __int__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile numpy __repr__ executed")


class _HostileFloat(float):
    """Float subclass that records forbidden conversion callbacks."""

    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("hostile __float__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile __repr__ executed")


class _HostileNpFloat(np.float64):
    """NumPy floating subclass that records forbidden conversion callbacks."""

    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("hostile numpy __float__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile numpy __repr__ executed")


class _HostileStr(str):
    """String subclass that records forbidden conversion callbacks."""

    calls = 0

    def __str__(self):
        type(self).calls += 1
        raise AssertionError("hostile __str__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile __repr__ executed")


class _HostileBool:
    """Boolean-like object that records forbidden truth-value callbacks."""

    calls = 0

    def __bool__(self):
        type(self).calls += 1
        raise AssertionError("hostile __bool__ executed")

    def __repr__(self):
        type(self).calls += 1
        raise AssertionError("hostile __repr__ executed")


def _binary() -> np.ndarray:
    """Return a deterministic valid response matrix."""

    return np.zeros((2, 2), dtype=np.float64)


def _tid() -> np.ndarray:
    """Return a deterministic valid two-item testlet assignment."""

    return np.array([0, 0], dtype=np.int64)


def _assert_rejected_without_callback(keyword: str, value: object, cls: type) -> None:
    """Require rejection before caller conversion and before native discovery."""

    cls.calls = 0
    with patch(
        "fast_mlsirm.fitstats._core_module",
        side_effect=AssertionError("native core discovery must not run"),
    ):
        with pytest.raises(ValueError):
            fit_testlet(_binary(), _tid(), q_gamma=7, **{keyword: value})
    assert cls.calls == 0


@pytest.mark.parametrize(
    ("keyword", "value", "cls"),
    [
        ("max_iter", _HostileInt(7), _HostileInt),
        ("max_iter", _HostileNpInt(7), _HostileNpInt),
        ("q_gamma", _HostileInt(7), _HostileInt),
        ("q_gamma", _HostileNpInt(7), _HostileNpInt),
        ("tol", _HostileFloat(1e-6), _HostileFloat),
        ("tol", _HostileNpFloat(1e-6), _HostileNpFloat),
        ("init_sigma2", _HostileFloat(0.5), _HostileFloat),
        ("init_sigma2", _HostileNpFloat(0.5), _HostileNpFloat),
        ("model", _HostileStr("rasch"), _HostileStr),
        ("estimate_sigma", _HostileBool(), _HostileBool),
        ("require_convergence", _HostileBool(), _HostileBool),
    ],
)
def test_fit_testlet_rejects_control_subclasses_before_callbacks(
    keyword: str,
    value: object,
    cls: type,
) -> None:
    """Rejected public controls must not execute caller conversion callbacks."""

    _assert_rejected_without_callback(keyword, value, cls)


def test_fit_testlet_preserves_genuine_numpy_scalars() -> None:
    """Exact NumPy scalar classes remain valid public controls."""

    class _Core:
        @staticmethod
        def fit_testlet(*args):
            assert args[6] == "rasch"
            assert type(args[7]) is int
            assert args[7] == 3
            assert type(args[8]) is float
            assert args[8] == pytest.approx(1e-6)
            assert type(args[9]) is int
            assert args[9] == 7
            assert type(args[10]) is bool
            assert args[10] is False
            assert type(args[11]) is float
            assert args[11] == pytest.approx(0.25)
            return {
                "model": "rasch",
                "a": [1.0, 1.0],
                "b": [0.0, 0.0],
                "beta": [0.0, 0.0],
                "sigma2": [0.0],
                "theta": [0.0, 0.0],
                "loglik_trace": [0.0],
                "n_iter": 1,
                "converged": True,
                "n_parameters": 2,
                "termination_reason": "tolerance",
                "final_loglik_change": 0.0,
            }

    with patch("fast_mlsirm.fitstats._core_module", return_value=_Core()):
        fit = fit_testlet(
            _binary(),
            _tid(),
            model="rasch",
            max_iter=np.int64(3),
            tol=np.float64(1e-6),
            q_gamma=np.int64(7),
            estimate_sigma=np.bool_(False),
            init_sigma2=np.float64(0.25),
            require_convergence=np.bool_(False),
        )
    assert fit.converged
