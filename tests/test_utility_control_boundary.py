"""Trust-boundary regressions for classical selection utility controls."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm.utility import selection_utility, taylor_russell


class _HostileFloat:
    """Arbitrary float-protocol object whose callback must stay unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller __float__ callback must not execute")


@pytest.mark.parametrize(
    "call",
    (
        lambda value: selection_utility(value, 1.0, 0.5, 0.5),
        lambda value: taylor_russell(value, 0.5, 0.5),
    ),
)
def test_utility_rejects_float_protocol_objects_without_callbacks(call) -> None:
    """Untrusted float-protocol objects fail before any caller callback executes."""
    _HostileFloat.reset()

    with pytest.raises(ValueError, match="must be a finite real number"):
        call(_HostileFloat())

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize(
    "call",
    (
        lambda value: selection_utility(value, 1.0, 0.5, 0.5),
        lambda value: taylor_russell(value, 0.5, 0.5),
    ),
)
@pytest.mark.parametrize("value", (True, False, math.inf, -math.inf, math.nan))
def test_utility_rejects_boolean_and_nonfinite_controls(call, value) -> None:
    """Boolean and non-finite scalars are rejected by the Python trust boundary."""
    with pytest.raises(ValueError, match="must be a finite real number"):
        call(value)


def test_numpy_real_scalars_preserve_native_results() -> None:
    """Trusted NumPy real scalars produce the same Rust-owned results as floats."""
    py_utility = selection_utility(10.0, 2.0, 0.4, 0.5, 1.0, 2.0)
    np_utility = selection_utility(
        np.float64(10.0),
        np.float64(2.0),
        np.float64(0.4),
        np.float64(0.5),
        np.float64(1.0),
        np.float64(2.0),
    )
    assert np_utility == py_utility

    py_tr = taylor_russell(0.4, 0.5, 0.6)
    np_tr = taylor_russell(np.float64(0.4), np.float64(0.5), np.float64(0.6))
    assert np_tr == py_tr
