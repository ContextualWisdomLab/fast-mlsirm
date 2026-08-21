"""Trust-boundary regressions for hierarchical longitudinal execution controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.multilevel import fit_hierarchical_longitudinal_irt


class _HostileOrder:
    """Control object whose comparison callback must never execute."""

    def __init__(self) -> None:
        self.calls = 0

    def __lt__(self, other: object) -> bool:
        self.calls += 1
        raise AssertionError("caller comparison callback executed")


class _HostileFloat:
    """Control object whose float conversion callback must never execute."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller float callback executed")


@pytest.mark.parametrize("name", ["worker_count", "max_iter"])
def test_integer_execution_controls_fail_closed_before_callbacks(name: str) -> None:
    """Integer controls reject alien ordering protocols with package errors."""
    value = _HostileOrder()
    with pytest.raises(ValueError):
        fit_hierarchical_longitudinal_irt(
            object(),
            np.zeros((1, 2), dtype=np.float64),
            **{name: value},
        )
    assert value.calls == 0


@pytest.mark.parametrize("name", ["worker_count", "max_iter"])
def test_integer_execution_controls_reject_unrepresentable_values(name: str) -> None:
    """Huge built-in integers must not leak native conversion OverflowError."""
    with pytest.raises(ValueError, match=name):
        fit_hierarchical_longitudinal_irt(
            object(),
            np.zeros((1, 2), dtype=np.float64),
            **{name: 10**400},
        )


@pytest.mark.parametrize("name", ["tolerance", "hessian_step"])
def test_real_execution_controls_fail_closed_before_callbacks(name: str) -> None:
    """Real controls reject alien conversion protocols with package errors."""
    value = _HostileFloat()
    with pytest.raises(ValueError):
        fit_hierarchical_longitudinal_irt(
            object(),
            np.zeros((1, 2), dtype=np.float64),
            **{name: value},
        )
    assert value.calls == 0


@pytest.mark.parametrize("name", ["tolerance", "hessian_step"])
def test_real_execution_controls_reject_unrepresentable_values(name: str) -> None:
    """Huge built-in integers must become package-owned real-control errors."""
    with pytest.raises(ValueError, match=name):
        fit_hierarchical_longitudinal_irt(
            object(),
            np.zeros((1, 2), dtype=np.float64),
            **{name: 10**400},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("worker_count", "1"),
        ("max_iter", "250"),
        ("tolerance", "1e-5"),
        ("hessian_step", "1e-3"),
    ],
)
def test_nonnumeric_execution_controls_raise_value_error(name: str, value: str) -> None:
    """Documented execution-control failures normalize to ValueError."""
    with pytest.raises(ValueError):
        fit_hierarchical_longitudinal_irt(
            object(),
            np.zeros((1, 2), dtype=np.float64),
            **{name: value},
        )
