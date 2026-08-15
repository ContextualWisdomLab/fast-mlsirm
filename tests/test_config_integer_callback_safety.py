"""Callback-safety regressions for public integer configuration controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig, MLS2PLMConfig


class _HostileIndex:
    """Integer-like caller object whose coercion callback must never execute."""

    calls = 0

    def __index__(self) -> int:
        """Record forbidden coercion and return an otherwise valid value."""
        type(self).calls += 1
        return 2


class _HostileInt(int):
    """Caller-defined integer subclass that is outside the trusted boundary."""


def _assert_rejected_without_index_callback(callable_) -> None:
    """Require validation to fail before invoking caller-controlled coercion."""
    _HostileIndex.calls = 0
    with pytest.raises(ValueError):
        callable_(_HostileIndex())
    assert _HostileIndex.calls == 0


@pytest.mark.parametrize(
    "field",
    ["n_persons", "n_dims", "items_per_dim", "latent_dim"],
)
def test_simulation_integer_controls_reject_index_callbacks(field: str) -> None:
    """Simulation-size validation rejects arbitrary index providers inertly."""
    _assert_rejected_without_index_callback(
        lambda value: MLS2PLMConfig(**{field: value}).validate()
    )


@pytest.mark.parametrize("field", ["lbfgs_history", "xi_points", "xi_seed"])
def test_fit_integer_controls_reject_index_callbacks(field: str) -> None:
    """Fit integer validation never dispatches arbitrary ``__index__`` hooks."""
    _assert_rejected_without_index_callback(
        lambda value: FitConfig(**{field: value}).validate()
    )


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        (MLS2PLMConfig, "n_persons"),
        (MLS2PLMConfig, "n_dims"),
        (MLS2PLMConfig, "items_per_dim"),
        (MLS2PLMConfig, "latent_dim"),
        (FitConfig, "lbfgs_history"),
        (FitConfig, "xi_points"),
        (FitConfig, "xi_seed"),
    ],
)
def test_integer_controls_reject_caller_int_subclasses(config_type, field: str) -> None:
    """Caller-defined ``int`` subclasses are not package-trusted controls."""
    with pytest.raises(ValueError):
        config_type(**{field: _HostileInt(2)}).validate()


@pytest.mark.parametrize("value", [1, np.int32(2), np.int64(3), np.uint64(4)])
def test_simulation_preserves_trusted_integer_scalars(value: object) -> None:
    """Built-in and genuine NumPy integers remain valid simulation controls."""
    MLS2PLMConfig(n_persons=value).validate()


@pytest.mark.parametrize("value", [1, np.int32(2), np.int64(3), np.uint64(4)])
def test_fit_preserves_trusted_integer_scalars(value: object) -> None:
    """Built-in and genuine NumPy integers remain valid fit controls."""
    FitConfig(lbfgs_history=value, xi_points=value, xi_seed=value).validate()
