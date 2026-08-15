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


@pytest.mark.parametrize(
    "field",
    [
        "latent_dim",
        "lbfgs_history",
        "max_iter",
        "n_restarts",
        "m_steps",
        "xi_points",
        "xi_seed",
    ],
)
def test_fit_integer_controls_reject_index_callbacks(field: str) -> None:
    """Fit integer validation never dispatches arbitrary ``__index__`` hooks."""
    _assert_rejected_without_index_callback(
        lambda value: FitConfig(**{field: value}).validate()
    )


@pytest.mark.parametrize(
    ("config_type", "field", "valid_value"),
    [
        (MLS2PLMConfig, "n_persons", 2),
        (MLS2PLMConfig, "n_dims", 2),
        (MLS2PLMConfig, "items_per_dim", 2),
        (MLS2PLMConfig, "latent_dim", 2),
        (FitConfig, "latent_dim", 2),
        (FitConfig, "lbfgs_history", 2),
        (FitConfig, "max_iter", 2),
        (FitConfig, "n_restarts", 2),
        (FitConfig, "q_theta", 21),
        (FitConfig, "q_xi", 11),
        (FitConfig, "q_u", 15),
        (FitConfig, "m_steps", 2),
        (FitConfig, "xi_points", 2),
        (FitConfig, "xi_seed", 2),
    ],
)
def test_integer_controls_reject_caller_int_subclasses(
    config_type,
    field: str,
    valid_value: int,
) -> None:
    """Caller-defined ``int`` subclasses are not package-trusted controls."""
    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        config_type(**{field: _HostileInt(valid_value)}).validate()


@pytest.mark.parametrize("value", [1, np.int32(2), np.int64(3), np.uint64(4)])
def test_simulation_preserves_trusted_integer_scalars(value: object) -> None:
    """Built-in and genuine NumPy integers remain valid simulation controls."""
    MLS2PLMConfig(n_persons=value).validate()


@pytest.mark.parametrize("value", [1, np.int32(2), np.int64(3), np.uint64(4)])
def test_fit_preserves_trusted_integer_scalars(value: object) -> None:
    """Built-in and genuine NumPy integers remain valid fit controls."""
    FitConfig(lbfgs_history=value, xi_points=value, xi_seed=value).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latent_dim", np.int32(2)),
        ("max_iter", np.int64(10)),
        ("n_restarts", np.uint64(2)),
        ("q_theta", np.int64(21)),
        ("q_xi", np.uint64(11)),
        ("q_u", np.int32(15)),
        ("m_steps", np.int64(4)),
    ],
)
def test_fit_preserves_trusted_integer_scalars_across_bounded_controls(
    field: str,
    value: object,
) -> None:
    """Trusted NumPy integer scalars remain valid across bounded fit controls."""
    FitConfig(**{field: value}).validate()
