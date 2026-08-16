"""Callback-safety regressions for public integer configuration controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig, MLS2PLMConfig
from fast_mlsirm.simulation import simulate


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
    ["n_persons", "n_dims", "items_per_dim", "latent_dim", "seed"],
)
def test_simulation_integer_controls_reject_index_callbacks(field: str) -> None:
    """Simulation-size validation rejects arbitrary index providers inertly."""
    _assert_rejected_without_index_callback(
        lambda value: MLS2PLMConfig(**{field: value})
    )


@pytest.mark.parametrize(
    "field",
    [
        "latent_dim",
        "lbfgs_history",
        "max_iter",
        "n_restarts",
        "q_theta",
        "q_xi",
        "q_u",
        "m_steps",
        "xi_points",
        "xi_seed",
        "seed",
        "verbose",
    ],
)
def test_fit_integer_controls_reject_index_callbacks(field: str) -> None:
    """Fit integer validation never dispatches arbitrary ``__index__`` hooks."""
    _assert_rejected_without_index_callback(
        lambda value: FitConfig(**{field: value})
    )


@pytest.mark.parametrize(
    ("config_type", "field", "valid_value"),
    [
        (MLS2PLMConfig, "n_persons", 2),
        (MLS2PLMConfig, "n_dims", 2),
        (MLS2PLMConfig, "items_per_dim", 2),
        (MLS2PLMConfig, "latent_dim", 2),
        (MLS2PLMConfig, "seed", 2),
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
        (FitConfig, "seed", 2),
        (FitConfig, "verbose", 2),
    ],
)
def test_integer_controls_reject_caller_int_subclasses(
    config_type,
    field: str,
    valid_value: int,
) -> None:
    """Caller-defined ``int`` subclasses are not package-trusted controls."""
    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        config_type(**{field: _HostileInt(valid_value)})


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


def test_simulation_stores_trusted_integers_as_builtin_ints() -> None:
    """Admitted NumPy sizes are stored as built-in integers after validation."""
    config = MLS2PLMConfig(
        n_persons=np.int32(2),
        n_dims=np.uint8(16),
        items_per_dim=np.uint8(16),
        latent_dim=np.int64(1),
        seed=np.uint8(7),
    )
    assert type(config.n_persons) is int
    assert type(config.n_dims) is int
    assert type(config.items_per_dim) is int
    assert type(config.latent_dim) is int
    assert type(config.seed) is int
    assert config.n_items == 256


def test_simulation_narrow_numpy_sizes_keep_true_item_count() -> None:
    """Narrow unsigned sizes must not wrap the published item-count product."""
    config = MLS2PLMConfig(
        n_persons=2,
        n_dims=np.uint8(16),
        items_per_dim=np.uint8(16),
        latent_dim=1,
        seed=3,
    )
    data = simulate(config)
    assert data.Y.shape == (2, 256)
    assert data.factor_id.shape == (256,)


def test_fit_stores_trusted_integers_as_builtin_ints() -> None:
    """Admitted NumPy fit controls are stored as built-in integers."""
    config = FitConfig(
        seed=np.uint8(250),
        verbose=np.int16(2),
        max_iter=np.int32(4),
        n_restarts=np.uint8(3),
    )
    assert type(config.seed) is int
    assert type(config.verbose) is int
    assert type(config.max_iter) is int
    assert type(config.n_restarts) is int
    assert config.seed == 250
    assert config.seed + 10 == 260


@pytest.mark.parametrize("value", [True, False])
def test_simulation_seed_rejects_bool(value: bool) -> None:
    """Boolean seeds are not package-trusted RNG controls."""
    with pytest.raises(ValueError, match="seed must be an integer"):
        MLS2PLMConfig(seed=value)


@pytest.mark.parametrize("value", [True, False])
def test_fit_seed_and_verbose_reject_bool(value: bool) -> None:
    """Boolean seed and verbosity controls stay outside the trusted boundary."""
    with pytest.raises(ValueError, match="seed must be an integer"):
        FitConfig(seed=value)
    with pytest.raises(ValueError, match="verbose must be an integer"):
        FitConfig(verbose=value)
