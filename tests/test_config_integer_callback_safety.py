import numpy as np
import pytest

from fast_mlsirm.config import FitConfig, MLS2PLMConfig


class HostileIndex:
    """Integer-like object whose index protocol must never run in validation."""

    def __init__(self) -> None:
        self.calls = 0

    def __index__(self) -> int:
        self.calls += 1
        raise AssertionError("caller __index__ callback executed")


class HostileInt(int):
    """Caller-defined integer identity that must be rejected before comparison."""

    def __lt__(self, other: object) -> bool:
        raise AssertionError("caller comparison callback executed")

    def __le__(self, other: object) -> bool:
        raise AssertionError("caller comparison callback executed")

    def __gt__(self, other: object) -> bool:
        raise AssertionError("caller comparison callback executed")

    def __ge__(self, other: object) -> bool:
        raise AssertionError("caller comparison callback executed")


@pytest.mark.parametrize(
    "name",
    ["n_persons", "n_dims", "items_per_dim", "latent_dim"],
)
def test_simulation_integer_controls_reject_index_protocol_without_callback(name: str) -> None:
    value = HostileIndex()

    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        MLS2PLMConfig(**{name: value}).validate()

    assert value.calls == 0


@pytest.mark.parametrize(
    "name",
    ["n_persons", "n_dims", "items_per_dim", "latent_dim"],
)
def test_simulation_integer_controls_reject_int_subclasses_before_comparison(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        MLS2PLMConfig(**{name: HostileInt(2)}).validate()


@pytest.mark.parametrize("name", ["lbfgs_history", "xi_points", "xi_seed"])
def test_fit_integer_controls_reject_index_protocol_without_callback(name: str) -> None:
    value = HostileIndex()

    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        FitConfig(**{name: value}).validate()

    assert value.calls == 0


@pytest.mark.parametrize("name", ["lbfgs_history", "xi_points", "xi_seed"])
def test_fit_integer_controls_reject_int_subclasses(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        FitConfig(**{name: HostileInt(2)}).validate()


def test_supported_numpy_integer_scalars_remain_accepted() -> None:
    MLS2PLMConfig(
        n_persons=np.int32(20),
        n_dims=np.uint8(2),
        items_per_dim=np.int16(3),
        latent_dim=np.uint16(2),
    ).validate()

    FitConfig(
        lbfgs_history=np.int16(8),
        xi_points=np.uint32(512),
        xi_seed=np.uint64((1 << 64) - 1),
    ).validate()
