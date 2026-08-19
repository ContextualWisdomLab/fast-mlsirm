"""Callback-safety regressions for :class:`ValidationPolicy` scalar admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import validation


_CALLBACKS: list[str] = []


class _HostileText(str):
    """String subclass that records any caller-controlled text dispatch."""

    def strip(self, *args: object, **kwargs: object) -> str:
        _CALLBACKS.append("strip")
        raise AssertionError("caller text callback executed")

    def __bool__(self) -> bool:
        _CALLBACKS.append("bool")
        raise AssertionError("caller truth callback executed")


class _HostileFloat(float):
    """Float subclass that records coercion or comparison dispatch."""

    def __float__(self) -> float:
        _CALLBACKS.append("float")
        raise AssertionError("caller float callback executed")

    def __le__(self, other: object) -> bool:
        _CALLBACKS.append("le")
        raise AssertionError("caller comparison callback executed")

    def __ge__(self, other: object) -> bool:
        _CALLBACKS.append("ge")
        raise AssertionError("caller comparison callback executed")


class _HostileInt(int):
    """Integer subclass that records coercion or comparison dispatch."""

    def __float__(self) -> float:
        _CALLBACKS.append("float")
        raise AssertionError("caller float callback executed")

    def __int__(self) -> int:
        _CALLBACKS.append("int")
        raise AssertionError("caller integer callback executed")

    def __index__(self) -> int:
        _CALLBACKS.append("index")
        raise AssertionError("caller index callback executed")

    def __lt__(self, other: object) -> bool:
        _CALLBACKS.append("lt")
        raise AssertionError("caller comparison callback executed")


@pytest.mark.parametrize("field", ["policy_id", "policy_version"])
def test_policy_identity_rejects_string_subclass_without_callbacks(field: str) -> None:
    """Policy identity admission must fail before subclass text methods can run."""
    _CALLBACKS.clear()

    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        validation.ValidationPolicy(**{field: _HostileText("hostile")})

    assert _CALLBACKS == []


@pytest.mark.parametrize(
    "field",
    [
        "qwk_min",
        "pearson_r_min",
        "degradation_max",
        "overall_smd_max",
        "subgroup_smd_max",
    ],
)
@pytest.mark.parametrize("hostile_type", [_HostileFloat, _HostileInt])
def test_policy_threshold_rejects_numeric_subclass_without_callbacks(
    field: str, hostile_type: type[float] | type[int]
) -> None:
    """Every governed threshold must reject subclass coercion before dispatch."""
    _CALLBACKS.clear()

    with pytest.raises(ValueError, match=rf"{field} must be a real number in 0\.\.1"):
        validation.ValidationPolicy(**{field: hostile_type(0)})

    assert _CALLBACKS == []


def test_min_subgroup_n_rejects_integer_subclass_without_callbacks() -> None:
    """The Rust integer control must be exact before comparison or marshalling."""
    _CALLBACKS.clear()

    with pytest.raises(ValueError, match=r"min_subgroup_n must be an integer >= 2"):
        validation.ValidationPolicy(min_subgroup_n=_HostileInt(2))

    assert _CALLBACKS == []


def test_policy_preserves_trusted_scalars_and_builtin_rust_marshalling() -> None:
    """Accepted legacy scalar identities normalize to inert built-ins for Rust."""
    policy = validation.ValidationPolicy(
        policy_id="research_diagnostic",
        policy_version="2.0",
        qwk_min=np.float64(0.80),
        pearson_r_min=1,
        degradation_max=0.05,
        overall_smd_max=0.20,
        subgroup_smd_max=0.15,
        min_subgroup_n=3,
    )

    assert type(policy.qwk_min) is float
    assert type(policy.pearson_r_min) is float
    assert type(policy.min_subgroup_n) is int
    kwargs = policy.rust_kwargs()
    assert all(type(kwargs[name]) is float for name in (
        "qwk_min",
        "pearson_r_min",
        "degradation_max",
        "overall_smd_max",
        "subgroup_smd_max",
    ))
    assert type(kwargs["min_subgroup_n"]) is int
