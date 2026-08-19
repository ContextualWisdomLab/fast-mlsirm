"""Callback-safety regressions for governed validation policy controls."""

from __future__ import annotations

import pytest

from fast_mlsirm import validation


_THRESHOLD_FIELDS = (
    "qwk_min",
    "pearson_r_min",
    "degradation_max",
    "overall_smd_max",
    "subgroup_smd_max",
)


class _HostileStr(str):
    """String subclass that records any attempted normalization callback."""

    callbacks = 0

    def strip(self, *args, **kwargs):  # type: ignore[override]
        type(self).callbacks += 1
        raise AssertionError("caller-controlled str.strip executed")


class _HostileFloat(float):
    """Float subclass that records any attempted numeric coercion callback."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller-controlled __float__ executed")


class _HostileInt(int):
    """Integer subclass that records coercion and comparison callbacks."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller-controlled int __float__ executed")

    def __lt__(self, other):  # type: ignore[override]
        type(self).callbacks += 1
        raise AssertionError("caller-controlled int comparison executed")


@pytest.mark.parametrize("field_name", ("policy_id", "policy_version"))
def test_policy_identity_rejects_str_subclass_without_callbacks(field_name: str) -> None:
    """Policy identities must reject caller string subclasses before ``strip``."""
    _HostileStr.callbacks = 0

    with pytest.raises(ValueError, match=rf"{field_name} must be a non-empty string"):
        validation.ValidationPolicy(**{field_name: _HostileStr("hostile_policy")})

    assert _HostileStr.callbacks == 0


@pytest.mark.parametrize("field_name", _THRESHOLD_FIELDS)
def test_policy_threshold_rejects_float_subclass_without_callbacks(field_name: str) -> None:
    """Every threshold must reject a hostile float subclass before coercion."""
    _HostileFloat.callbacks = 0

    with pytest.raises(ValueError, match=rf"{field_name} must be a real number in 0\.\.1"):
        validation.ValidationPolicy(**{field_name: _HostileFloat(0.5)})

    assert _HostileFloat.callbacks == 0


@pytest.mark.parametrize("field_name", _THRESHOLD_FIELDS)
def test_policy_threshold_rejects_int_subclass_without_callbacks(field_name: str) -> None:
    """Integer subclasses cannot masquerade as threshold controls."""
    _HostileInt.callbacks = 0

    with pytest.raises(ValueError, match=rf"{field_name} must be a real number in 0\.\.1"):
        validation.ValidationPolicy(**{field_name: _HostileInt(1)})

    assert _HostileInt.callbacks == 0


def test_min_subgroup_n_rejects_int_subclass_without_callbacks() -> None:
    """The subgroup-size control must fail before caller comparisons execute."""
    _HostileInt.callbacks = 0

    with pytest.raises(ValueError, match=r"min_subgroup_n must be an integer >= 2"):
        validation.ValidationPolicy(min_subgroup_n=_HostileInt(3))

    assert _HostileInt.callbacks == 0


def test_exact_builtin_policy_controls_remain_normalized() -> None:
    """Valid built-in controls retain their public values and Rust marshalling."""
    policy = validation.ValidationPolicy(
        policy_id="research_diagnostic",
        policy_version="2.0",
        qwk_min=1,
        pearson_r_min=0.9,
        degradation_max=0.05,
        overall_smd_max=0.2,
        subgroup_smd_max=0.15,
        min_subgroup_n=3,
    )

    assert policy.policy_id == "research_diagnostic"
    assert policy.policy_version == "2.0"
    assert policy.rust_kwargs() == {
        "qwk_min": 1.0,
        "pearson_r_min": 0.9,
        "degradation_max": 0.05,
        "overall_smd_max": 0.2,
        "subgroup_smd_max": 0.15,
        "min_subgroup_n": 3,
    }
