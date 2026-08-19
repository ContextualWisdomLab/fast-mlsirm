"""Adversarial public Python-boundary contracts for bifactor scoreability."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import bifactor_scoreability
from fast_mlsirm.bifactor_scoreability import (
    MAX_BIFACTOR_FACTORS,
    MAX_BIFACTOR_WORK_UNITS,
    bifactor_scoreability_from_logit_slopes,
)


class _NeverMaterialize:
    """Fail if rejected semantic controls allow data materialization to start."""

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("bifactor data must not be materialized")


class _HostileIndexProvider:
    """Count any package-triggered integer-protocol callback."""

    def __init__(self) -> None:
        self.calls = 0

    def __index__(self) -> int:
        self.calls += 1
        raise AssertionError("caller __index__ must not execute")


class _HostileFloatProvider:
    """Count any package-triggered real-number protocol callback."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ must not execute")


class _HostileInt(int):
    """An integer subclass whose coercion hooks must never be trusted."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller int subclass must not be coerced")

    def __index__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller int subclass must not be indexed")


class _HostileFloat(float):
    """A float subclass whose coercion hook must never be trusted."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float subclass must not be coerced")


def _loadings() -> np.ndarray:
    """Return one standardized strict-bifactor loading matrix."""
    return np.asarray(
        [
            [0.70, 0.40, 0.00],
            [0.70, 0.30, 0.00],
            [0.70, 0.00, 0.50],
            [0.70, 0.00, 0.60],
        ],
        dtype=np.float64,
    )


def _uniquenesses() -> np.ndarray:
    """Return residual variances satisfying the standardized identity."""
    return np.asarray([0.35, 0.42, 0.26, 0.15], dtype=np.float64)


def test_result_vectors_cannot_reenable_write_access():
    """Immutable results resist assignment and NumPy write-flag reactivation."""
    result = bifactor_scoreability(_loadings(), _uniquenesses())
    for vector in (
        result.ecv_ss,
        result.ecv_sg,
        result.ecv_gs,
        result.item_ecv,
        result.omega_total,
        result.omega_hierarchical,
        result.construct_replicability,
    ):
        assert vector.flags.writeable is False
        with pytest.raises(ValueError):
            vector[0] = 0.0
        with pytest.raises(ValueError):
            vector.setflags(write=True)


def test_oversized_object_ndarray_is_rejected_before_float_conversion():
    """Shape/work preflight runs before an untrusted ndarray dtype conversion."""
    n_items = MAX_BIFACTOR_WORK_UNITS // (MAX_BIFACTOR_FACTORS**2) + 1
    loadings = np.empty((n_items, MAX_BIFACTOR_FACTORS), dtype=object)
    uniquenesses = np.ones(n_items, dtype=np.float64)
    with pytest.raises(ValueError, match="work budget"):
        bifactor_scoreability(loadings, uniquenesses)


@pytest.mark.parametrize(
    ("kwargs", "message", "control"),
    [
        (
            {"general_factor": _HostileIndexProvider()},
            "general_factor must be an integer",
            "general_factor",
        ),
        (
            {"zero_tolerance": _HostileFloatProvider()},
            "zero_tolerance must be a real number",
            "zero_tolerance",
        ),
    ],
)
def test_standardized_controls_fail_before_data_materialization(kwargs, message, control):
    """Rejected controls never allow loading or uniqueness conversion to start."""
    hostile = kwargs[control]
    with pytest.raises(ValueError, match=message):
        bifactor_scoreability(_NeverMaterialize(), _NeverMaterialize(), **kwargs)
    assert hostile.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message", "control"),
    [
        (
            {"general_factor": _HostileIndexProvider()},
            "general_factor must be an integer",
            "general_factor",
        ),
        (
            {"zero_tolerance": _HostileFloatProvider()},
            "zero_tolerance must be a real number",
            "zero_tolerance",
        ),
    ],
)
def test_logit_controls_fail_before_data_materialization(kwargs, message, control):
    """The logit-slope wrapper rejects controls before slope conversion."""
    hostile = kwargs[control]
    with pytest.raises(ValueError, match=message):
        bifactor_scoreability_from_logit_slopes(_NeverMaterialize(), **kwargs)
    assert hostile.calls == 0


def test_caller_numeric_subclasses_are_rejected_without_callbacks():
    """Subclass-permissive PyO3 coercion cannot define scoreability controls."""
    _HostileInt.calls = 0
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match="general_factor must be an integer"):
        bifactor_scoreability(
            _loadings(),
            _uniquenesses(),
            general_factor=_HostileInt(0),
        )
    with pytest.raises(ValueError, match="zero_tolerance must be a real number"):
        bifactor_scoreability_from_logit_slopes(
            _loadings(),
            zero_tolerance=_HostileFloat(0.0),
        )

    assert _HostileInt.calls == 0
    assert _HostileFloat.calls == 0


def test_boolean_controls_are_not_numeric_scoreability_controls():
    """Python booleans cannot silently become factor indices or tolerances."""
    with pytest.raises(ValueError, match="general_factor must be an integer"):
        bifactor_scoreability(_loadings(), _uniquenesses(), general_factor=False)
    with pytest.raises(ValueError, match="zero_tolerance must be a real number"):
        bifactor_scoreability_from_logit_slopes(_loadings(), zero_tolerance=False)


def test_concrete_numpy_scoreability_controls_remain_supported():
    """Trusted concrete NumPy integer/real scalars preserve the public contract."""
    expected = bifactor_scoreability(_loadings(), _uniquenesses())
    actual = bifactor_scoreability(
        _loadings(),
        _uniquenesses(),
        general_factor=np.int64(0),
        zero_tolerance=np.float32(0.0),
    )

    assert actual.factor_item_counts == expected.factor_item_counts
    assert actual.is_strict_bifactor is expected.is_strict_bifactor
    np.testing.assert_allclose(actual.ecv_ss, expected.ecv_ss, rtol=0.0, atol=0.0)
