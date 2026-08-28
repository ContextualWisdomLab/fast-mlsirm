"""Regression coverage for multigroup population-label narrowing safety."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_mlsirm.fit import _compact_population_labels


def test_population_labels_reject_top_level_array_provider_without_callback() -> None:
    """Population membership must not be synthesized through caller ``__array__``."""

    class ArrayProvider:
        calls = 0

        def __array__(self, dtype=None):
            type(self).calls += 1
            return np.array([0, 1], dtype=np.int64)

    provider = ArrayProvider()
    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels(provider, 2, "group_id")

    assert ArrayProvider.calls == 0


def test_population_labels_reject_object_array_float_provider_without_callback() -> None:
    """Object-array elements must not gain authority through caller ``__float__``."""

    class FloatProvider:
        calls = 0

        def __float__(self):
            type(self).calls += 1
            return 0.0

    labels = np.array([FloatProvider(), 1], dtype=object)
    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels(labels, 2, "group_id")

    assert FloatProvider.calls == 0


def test_population_labels_preserve_callback_free_scalar_sequence() -> None:
    """Concrete Python/NumPy scalar sequences keep sorted-unique compaction."""

    labels = [np.int32(4), np.float32(2.0), True, 4]
    ids, n_populations = _compact_population_labels(labels, 4, "group_id")

    assert n_populations == 3
    assert ids.tolist() == [2, 1, 0, 2]


def test_population_labels_reject_ndarray_subclass_before_array_protocol() -> None:
    """Container subclasses cannot replace the trusted ndarray carrier."""

    class ArraySubclass(np.ndarray):
        calls = 0

        def __array__(self, dtype=None, copy=None):
            del dtype, copy
            type(self).calls += 1
            return np.array([0, 1], dtype=np.int64)

    labels = np.array([0, 1], dtype=np.int64).view(ArraySubclass)
    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels(labels, 2, "group_id")

    assert ArraySubclass.calls == 0


def test_population_labels_reject_container_and_scalar_subclasses() -> None:
    """Exact built-in containers and numeric scalars are the only sequences."""

    class ListSubclass(list):
        pass

    class IntSubclass(int):
        calls = 0

        def __int__(self):
            type(self).calls += 1
            return super().__int__()

    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels(ListSubclass([0, 1]), 2, "group_id")
    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels([IntSubclass(0), 1], 2, "group_id")

    assert IntSubclass.calls == 0


@pytest.mark.parametrize(
    "labels",
    [np.array(["0", "1"]), np.array([object(), object()], dtype=object)],
)
def test_population_labels_reject_string_and_object_storage(labels: np.ndarray) -> None:
    """Non-numeric ndarray storage is rejected without element coercion."""
    with pytest.raises(ValueError, match="group_id"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_unsigned_int64_narrowing_overflow() -> None:
    """Unsigned labels above INT64_MAX must not wrap into the reference group."""
    labels = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="signed 64-bit"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_float_int64_narrowing_overflow_without_warning() -> None:
    """Integral floats at 2**63 must fail closed rather than warn and wrap."""
    labels = np.array([0.0, float(2**63)], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="signed 64-bit"):
            _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_integer_sequence_at_signed_boundary() -> None:
    """A Python sequence containing 2**63 must fail regardless of NumPy promotion."""
    labels = [0, 2**63]

    with pytest.raises(ValueError, match="signed 64-bit"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_integer_sequence_below_signed_boundary() -> None:
    """A Python sequence below INT64_MIN must keep the signed-range diagnostic."""
    labels = [0, -(2**63) - 1]

    with pytest.raises(ValueError, match="signed 64-bit"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_preserve_signed_int64_upper_boundary() -> None:
    """The largest valid signed label remains admissible and order preserving."""
    labels = np.array([0, np.iinfo(np.int64).max], dtype=np.int64)

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]


def test_population_labels_preserve_extended_precision_int64_upper_boundary() -> None:
    """A wider real dtype must preserve an exactly representable INT64_MAX label."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no additional precision on this platform")

    labels = np.array(
        [np.longdouble(0), np.longdouble(np.iinfo(np.int64).max)],
        dtype=np.longdouble,
    )

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]


def test_population_labels_preserve_float16_without_boundary_warning() -> None:
    """Small floating dtypes must not overflow merely constructing the int64 bound."""
    labels = np.array([0.0, 1.0], dtype=np.float16)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]


def test_population_labels_preserve_mixed_sequence_identity_before_numpy_promotion() -> None:
    """Heterogeneous exact labels must not collapse through float64 promotion."""
    labels = [2**53 + 1, float(2**53)]

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [1, 0]
