"""Regression coverage for multigroup population-label narrowing safety."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_mlsirm.fit import _compact_population_labels


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


def test_population_labels_preserve_signed_int64_upper_boundary() -> None:
    """The largest valid signed label remains admissible and order preserving."""
    labels = np.array([0, np.iinfo(np.int64).max], dtype=np.int64)

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]
