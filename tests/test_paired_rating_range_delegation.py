"""Direct Rust-delegation contracts for paired rating-range evidence."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from fast_mlsirm import RatingRangeEvidence, paired_rating_range_evidence
from fast_mlsirm._rating_range_core_loader import rating_range_core
import fast_mlsirm.rating_range as rating_range_module


def test_every_public_field_matches_the_raw_rust_payload() -> None:
    """Python validates and marshals but never recomputes a numerical field."""

    automated = np.array([1, 1, 2, 3, 3, 2], dtype=np.uint32)
    reference = np.array([0, 1, 2, 3, 4, 4], dtype=np.uint32)

    result = paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )
    raw = rating_range_core().paired_rating_range_evidence(
        automated,
        reference,
        5,
    )

    assert isinstance(result, RatingRangeEvidence)
    assert {field.name for field in fields(RatingRangeEvidence)} == set(raw)
    for field in fields(RatingRangeEvidence):
        actual = getattr(result, field.name)
        expected = raw[field.name]
        if isinstance(actual, float):
            assert actual == float(expected)
        elif actual is None:
            assert expected is None
        else:
            assert actual == expected


def test_python_wrapper_does_not_mutate_caller_arrays() -> None:
    """Delegating to Rust leaves the exact caller-owned rating buffers unchanged."""

    automated = np.array([1, 2, 3, 2], dtype=np.uint32)
    reference = np.array([0, 2, 4, 3], dtype=np.uint32)
    automated_before = automated.copy()
    reference_before = reference.copy()

    paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    np.testing.assert_array_equal(automated, automated_before)
    np.testing.assert_array_equal(reference, reference_before)


def test_rating_input_boundary_rejects_oversized_non_numeric_and_nonfinite_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rust scoring call is preceded by bounded numeric label validation."""
    monkeypatch.setattr(rating_range_module, "MAX_OBSERVATIONS", 2)
    with pytest.raises(ValueError, match="observation limit"):
        paired_rating_range_evidence(
            np.array([0, 1, 2]), np.array([0, 1, 2]), category_count=3
        )
    with pytest.raises(ValueError, match="numeric integer labels"):
        paired_rating_range_evidence(
            np.array(["0", "1"]), np.array([0, 1]), category_count=3
        )
    with pytest.raises(ValueError, match="finite integer labels"):
        paired_rating_range_evidence(
            np.array([0.0, np.inf]), np.array([0, 1]), category_count=3
        )


def test_rating_input_boundary_accepts_float_integer_labels() -> None:
    """Integral floating-point labels are normalized without changing meaning."""
    result = paired_rating_range_evidence(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), category_count=3
    )
    assert result.sample_size == 2
