"""Contract tests for the Rust-owned Holm--Wald superiority foundation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fast_mlsirm.decision_superiority import assess_wald_superiority


def test_nearly_equal_estimates_are_indeterminate() -> None:
    """A unique point maximum cannot win while its uncertainty overlaps."""
    result = assess_wald_superiority(
        ("candidate-a", "candidate-b"),
        np.array([0.5001, 0.5]),
        np.array([[0.01, 0.0], [0.0, 0.01]]),
        familywise_error_rate=0.05,
    )

    assert result.winner_id is None
    assert result.decision == "indeterminate"


def test_separated_estimate_requires_all_holm_wald_comparisons() -> None:
    """A separated candidate wins only after every outgoing null is rejected."""
    result = assess_wald_superiority(
        ("candidate-a", "candidate-b", "candidate-c"),
        np.array([3.0, 0.0, -1.0]),
        np.diag([0.01, 0.01, 0.01]),
        familywise_error_rate=0.05,
    )

    assert result.winner_id == "candidate-a"
    assert result.decision == "superior"
    assert result.algorithm == "holm-wald-one-sided-v1"
    assert all(
        comparison.null_rejected
        for comparison in result.comparisons
        if comparison.candidate_id == "candidate-a"
    )


def test_candidate_permutation_cannot_change_winner_identity() -> None:
    """Candidate input order changes indices, never the substantive decision."""
    covariance = np.diag([0.01, 0.01, 0.01])
    original = assess_wald_superiority(
        ("candidate-a", "candidate-b", "candidate-c"),
        np.array([3.0, 0.0, -1.0]),
        covariance,
        familywise_error_rate=0.05,
    )
    permutation = np.array([2, 0, 1])
    permuted = assess_wald_superiority(
        ("candidate-c", "candidate-a", "candidate-b"),
        np.array([3.0, 0.0, -1.0])[permutation],
        covariance[np.ix_(permutation, permutation)],
        familywise_error_rate=0.05,
    )

    assert original.winner_id == permuted.winner_id == "candidate-a"


def test_every_permutation_preserves_nonexchangeable_evidence() -> None:
    """Permutation invariance includes correlated, unequal candidate variances."""
    identifiers = np.array(["candidate-a", "candidate-b", "candidate-c"])
    estimates = np.array([1.0, 0.0, -0.5], dtype=np.float64)
    covariance = np.array(
        [[0.01, 0.002, -0.001], [0.002, 0.02, 0.003], [-0.001, 0.003, 0.03]],
        dtype=np.float64,
    )
    for order_tuple in itertools.permutations(range(3)):
        order = np.array(order_tuple)
        result = assess_wald_superiority(
            identifiers[order].tolist(),
            estimates[order],
            covariance[np.ix_(order, order)],
            familywise_error_rate=0.05,
        )
        assert result.winner_id == "candidate-a"


def test_tied_small_p_bounds_follow_holm_without_order_tie_break() -> None:
    """Equal outgoing evidence is corrected as a family, not by candidate order."""
    result = assess_wald_superiority(
        ("candidate-a", "candidate-b", "candidate-c"),
        np.array([3.0, 0.0, 0.0], dtype=np.float64),
        np.diag(np.array([0.01, 0.01, 0.01], dtype=np.float64)),
        familywise_error_rate=0.05,
    )
    outgoing = [
        comparison
        for comparison in result.comparisons
        if comparison.candidate_id == "candidate-a"
    ]
    assert len(outgoing) == 2
    assert outgoing[0].p_value_upper_bound == outgoing[1].p_value_upper_bound
    assert all(comparison.null_rejected for comparison in outgoing)


def test_multiple_directional_rejections_are_indeterminate() -> None:
    """A caller-supplied high alpha cannot turn non-uniqueness into an error."""
    result = assess_wald_superiority(
        ("candidate-a", "candidate-b"),
        np.array([0.15, 0.0], dtype=np.float64),
        np.diag(np.array([0.5, 0.5], dtype=np.float64)),
        familywise_error_rate=0.9,
    )
    assert result.winner_id is None
    assert result.decision == "indeterminate"
    assert all(comparison.null_rejected for comparison in result.comparisons)


def test_equal_candidates_do_not_receive_an_order_tie_break() -> None:
    """The null case remains indeterminate for either candidate ordering."""
    covariance = np.diag([0.01, 0.01])
    forward = assess_wald_superiority(
        ("candidate-a", "candidate-b"),
        np.array([0.5, 0.5]),
        covariance,
        familywise_error_rate=0.05,
    )
    reverse = assess_wald_superiority(
        ("candidate-b", "candidate-a"),
        np.array([0.5, 0.5]),
        covariance,
        familywise_error_rate=0.05,
    )

    assert forward.winner_id is None
    assert reverse.winner_id is None


@pytest.mark.parametrize(
    ("estimates", "covariance", "message"),
    [
        (np.array([0.5, np.nan]), np.eye(2), "estimates must be finite"),
        (
            np.array([0.5, 0.4]),
            np.array([[1.0, 0.2], [0.1, 1.0]]),
            "covariance must be symmetric",
        ),
        (np.array([0.5, 0.4]), np.array([[1.0, 2.0], [2.0, 1.0]]), "covariance must be positive definite"),
    ],
)
def test_invalid_uncertainty_fails_closed(
    estimates: np.ndarray,
    covariance: np.ndarray,
    message: str,
) -> None:
    """Nonfinite or unidentified uncertainty cannot produce a winner."""
    with pytest.raises(ValueError, match=message):
        assess_wald_superiority(
            ("candidate-a", "candidate-b"),
            estimates,
            covariance,
            familywise_error_rate=0.05,
        )


@pytest.mark.parametrize(
    "dtype",
    [np.int64, np.uint64, np.longdouble],
)
def test_lossy_numeric_dtypes_fail_before_rust_marshalling(dtype: np.dtype) -> None:
    """Evidence is never silently rounded into a different float64 problem."""
    with pytest.raises(ValueError, match="native float64"):
        assess_wald_superiority(
            ("candidate-a", "candidate-b"),
            np.array([2**53 + 1, 2**53], dtype=dtype),
            np.eye(2, dtype=np.float64),
            familywise_error_rate=0.05,
        )

    with pytest.raises(ValueError, match="native float64"):
        assess_wald_superiority(
            ("candidate-a", "candidate-b"),
            np.array([1.0, 0.0], dtype=np.float64),
            np.eye(2, dtype=dtype),
            familywise_error_rate=0.05,
        )
