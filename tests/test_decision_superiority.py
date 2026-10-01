"""Contract tests for the Rust-owned Holm--Wald superiority foundation."""

from __future__ import annotations

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
