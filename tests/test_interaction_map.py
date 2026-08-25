"""Public residual interaction-map contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from fast_mlsirm import residual_interaction_map


def test_residual_interaction_map_preserves_rank_one_reconstruction() -> None:
    """The Rust public API reconstructs a known rank-one residual exactly."""
    observed = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected = np.ones((2, 2))

    result = residual_interaction_map(observed, expected, axis_count=2)

    np.testing.assert_allclose(result.reconstruction, observed - expected, atol=1e-12)
    np.testing.assert_allclose(result.residual, observed - expected, atol=1e-12)
    assert np.all(np.isfinite(result.distance))
    assert np.all(result.distance >= 0.0)
    np.testing.assert_allclose(result.axis_shares, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.unexplained, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.cross_share, 0.0, atol=1e-12)


def test_residual_interaction_map_reports_scored_and_complete_case_coverage() -> None:
    """Coverage distinguishes scored rows from rows admitted to the map."""
    observed = np.array([[2.0, np.nan], [1.0, 2.0]])
    result = residual_interaction_map(observed, np.ones((2, 2)), axis_count=2)

    assert result.scored_person_count == 2
    assert result.scored_item_count == 2
    np.testing.assert_array_equal(result.person_indices, [1])
    np.testing.assert_array_equal(result.item_indices, [0, 1])


@pytest.mark.parametrize("axis_count", [0, -1])
def test_residual_interaction_map_rejects_nonpositive_axis_count(
    axis_count: int,
) -> None:
    """A consumer must request at least one reader-visible map axis."""
    with pytest.raises(ValueError, match="axis_count"):
        residual_interaction_map(
            np.ones((1, 1)), np.zeros((1, 1)), axis_count=axis_count
        )


@pytest.mark.parametrize("axis_count", [True, 1.5])
def test_residual_interaction_map_rejects_nonintegral_axis_count(
    axis_count: object,
) -> None:
    """Boolean and real-valued controls do not silently become dimensions."""
    with pytest.raises(TypeError, match="axis_count"):
        residual_interaction_map(
            np.ones((1, 1)),
            np.zeros((1, 1)),
            axis_count=axis_count,  # type: ignore[arg-type]
        )
