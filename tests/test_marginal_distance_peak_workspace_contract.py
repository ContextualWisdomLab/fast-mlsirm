"""Peak-allocation contracts for bounded marginal pairwise distances."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


def test_distance_preflight_counts_output_and_live_row_norm_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output that fits alone still fails when its live norms exceed the budget."""

    # Output: 12 * 10 * 8 = 960 bytes. Live row norms add
    # (12 + 10) * 8 = 176 bytes, so the true peak is 1,136 bytes.
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=12,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_pairwise_helper_applies_the_same_peak_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct helper use cannot bypass the estimator's peak preflight."""

    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)
    left = np.zeros((12, 2), dtype=np.float64)
    right = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._pairwise_euclidean_distances(
            left,
            right,
            eps_distance=1e-8,
        )


def test_pairwise_peak_equal_to_limit_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The byte ceiling is inclusive and deterministic at the exact boundary."""

    peak_bytes = (12 * 10 + 12 + 10) * np.dtype(np.float64).itemsize
    monkeypatch.setattr(
        marginal,
        "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES",
        peak_bytes,
    )
    left = np.zeros((12, 2), dtype=np.float64)
    right = np.zeros((10, 2), dtype=np.float64)

    result = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-8,
    )

    assert result.shape == (12, 10)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.sqrt(1e-8))
