"""Peak-live scratch contracts for bounded marginal distance workspaces."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


def test_distance_preflight_counts_pairwise_row_norm_scratch(monkeypatch) -> None:
    """A matrix that fits alone still fails when its live row norms exceed budget."""
    # 10 x 10 float64 output = 800 bytes; two 10-element row-norm vectors add
    # 160 bytes. The output alone fits under 900 bytes, but the proven live
    # distance workspace does not.
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 900)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=10,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_direct_pairwise_helper_enforces_the_same_live_scratch_budget(
    monkeypatch,
) -> None:
    """Direct helper use cannot bypass output-plus-row-norm byte accounting."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 900)
    left = np.zeros((10, 2), dtype=np.float64)
    right = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._pairwise_euclidean_distances(
            left,
            right,
            eps_distance=1e-8,
        )
