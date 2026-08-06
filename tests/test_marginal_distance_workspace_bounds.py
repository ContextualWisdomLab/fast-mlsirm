"""Resource and numerical contracts for marginal latent-space distances."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


def test_pairwise_distance_matches_explicit_broadcast() -> None:
    """The bounded identity must reproduce the former Euclidean equation."""
    left = np.array([[0.0, 0.0], [1.5, -0.5], [-2.0, 3.0]])
    right = np.array([[0.25, -0.75], [1.0, 2.0], [-2.0, 3.0], [4.0, -1.0]])
    expected = np.sqrt(
        1e-8 + np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    )

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-8,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_pairwise_distance_handles_zero_and_roundoff() -> None:
    """Identical and nearly identical rows remain finite and non-negative."""
    left = np.array([[1.0, 2.0, 3.0], [1e8, 1e8, 1e8]])
    right = left.copy()

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-12,
    )

    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(np.diag(actual), np.sqrt(1e-12), rtol=1e-12)


def test_distance_workspace_guard_fails_before_pairwise_allocation(monkeypatch) -> None:
    """Oversized output dimensions fail without asking NumPy for the matrix."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_WORKING_SET", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=101,
            n_x=10,
            latent_dim=3,
            uses_space=True,
        )


def test_distance_workspace_guard_fails_before_gradient_allocation(monkeypatch) -> None:
    """The item-local node-by-dimension gradient workspace is also bounded."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_WORKING_SET", 1_000)

    with pytest.raises(ValueError, match="item distance gradient workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=2,
            n_x=501,
            latent_dim=2,
            uses_space=True,
        )


@pytest.mark.parametrize(
    "dimensions",
    [(-1, 2), (True, 2), (2.5, 2), (2, False)],
)
def test_checked_workspace_product_rejects_invalid_dimensions(dimensions) -> None:
    """Malformed dimensions never reach multiplication or allocation."""
    with pytest.raises(ValueError, match="non-negative integers"):
        marginal._checked_marginal_workspace_product("test workspace", *dimensions)


def test_checked_workspace_product_rejects_invalid_limit() -> None:
    """A malformed caller-supplied ceiling cannot weaken the resource gate."""
    with pytest.raises(ValueError, match="workspace limit"):
        marginal._checked_marginal_workspace_product(
            "test workspace",
            2,
            3,
            limit=True,
        )


def test_non_spatial_models_skip_distance_workspace_contract() -> None:
    """MIRT does not incur latent-space distance workspaces."""
    marginal._validate_marginal_distance_workspaces(
        n_items=10**9,
        n_x=10**9,
        latent_dim=3,
        uses_space=False,
    )


def test_distance_helpers_reject_invalid_shapes_and_epsilon() -> None:
    """Direct helper use fails closed for malformed numerical contracts."""
    with pytest.raises(ValueError, match="two-dimensional"):
        marginal._pairwise_euclidean_distances(
            np.zeros(2),
            np.zeros((2, 1)),
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="share one latent dimension"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 2)),
            np.zeros((3, 1)),
            eps_distance=1e-8,
        )
    for invalid_epsilon in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="eps_distance"):
            marginal._pairwise_euclidean_distances(
                np.zeros((2, 2)),
                np.zeros((3, 2)),
                eps_distance=invalid_epsilon,
            )


def test_covariate_path_has_no_three_dimensional_distance_broadcast() -> None:
    """The removed item-by-node-by-dimension temporary cannot silently return."""
    source = inspect.getsource(marginal.fit_marginal_numpy)
    assert "x_grid[None, :, :] - zeta[:, None, :]" not in source
