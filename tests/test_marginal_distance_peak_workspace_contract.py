"""Peak-allocation contracts for bounded marginal pairwise distances."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal

_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_BOOL_BYTES = np.dtype(np.bool_).itemsize


def test_distance_preflight_counts_output_norms_and_result_finiteness_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output that fits alone still fails when all simultaneous temporaries peak."""

    # Output: 12 * 10 * 8 = 960 bytes. Live row norms add
    # (12 + 10) * 8 = 176 bytes and the post-result finiteness mask adds
    # 12 * 10 = 120 bytes, so the true post-BLAS peak is 1,256 bytes.
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=12,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_post_result_finiteness_mask_is_part_of_the_peak_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old output-plus-norm boundary is rejected once the live bool mask is counted."""

    old_peak_without_mask = (12 * 10 + 12 + 10) * _FLOAT64_BYTES
    monkeypatch.setattr(
        marginal,
        "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES",
        old_peak_without_mask,
    )

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=12,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_input_finiteness_mask_is_preflighted_before_numpy_mask_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wide direct-helper input fails before ``np.isfinite`` can allocate its mask."""

    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 100)
    left = np.zeros((1, 101), dtype=np.float64)
    right = np.zeros((1, 101), dtype=np.float64)

    def forbidden_isfinite(*_args: object, **_kwargs: object) -> np.ndarray:
        """Prove resource rejection happens before temporary-producing finiteness checks."""

        raise AssertionError("np.isfinite must not run before workspace preflight")

    monkeypatch.setattr(marginal.np, "isfinite", forbidden_isfinite)
    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._pairwise_euclidean_distances(
            left,
            right,
            eps_distance=1e-8,
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

    output_bytes = 12 * 10 * _FLOAT64_BYTES
    norm_bytes = (12 + 10) * _FLOAT64_BYTES
    output_mask_bytes = 12 * 10 * _BOOL_BYTES
    input_mask_peak = max(12 * 2, 10 * 2) * _BOOL_BYTES
    peak_bytes = max(output_bytes + norm_bytes + output_mask_bytes, input_mask_peak)
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
