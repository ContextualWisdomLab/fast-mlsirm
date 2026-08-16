"""Regression tests for bounded GPCM expected-count accumulation."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators import marginal


def _reference_expected_counts(
    post: np.ndarray, item_responses: np.ndarray, n_cat: int
) -> np.ndarray:
    """Return the pre-optimization expected-count reference."""
    return np.stack(
        [post[item_responses == k].sum(axis=0) for k in range(n_cat)], axis=1
    )


def test_gpcm_expected_counts_matches_reference() -> None:
    """The bounded accumulator must preserve the historical expected counts."""
    post = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
            [0.4, 0.4, 0.2],
            [0.3, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    responses = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    scores = np.arange(3, dtype=np.float64)

    expected = _reference_expected_counts(post, responses, scores.size)
    actual = marginal._gpcm_expected_counts(post, responses, scores)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_fit_gpcm_numpy_accumulates_expected_counts_per_item(monkeypatch) -> None:
    """GPCM fitting must not retain a persons-by-items-by-categories mask."""
    original = marginal._gpcm_expected_counts
    calls: list[tuple[int, int]] = []

    def checked(
        post: np.ndarray, item_responses: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        assert item_responses.ndim == 1
        assert item_responses.shape[0] == post.shape[0]
        calls.append((item_responses.size, scores.size))
        return original(post, item_responses, scores)

    monkeypatch.setattr(marginal, "_gpcm_expected_counts", checked)
    y = np.array(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )

    marginal.fit_gpcm_numpy(y, n_cat=3, q_theta=7, max_iter=1)

    assert calls == [(y.shape[0], 3)] * y.shape[1]
