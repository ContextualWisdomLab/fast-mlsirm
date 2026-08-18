"""Regression coverage for bounded-memory GPCM NumPy reference counts."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators.marginal import _gpcm_expected_counts


def test_gpcm_expected_counts_reuses_one_item_workspace() -> None:
    """Expected counts use a reusable 2-D item mask, never a response tensor."""
    post = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )
    responses = np.array(
        [
            [0, 2],
            [1, 1],
            [2, 0],
            [1, 2],
        ],
        dtype=np.int64,
    )
    workspace = np.empty((responses.shape[0], 3), dtype=np.float64)
    person_index = np.arange(responses.shape[0])

    first = _gpcm_expected_counts(post, responses[:, 0], workspace, person_index)
    second = _gpcm_expected_counts(post, responses[:, 1], workspace, person_index)

    first_reference = np.stack(
        [post[responses[:, 0] == category].sum(axis=0) for category in range(3)],
        axis=1,
    )
    second_reference = np.stack(
        [post[responses[:, 1] == category].sum(axis=0) for category in range(3)],
        axis=1,
    )
    np.testing.assert_allclose(first, first_reference)
    np.testing.assert_allclose(second, second_reference)
    assert workspace.shape == (responses.shape[0], 3)
