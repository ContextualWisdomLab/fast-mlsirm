"""Ownership regressions for Mokken response admission."""

from __future__ import annotations

import numpy as np

from fast_mlsirm import mokken


def test_top_level_int64_ndarray_scores_are_package_owned() -> None:
    """Admitted Rust-bound scores must not alias caller-owned response storage."""
    responses = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    scores, n_persons, n_items = mokken._validated_scores(responses)

    assert (n_persons, n_items) == (3, 2)
    assert scores.dtype == np.int64
    assert scores.flags.c_contiguous
    assert not np.shares_memory(scores, responses)
    assert scores.tolist() == [0, 1, 1, 0, 1, 1]

    responses[0, 0] = 1
    responses[2, 1] = 0

    assert scores.tolist() == [0, 1, 1, 0, 1, 1]
