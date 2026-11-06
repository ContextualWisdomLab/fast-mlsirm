"""Row-level ownership regressions for built-in many-facet responses."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.facets as facets


class _MutatingDenseOwner:
    """Mutate the caller row after the first dense-owner write."""

    def __init__(self, array: np.ndarray, source_row: list[float]) -> None:
        self._array = array
        self._source_row = source_row
        self._mutated = False

    def __setitem__(self, key: object, value: object) -> None:
        self._array[key] = value
        if not self._mutated:
            self._mutated = True
            self._source_row[:] = [1.0, 1.0]

    def __array__(self, dtype: object = None) -> np.ndarray:
        return np.asarray(self._array, dtype=dtype)


def test_builtin_response_snapshot_never_mixes_two_states_of_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One admitted rater row is observed from one coherent built-in snapshot."""

    source = [[[0.0, 0.0]]]
    source_row = source[0][0]
    original_empty = facets.np.empty

    def mutating_empty(*args: object, **kwargs: object) -> _MutatingDenseOwner:
        array = original_empty(*args, **kwargs)
        return _MutatingDenseOwner(array, source_row)

    monkeypatch.setattr(facets.np, "empty", mutating_empty)

    snapshot = facets._snapshot_builtin_response_tree(
        source,
        n_persons=1,
        n_items=1,
        n_raters=2,
    )

    np.testing.assert_array_equal(snapshot, np.array([[[0.0, 0.0]]]))
