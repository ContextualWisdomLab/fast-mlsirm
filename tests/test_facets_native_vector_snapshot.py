"""Mutable native-vector snapshot evidence for the facets result bridge."""

from __future__ import annotations

import numpy as np

import fast_mlsirm.facets as facets


class _HostileFloat:
    """Track whether a post-snapshot native replacement gets coerced."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        return 9.0


def test_mutable_native_list_is_snapshotted_before_numpy_conversion(
    monkeypatch,
) -> None:
    """A retained native list cannot redefine evidence at the NumPy seam."""

    source = [-0.25, 0.25]
    result = {"item_difficulty": source}
    real_asarray = facets.np.asarray

    def asarray_after_native_rebind(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray:
        source[:] = [9.0, 10.0]
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(facets.np, "asarray", asarray_after_native_rebind)

    owned = facets._native_float_vector(
        result,
        "item_difficulty",
        exact_length=2,
    )

    np.testing.assert_array_equal(owned, np.array([-0.25, 0.25], dtype=np.float64))
    np.testing.assert_array_equal(source, np.array([9.0, 10.0], dtype=np.float64))


def test_post_snapshot_hostile_list_rebind_executes_no_numeric_callback(
    monkeypatch,
) -> None:
    """A hostile replacement after the bounded snapshot is never coerced."""

    source = [-0.25, 0.25]
    result = {"item_difficulty": source}
    hostile = _HostileFloat()
    real_asarray = facets.np.asarray

    def asarray_after_hostile_rebind(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray:
        source[:] = [hostile, hostile]
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(facets.np, "asarray", asarray_after_hostile_rebind)

    owned = facets._native_float_vector(
        result,
        "item_difficulty",
        exact_length=2,
    )

    np.testing.assert_array_equal(owned, np.array([-0.25, 0.25], dtype=np.float64))
    assert source[0] is hostile
    assert source[1] is hostile
    assert hostile.calls == 0
