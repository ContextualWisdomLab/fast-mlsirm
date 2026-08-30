"""Mutable native-vector snapshot evidence for the facets result bridge."""

from __future__ import annotations

import numpy as np

import fast_mlsirm.facets as facets


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
