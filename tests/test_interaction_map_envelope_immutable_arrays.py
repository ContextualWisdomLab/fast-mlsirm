"""Mutation-hardening regression for residual interaction-map public evidence."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import residual_interaction_map_envelope


_ARRAY_FIELDS = (
    "person_indices",
    "item_indices",
    "person_coordinates",
    "item_coordinates",
    "singular_values",
    "axis_shares",
    "observed",
    "expected",
    "residual",
    "distance",
    "reconstruction",
    "unexplained",
    "cross_share",
)


def test_public_envelope_arrays_cannot_reenable_writes() -> None:
    """Digest-bound array evidence must not regain write access after publication."""
    result = residual_interaction_map_envelope(
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        person_ids=["person-a", "person-b"],
        item_ids=["item-a", "item-b"],
        axis_count=2,
    )

    for field_name in _ARRAY_FIELDS:
        value = getattr(result, field_name)
        assert isinstance(value, np.ndarray)
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value.setflags(write=True)
