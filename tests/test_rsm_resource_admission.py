"""Resource-bound regressions for rating-scale-model response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rsm as rsm
from fast_mlsirm import fit_rsm


MAX_RSM_RESPONSE_CELLS = 20_000_000


def _unexpected_asarray(*args, **kwargs):
    """Fail if NumPy materialization runs before the RSM resource gate."""

    raise AssertionError("NumPy materialization executed before RSM resource admission")


def test_oversized_exact_rsm_response_array_fails_before_materialization(monkeypatch):
    """A huge broadcast response view is rejected without a dense copy."""

    responses = np.broadcast_to(
        np.array([[0, 1]], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS // 2 + 1, 2),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell RSM evidence budget",
    ):
        fit_rsm(responses, n_cat=2)


def test_oversized_exact_numpy_row_fails_before_sequence_materialization(monkeypatch):
    """An oversized trusted NumPy row is bounded before list stacking."""

    row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS + 1,),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell RSM evidence budget",
    ):
        fit_rsm([row], n_cat=2)


def test_rsm_resource_preflight_preserves_small_shared_rows():
    """Repeated inert built-in rows remain valid inside the bounded envelope."""

    shared_row = [0, 1]
    assert rsm._trusted_response_source([shared_row, shared_row]) == [
        shared_row,
        shared_row,
    ]
