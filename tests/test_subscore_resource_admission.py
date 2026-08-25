"""Resource-bound regressions for subscore scientific-evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.subscores as subscores


def _unexpected_core_discovery():
    """Fail if rejected evidence reaches the compiled-core discovery boundary."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


def _unexpected_asarray(*args, **kwargs):
    """Fail if oversized evidence reaches NumPy materialization."""

    raise AssertionError("oversized evidence must fail before np.asarray")


def test_subscore_rejects_oversized_exact_response_before_numpy(monkeypatch) -> None:
    """Logical response size is bounded before any dense/materializing NumPy call."""

    oversized = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (3, 6_666_667),
    )
    assert oversized.size == 20_000_001

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(ValueError, match="responses must contain at most 20,000,000 logical cells"):
        subscores.subscore_analysis(oversized, [0, 0, 1, 1])


def test_subscore_charges_exact_numpy_leaf_before_parent_materialization(monkeypatch) -> None:
    """A tiny-backing broadcast leaf cannot bypass the logical-cell envelope."""

    oversized_row = np.broadcast_to(
        np.array([0.0], dtype=np.float64),
        (20_000_001,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(ValueError, match="responses must contain at most 20,000,000 logical cells"):
        subscores.subscore_analysis(responses, [0, 0, 1, 1])
