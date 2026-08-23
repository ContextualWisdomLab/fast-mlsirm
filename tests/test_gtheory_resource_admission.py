"""Resource-admission regressions for Rust-backed G-theory entry points."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


def _unexpected_core_discovery(name: str) -> object:
    """Fail if invalid G-theory evidence reaches native capability lookup."""
    raise AssertionError(f"invalid G-theory evidence reached Rust discovery: {name}")


def test_gtheory_pi_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """One-facet score trees fail on rank before dense NumPy materialization."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank G-theory evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be a 2-D persons x items array"):
        gtheory.gtheory_pi([[[0.0, 1.0]]], n_i_prime=[2])


def test_gtheory_pio_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """Two-facet score trees fail on rank before dense NumPy materialization."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank G-theory evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=r"data must be a 3-D persons x items x occasions array",
    ):
        gtheory.gtheory_pio([[[[0.0, 1.0]]]], n_prime=[(2, 2)])


def test_phi_lambda_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """Mastery G-theory score trees share the two-dimensional rank boundary."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank Phi(lambda) evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be a 2-D persons x items array"):
        gtheory.phi_lambda([[[0.0, 1.0]]], 0.5, n_i_prime=[2])


def test_gtheory_pi_rejects_oversized_exact_view_before_contiguous_copy(
    monkeypatch,
) -> None:
    """Logical cell bounds apply before an oversized view is densely copied."""
    oversized = np.broadcast_to(
        np.array(0.0, dtype=np.float64),
        (1, gtheory.MAX_GTHEORY_SCORE_CELLS + 1),
    )

    def unexpected_contiguous_copy(*args, **kwargs):
        raise AssertionError("oversized G-theory evidence reached contiguous copy")

    monkeypatch.setattr(gtheory.np, "ascontiguousarray", unexpected_contiguous_copy)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=rf"data exceeds the {gtheory.MAX_GTHEORY_SCORE_CELLS}-cell G-theory limit",
    ):
        gtheory.gtheory_pi(oversized, n_i_prime=[2])
