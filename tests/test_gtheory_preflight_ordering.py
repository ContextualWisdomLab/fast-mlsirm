"""Fail-first ordering contracts for public G-theory semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


def _unexpected_core_discovery(name: str) -> object:
    """Fail if an invalid semantic control reaches native capability discovery."""
    raise AssertionError(f"invalid G-theory control reached Rust discovery: {name}")


def _pi_data() -> np.ndarray:
    """Return a minimal two-dimensional score matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _pio_data() -> np.ndarray:
    """Return a minimal three-dimensional score tensor."""
    return np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )


def test_gtheory_pi_rejects_invalid_size_before_core_discovery(monkeypatch) -> None:
    """Invalid one-facet D-study sizes fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[0])


def test_gtheory_pio_rejects_invalid_pair_before_core_discovery(monkeypatch) -> None:
    """Invalid two-facet D-study pairs fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_pio_data(), n_prime=[(2, 0)])


def test_phi_lambda_rejects_invalid_cut_before_core_discovery(monkeypatch) -> None:
    """Invalid mastery cuts fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), np.inf, n_i_prime=[2])


def test_phi_lambda_rejects_invalid_size_before_core_discovery(monkeypatch) -> None:
    """Invalid Phi(lambda) D-study sizes fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.phi_lambda(_pi_data(), 0.5, n_i_prime=[0])
