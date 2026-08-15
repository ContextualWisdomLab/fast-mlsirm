"""Regression tests for inert answer-copying integer controls.

Production answer-copying arithmetic remains Rust-owned.  These tests prove the
Python boundary establishes trusted integer identities without executing caller
conversion/comparison callbacks before native-core discovery.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.security import k_index, k_variants, wollack_omega


def _responses() -> np.ndarray:
    """Return a small complete binary matrix with a nondegenerate source row."""
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )


def _omega_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid Wollack-omega vectors for three response options."""
    copier = np.array([0, 1, 2, 0], dtype=np.int64)
    source = np.array([0, 1, 0, 2], dtype=np.int64)
    probs = np.full((4, 3), 1.0 / 3.0)
    return copier, source, probs


def test_wollack_omega_rejects_hostile_option_count_without_callbacks():
    """A caller-defined ``int`` subclass must be rejected before comparison."""
    callbacks: list[str] = []

    class HostileInt(int):
        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            return False

        def __int__(self) -> int:
            callbacks.append("int")
            return 3

    copier, source, probs = _omega_inputs()
    with patch("fast_mlsirm.fitstats._core_module") as core_loader:
        with pytest.raises(ValueError, match="n_options must be an integer"):
            wollack_omega(copier, source, probs, HostileInt(3))
    assert callbacks == []
    core_loader.assert_not_called()


@pytest.mark.parametrize("function", [k_index, k_variants])
def test_answer_copying_indices_reject_hostile_row_indices_without_callbacks(function):
    """K-family row indices must not execute caller ordering/conversion hooks."""
    callbacks: list[str] = []

    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            callbacks.append("lt")
            return False

        def __int__(self) -> int:
            callbacks.append("int")
            return 0

    with patch("fast_mlsirm.fitstats._core_module") as core_loader:
        with pytest.raises(ValueError, match="copier must be an integer row index"):
            function(_responses(), HostileInt(0), 1)
    assert callbacks == []
    core_loader.assert_not_called()


@pytest.mark.parametrize("function", [k_index, k_variants])
def test_answer_copying_indices_accept_exact_numpy_integer_scalars(function):
    """Genuine NumPy integer scalars remain compatible before Rust discovery."""
    with patch("fast_mlsirm.fitstats._core_module", return_value=None) as core_loader:
        with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
            function(_responses(), np.int64(0), np.uint64(1))
    core_loader.assert_called_once_with()


def test_wollack_omega_accepts_exact_numpy_integer_scalar():
    """A genuine NumPy option count reaches the established native boundary."""
    copier, source, probs = _omega_inputs()
    with patch("fast_mlsirm.fitstats._core_module", return_value=None) as core_loader:
        with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
            wollack_omega(copier, source, probs, np.int64(3))
    core_loader.assert_called_once_with()
