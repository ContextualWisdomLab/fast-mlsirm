"""Regression contracts for subscore complex-evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.subscores import subscore_analysis


def _unexpected_core_discovery():
    """Fail if rejected evidence reaches the compiled-core discovery boundary."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


def _valid_responses() -> np.ndarray:
    """Return a complete response matrix satisfying the Python shape boundary."""

    return np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 3.0, 4.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_subscore_rejects_complex_responses_before_real_narrowing(monkeypatch) -> None:
    """Imaginary response evidence cannot be projected onto a real score matrix."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = _valid_responses().astype(np.complex128)
    responses[1, 2] += 1.0j

    with pytest.raises(ValueError, match="responses must be real-valued"):
        subscore_analysis(responses, np.array([0, 0, 1, 1], dtype=np.int64))


def test_subscore_rejects_complex_groups_before_real_narrowing(monkeypatch) -> None:
    """Imaginary group evidence cannot be projected onto another partition."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    groups = np.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="groups must be real-valued"):
        subscore_analysis(_valid_responses(), groups)
