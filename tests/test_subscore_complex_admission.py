"""Regression contracts for subscore scientific-evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.subscores import subscore_analysis


def _unexpected_core_discovery():
    """Fail if rejected evidence reaches the compiled-core discovery boundary."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


class _HostileArrayProvider:
    """Array provider whose protocol must never run during admission."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        raise AssertionError("caller __array__ must not execute during admission")


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


def test_subscore_rejects_response_array_provider_before_callback(monkeypatch) -> None:
    """Caller protocols cannot synthesize the person-by-item response evidence."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = _HostileArrayProvider(_valid_responses())

    with pytest.raises(ValueError, match="responses must be real-numeric evidence"):
        subscore_analysis(responses, np.array([0, 0, 1, 1], dtype=np.int64))

    assert responses.calls == 0


def test_subscore_rejects_group_array_provider_before_callback(monkeypatch) -> None:
    """Caller protocols cannot synthesize the item-to-subscale partition."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    groups = _HostileArrayProvider(np.array([0, 0, 1, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="groups must be real-numeric evidence"):
        subscore_analysis(_valid_responses(), groups)

    assert groups.calls == 0


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
