"""Fail-first trust-boundary tests for classification cut-score controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.classification as classification


class _HostileFloat(float):
    """A float subclass whose coercion is observable and must never run."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile __float__ callback executed")


def _unexpected_core_discovery(name: str) -> object:
    """Fail if an invalid cut-score reaches native capability discovery."""
    raise AssertionError(f"invalid cut-score reached Rust discovery: {name}")


def test_rudner_rejects_untrusted_cut_before_core_and_without_callback(
    monkeypatch,
) -> None:
    """Rudner cut controls reject scalar subclasses before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            [_HostileFloat(0.0)],
        )

    assert _HostileFloat.calls == 0


def test_lee_rejects_untrusted_cut_before_core_and_without_callback(monkeypatch) -> None:
    """Lee cut controls reject scalar subclasses before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.lee_classification(
            np.array([[0.25, 0.75]], dtype=np.float64),
            [_HostileFloat(1.0)],
        )

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("cut", [True, np.bool_(False), np.inf, np.float64(np.nan)])
def test_rudner_rejects_invalid_cut_identity_or_finiteness_before_core(
    monkeypatch,
    cut: object,
) -> None:
    """Boolean and non-finite cuts fail closed before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            [cut],
        )
