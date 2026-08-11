"""Ownership contracts for public CAT item information."""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
from fast_mlsirm.test_design import item_information
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small one-dimensional calibrated bank."""
    discrimination = np.array([0.8, 1.1, 1.4, 1.7], dtype=np.float64)
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(discrimination),
        b=np.array([-1.0, -0.2, 0.4, 1.2], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_public_item_information_delegates_to_rust(monkeypatch) -> None:
    """The public information vector must be transported from the Rust owner."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_information(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "item_info": [0.125, 0.25, 0.5, 0.75],
            "test_info": [1.625],
        }

    monkeypatch.setattr(core, "bank_information", fake_information)
    result = item_information(
        bank,
        factor_id,
        theta=np.array([0.25], dtype=np.float64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert np.array_equal(result, np.array([0.125, 0.25, 0.5, 0.75]))
    args, kwargs = calls[0]
    assert int(args[2]) == 1
    assert kwargs["model"] == "MIRT"
    assert kwargs["device"] == "auto"


def test_public_item_information_preserves_inputs(monkeypatch) -> None:
    """Marshalling into the Rust owner must not mutate caller arrays."""
    bank, factor_id = _bank()
    theta = np.array([-0.5], dtype=np.float64)
    theta_before = theta.copy()
    factor_before = factor_id.copy()

    def fake_information(*args, **kwargs):
        return {"item_info": [1.0, 2.0, 3.0, 4.0], "test_info": [10.0]}

    monkeypatch.setattr(core, "bank_information", fake_information)
    item_information(bank, factor_id, theta=theta, model="MIRT")

    assert np.array_equal(theta, theta_before)
    assert np.array_equal(factor_id, factor_before)
