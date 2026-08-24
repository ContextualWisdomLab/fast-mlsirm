"""Bounded-materialization contracts for ATA target-curve evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a tiny one-dimensional calibrated bank for resource tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_information_grid_limit_precedes_scalar_scan_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provably oversized dense grid must fail before per-cell or scoring work."""
    bank, factor_id = _bank()
    monkeypatch.setattr(ata, "MAX_ATA_INFORMATION_CELLS", 4, raising=False)
    scalar_calls = 0
    information_calls = 0

    def unexpected_scalar_validation(*_args: object, **_kwargs: object) -> None:
        nonlocal scalar_calls
        scalar_calls += 1
        raise AssertionError("oversized target grid reached scalar validation")

    def unexpected_item_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        raise AssertionError("oversized target grid reached psychometric scoring")

    monkeypatch.setattr(ata, "_require_lossless_float64_scalar", unexpected_scalar_validation)
    monkeypatch.setattr(ata, "item_information", unexpected_item_information)
    target_thetas = np.broadcast_to(np.array([0.0], dtype=np.float64), (2,))

    with pytest.raises(
        ValueError,
        match=r"target information matrix exceeds the 4-cell ATA limit",
    ):
        ata.item_information_matrix(bank, factor_id, target_thetas, model="MIRT")

    assert scalar_calls == 0
    assert information_calls == 0


def test_malformed_builtin_fanout_has_bounded_traversal_before_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell container fan-out must exhaust a package work budget first."""
    monkeypatch.setattr(ata, "MAX_ATA_TARGET_CELLS", 1, raising=False)
    numpy_calls = 0
    original_asarray = ata.np.asarray

    def unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal numpy_calls
        numpy_calls += 1
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(ata.np, "asarray", unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"target_info exceeds the 1-cell ATA evidence limit",
    ):
        ata._trusted_real_array([[], [], []], "target_info")

    assert numpy_calls == 0
