"""Fail-first resource contracts for response-time calibration iteration controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.rt import fit_response_times, fit_speed_accuracy


class _BombRtCore:
    """Compiled-core sentinel proving invalid controls fail before numerical work."""

    def fit_rt_lognormal(self, *args, **kwargs):
        raise AssertionError("standalone RT core reached with invalid max_iter")

    def fit_speed_accuracy_covariance(self, *args, **kwargs):
        raise AssertionError("joint RT core reached with invalid max_iter")


def _install_bomb_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a compiled-core sentinel with both RT entrypoints present."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _BombRtCore())


def test_fit_response_times_rejects_oversized_max_iter_before_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone RT calibration must bound caller iterations before Rust execution."""
    _install_bomb_core(monkeypatch)
    times = np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64)

    with pytest.raises(ValueError, match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}"):
        fit_response_times(times, max_iter=MAX_MAX_ITER + 1)


def test_fit_speed_accuracy_rejects_oversized_max_iter_before_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint RT calibration must enforce the same public iteration ceiling."""
    _install_bomb_core(monkeypatch)
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    times = np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64)
    a = np.array([1.0, 1.2], dtype=np.float64)
    b = np.array([0.0, -0.2], dtype=np.float64)
    alpha = np.array([1.1, 1.3], dtype=np.float64)
    beta = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}"):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            max_iter=MAX_MAX_ITER + 1,
        )
