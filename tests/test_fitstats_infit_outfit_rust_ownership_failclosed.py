"""Fail-first ownership contracts for public infit/outfit statistics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats_module
from fast_mlsirm.fitstats import infit_outfit


def _fixture() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Return a small valid dichotomous fixture for ownership tests."""
    responses = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(responses.shape[1], dtype=np.float64),
        b=np.linspace(-0.5, 0.5, responses.shape[1]),
        zeta=np.zeros((responses.shape[1], 1), dtype=np.float64),
        tau=-30.0,
        theta=np.linspace(-1.0, 1.0, responses.shape[0])[:, None],
        xi=np.zeros((responses.shape[0], 1), dtype=np.float64),
    )
    return responses, factor_id, params


def _forbidden_python_exp(*_args, **_kwargs):
    raise AssertionError("Python infit/outfit probability arithmetic executed")


def test_infit_outfit_missing_core_fails_before_python_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Rust core must fail closed before NumPy statistic arithmetic."""
    responses, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: None)
    monkeypatch.setattr(fitstats_module.np, "exp", _forbidden_python_exp)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        infit_outfit(responses, factor_id, params, "MIRT")


def test_infit_outfit_incomplete_core_fails_before_python_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete compiled core must not select the Python fallback."""
    responses, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: SimpleNamespace())
    monkeypatch.setattr(fitstats_module.np, "exp", _forbidden_python_exp)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        infit_outfit(responses, factor_id, params, "MIRT")


def test_infit_outfit_dispatches_to_rust_without_python_probability_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatible cores own the public infit/outfit numerical result."""
    responses, factor_id, params = _fixture()

    class RecordingCore:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def infit_outfit_stat(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {
                "infit": np.array([0.91, 0.92, 0.93, 0.94], dtype=np.float64),
                "outfit": np.array([1.01, 1.02, 1.03, 1.04], dtype=np.float64),
            }

    core = RecordingCore()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: core)
    monkeypatch.setattr(fitstats_module.np, "exp", _forbidden_python_exp)

    result = infit_outfit(responses, factor_id, params, "MIRT")

    assert len(core.calls) == 1
    assert np.array_equal(result["infit"], np.array([0.91, 0.92, 0.93, 0.94]))
    assert np.array_equal(result["outfit"], np.array([1.01, 1.02, 1.03, 1.04]))
