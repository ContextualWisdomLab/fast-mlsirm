"""Trust-boundary regressions for Warm WLE item and response evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.wle import score_wle, score_wle_poly


class _HostileArrayProvider:
    """Record attempts to execute a caller-owned NumPy array protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller array protocol must not run")


class _HostileFloatProvider:
    """Record attempts to execute caller-owned numeric conversion."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller float protocol must not run")


def _forbid_core(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def unexpected_core():
        calls.append(1)
        raise AssertionError("compiled-core discovery must not run")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    return calls


def test_score_wle_rejects_top_level_array_provider_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="a must be a numeric array"):
        score_wle(hostile, np.array([0.0]), np.array([[1.0]]))

    assert hostile.calls == 0
    assert core_calls == []


def test_score_wle_poly_rejects_nested_numeric_provider_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _HostileFloatProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        score_wle_poly(
            [[hostile]],
            [np.float32(1.0)],
            [[np.float32(0.0)]],
            np.int16(2),
        )

    assert hostile.calls == 0
    assert core_calls == []


def test_wle_preserves_inert_numpy_rows_and_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, np.ndarray] = {}

    class _Core:
        def score_wle(self, *args):
            seen["a"] = args[0]
            seen["responses"] = args[4]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

        def score_wle_poly(self, *args):
            seen["poly_responses"] = args[0]
            seen["slope"] = args[4]
            seen["cat_params"] = args[5]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    score_wle(
        [np.float32(1.0)],
        [np.int16(0)],
        [np.array([np.bool_(True)])],
    )
    score_wle_poly(
        [np.array([np.uint8(0)])],
        [np.float32(1.0)],
        [np.array([np.float32(0.0)])],
        np.int16(2),
    )

    assert seen["a"].dtype == np.float64
    assert seen["responses"].dtype == np.float64
    assert seen["poly_responses"].dtype == np.int64
    assert seen["slope"].dtype == np.float64
    assert seen["cat_params"].dtype == np.float64
