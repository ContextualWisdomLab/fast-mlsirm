"""Trust-boundary regressions for answer-copying integer controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.security import k_index, k_variants, wollack_omega


class _HostileInt(int):
    """Integer subclass whose comparison/coercion callbacks must stay unreachable."""

    calls = 0

    def __int__(self):
        type(self).calls += 1
        return int.__int__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return int.__le__(self, other)


def _unexpected_core_discovery():
    """Fail if an invalid scalar reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid controls")


def _responses() -> np.ndarray:
    """Return complete binary response data for row-index boundary tests."""

    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _options() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid nominal-option inputs for Wollack omega."""

    copier = np.array([0, 1, 2], dtype=np.int64)
    source = np.array([0, 1, 0], dtype=np.int64)
    probs = np.full((3, 3), 1.0 / 3.0)
    return copier, source, probs


def test_wollack_rejects_integer_subclass_without_callbacks(monkeypatch):
    """Wollack option-count validation never executes caller scalar hooks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.calls = 0
    copier, source, probs = _options()

    with pytest.raises(ValueError, match="n_options must be an integer"):
        wollack_omega(copier, source, probs, _HostileInt(3))

    assert _HostileInt.calls == 0


def test_k_index_rejects_integer_subclass_without_callbacks(monkeypatch):
    """K-index row validation never executes caller scalar hooks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.calls = 0

    with pytest.raises(ValueError, match="copier must be an integer row index"):
        k_index(_responses(), _HostileInt(0), 1)

    assert _HostileInt.calls == 0


def test_k_variants_rejects_integer_subclass_without_callbacks(monkeypatch):
    """K1/K2/S1/S2 row validation never executes caller scalar hooks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.calls = 0

    with pytest.raises(ValueError, match="source must be an integer row index"):
        k_variants(_responses(), 0, _HostileInt(1))

    assert _HostileInt.calls == 0


def test_genuine_numpy_integer_controls_reach_dispatch_boundary(monkeypatch):
    """Exact NumPy integer scalars preserve compatibility after trusted normalization."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    copier, source, probs = _options()

    with pytest.raises(RuntimeError, match="wollack_omega requires the compiled Rust core"):
        wollack_omega(copier, source, probs, np.int64(3))
    with pytest.raises(RuntimeError, match="k_index requires the compiled Rust core"):
        k_index(_responses(), np.int64(0), np.int64(1))
    with pytest.raises(RuntimeError, match="k_variants requires the compiled Rust core"):
        k_variants(_responses(), np.int64(0), np.int64(1))

    assert calls == 3
