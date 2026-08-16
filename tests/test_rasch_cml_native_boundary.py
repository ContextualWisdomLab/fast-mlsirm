"""Trust-boundary regressions for Rasch CML public controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.rasch_cml import andersen_lr_test, fit_rasch_cml


def _binary() -> np.ndarray:
    """Return a small valid complete binary response matrix."""

    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


class _HostileInt(int):
    """Integer subclass whose coercion must never run."""

    calls = 0

    def __int__(self):
        type(self).calls += 1
        raise AssertionError("caller-owned __int__ executed")


class _HostileFloat(float):
    """Float subclass whose coercion must never run."""

    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("caller-owned __float__ executed")


def _unexpected_core_discovery():
    """Fail if rejected public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid public input")


def test_fit_rasch_cml_rejects_bad_shape_before_core_discovery(monkeypatch):
    """Malformed responses remain a package validation failure."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        fit_rasch_cml(np.zeros(3))


def test_fit_rasch_cml_rejects_hostile_controls_without_callbacks(monkeypatch):
    """Scalar subclasses cannot execute coercion hooks before native dispatch."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.calls = 0
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match="max_iter"):
        fit_rasch_cml(_binary(), max_iter=_HostileInt(10))
    with pytest.raises(ValueError, match="tol"):
        fit_rasch_cml(_binary(), tol=_HostileFloat(1e-8))

    assert _HostileInt.calls == 0
    assert _HostileFloat.calls == 0


def test_andersen_rejects_bad_group_before_core_discovery(monkeypatch):
    """Malformed group labels fail before compiled-core discovery."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="length-n_persons"):
        andersen_lr_test(_binary(), np.array([0, 1]))


def test_numpy_controls_reach_core_discovery_after_validation(monkeypatch):
    """Genuine NumPy scalar controls preserve the public compatibility contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)

    with pytest.raises(RuntimeError, match="fit_rasch_cml requires the compiled Rust core"):
        fit_rasch_cml(_binary(), max_iter=np.int64(10), tol=np.float64(1e-8))

    assert calls == 1
