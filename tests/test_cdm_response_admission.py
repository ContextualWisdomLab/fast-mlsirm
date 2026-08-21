"""Trust-boundary regressions for cognitive-diagnosis response admission."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import fit_cdm, fit_gdina


class _ArraySentinel:
    """Caller array provider that must remain untouched for invalid controls."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("caller array materialization executed")


class _HostileFloat:
    """Object-array element whose real conversion must never execute."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller numeric conversion executed")


def _unexpected_core_discovery():
    """Fail if invalid public evidence reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid CDM input")


def _q_matrix() -> np.ndarray:
    """Return a minimal valid item-by-attribute design."""

    return np.array([[1], [1]], dtype=np.int64)


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_complex_responses_before_lossy_coercion(monkeypatch, fit):
    """Imaginary observed evidence cannot be projected onto binary real data."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit(responses, _q_matrix())


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_object_storage_before_element_coercion(monkeypatch, fit):
    """Object response storage fails before caller numeric conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileFloat.reset()
    responses = np.array([[_HostileFloat(), 1], [1, 0]], dtype=object)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit(responses, _q_matrix())

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize(
    ("fit", "kwargs", "message"),
    [
        (fit_cdm, {"model": "other"}, "model must be 'dina' or 'dino'"),
        (fit_cdm, {"max_iter": 0}, "max_iter must be an integer between"),
        (fit_cdm, {"tol": 0.0}, "tol must be a finite number > 0"),
        (fit_gdina, {"max_iter": 0}, "max_iter must be an integer between"),
        (fit_gdina, {"tol": 0.0}, "tol must be a finite number > 0"),
    ],
)
def test_invalid_controls_fail_before_response_materialization(
    monkeypatch, fit: Callable[..., object], kwargs: dict[str, object], message: str
):
    """Semantic controls are rejected before caller response array protocols."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _ArraySentinel.reset()

    with pytest.raises(ValueError, match=message):
        fit(_ArraySentinel(), _q_matrix(), **kwargs)

    assert _ArraySentinel.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_valid_real_evidence_reaches_native_dispatch_boundary(monkeypatch, fit):
    """Ordinary real binary responses retain the existing Rust dispatch contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array([[0.0, 1.0], [1.0, np.nan]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fit(responses, _q_matrix())

    assert calls == 1
