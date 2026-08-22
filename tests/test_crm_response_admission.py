"""Data-integrity regressions for continuous-response-model response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import crm


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core CRM result."""

    return {
        "slope": [1.0],
        "intercept": [0.0],
        "resid_sd": [1.0],
        "discrimination": [1.0],
        "difficulty": [0.0],
        "theta": [0.0, 0.0],
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": 3,
        "termination_reason": "tolerance",
        "final_delta": 0.0,
        "stopping_tolerance": 1e-6,
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """A non-zero imaginary response must never be projected onto the real axis."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered after lossy complex coercion")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.25 + 0.0j], [0.75 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        crm.fit_crm(responses)


def test_object_complex_responses_use_package_error_before_native_discovery(monkeypatch):
    """Object storage containing complex evidence must fail at package admission."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before object-complex rejection")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.25], [0.75 + 1.0j]], dtype=object)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        crm.fit_crm(responses)


def test_real_responses_preserve_existing_native_marshalling(monkeypatch):
    """Ordinary real-valued response arrays keep the existing Rust dispatch shape."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = crm.fit_crm(np.array([[0.25], [0.75]], dtype=np.float32), max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.25, 0.75], dtype=np.float64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[2:4] == (2, 1)
    assert fitted.n_parameters == 3
