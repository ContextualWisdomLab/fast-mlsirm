"""Data-integrity regressions for graded-response-model response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.grm import fit_grm


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core GRM result."""

    return {
        "slope": [1.0],
        "threshold": [0.0],
        "theta": [0.0, 0.0],
        "n_cat": 2,
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "termination_reason": "tolerance_met",
        "final_loglik_change": 0.0,
        "n_parameters": 2,
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """A non-zero imaginary category must never be projected onto the real axis."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered after lossy complex coercion")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.0 + 0.0j], [1.0 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_grm(responses, n_cat=2, max_iter=1)


def test_real_categories_preserve_existing_native_marshalling(monkeypatch):
    """Ordinary real category arrays retain the existing Rust dispatch representation."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_grm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = fit_grm(np.array([[0.0], [1.0]], dtype=np.float32), n_cat=2, max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[3:6] == (2, 1, 1)
    assert fitted.n_cat == 2
