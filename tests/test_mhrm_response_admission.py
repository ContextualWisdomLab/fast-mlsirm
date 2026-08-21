"""Data-integrity regressions for MH-RM public response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mhrm


def _unexpected_core() -> object:
    """Fail if native capability discovery happens after lossy complex narrowing."""

    raise AssertionError("compiled core discovered after lossy MH-RM response coercion")


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core 2PL MH-RM result."""

    return {
        "loading": [1.0, 1.0],
        "intercept": [0.0, 0.0],
        "step": [],
        "n_cat": 2,
        "theta": [0.0, 0.0],
        "corr": [1.0],
        "se_loading": [],
        "se_intercept": [],
        "se_step": [],
        "acceptance_rate": 0.25,
        "n_cycles": 2,
        "converged": False,
        "termination_reason": "max_cycles_reached",
        "final_param_change": 0.1,
        "n_parameters": 4,
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """A non-zero imaginary response must never be projected onto a real category."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array(
        [[0.0 + 0.0j, 1.0 + 1.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        mhrm.fit_mhrm(responses, 1, max_cycles=2, burn_in=1, mh_steps=1)


def test_real_responses_preserve_existing_native_marshalling(monkeypatch):
    """Ordinary real response matrices keep their existing integer Rust payload."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        """Capture trusted MH-RM arguments without running stochastic arithmetic."""

        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    fitted = mhrm.fit_mhrm(
        responses,
        1,
        max_cycles=2,
        burn_in=1,
        mh_steps=1,
        estimate_se=False,
    )

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(args[1], np.array([True, True, True, True]))
    assert args[3:6] == (2, 2, 1)
    assert fitted.loading.shape == (2, 1)
    assert fitted.theta.shape == (2, 1)
