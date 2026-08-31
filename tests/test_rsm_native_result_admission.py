"""Fail-closed admission tests for Rating Scale Model native results."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import fit_rsm


def _native_result() -> dict[str, object]:
    return {
        "item_location": np.array([0.1, -0.1], dtype=np.float64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "theta": np.array([-0.2, 0.2], dtype=np.float64),
        "loglik_trace": np.array([-10.0, -9.0], dtype=np.float64),
        "n_iter": 2,
        "converged": True,
        "n_parameters": 2,
    }


def _fit_with_result(monkeypatch, result: object):
    class Core:
        @staticmethod
        def fit_rsm(*args, **kwargs):
            return result

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    responses = np.array([[0, 1], [1, 0]], dtype=np.float64)
    return fit_rsm(responses, n_cat=2)


def test_fit_rsm_seals_native_vector_storage(monkeypatch):
    """Returned evidence must not alias provider-owned native arrays."""

    result = _native_result()
    fitted = _fit_with_result(monkeypatch, result)

    expected_item_location = fitted.item_location.copy()
    expected_thresholds = fitted.thresholds.copy()
    expected_theta = fitted.theta.copy()
    expected_loglik = fitted.loglik_trace.copy()

    result["item_location"][:] = 99.0
    result["thresholds"][:] = 99.0
    result["theta"][:] = 99.0
    result["loglik_trace"][:] = 99.0

    np.testing.assert_array_equal(fitted.item_location, expected_item_location)
    np.testing.assert_array_equal(fitted.thresholds, expected_thresholds)
    np.testing.assert_array_equal(fitted.theta, expected_theta)
    np.testing.assert_array_equal(fitted.loglik_trace, expected_loglik)


def test_fit_rsm_rejects_foreign_native_root_shape(monkeypatch):
    """Unexpected native fields cannot be silently ignored."""

    result = _native_result()
    result["unexpected"] = 1

    with pytest.raises(RuntimeError, match="invalid RSM Rust result payload"):
        _fit_with_result(monkeypatch, result)


def test_fit_rsm_rejects_wrong_native_vector_cardinality(monkeypatch):
    """Native vectors must match the deterministic public result dimensions."""

    result = _native_result()
    result["item_location"] = np.array([0.1], dtype=np.float64)

    with pytest.raises(RuntimeError, match="invalid RSM Rust result payload"):
        _fit_with_result(monkeypatch, result)


def test_fit_rsm_rejects_callback_bearing_native_scalar(monkeypatch):
    """Native scalar coercion must not execute provider conversion callbacks."""

    called = False

    class CallbackInt(int):
        def __int__(self):
            nonlocal called
            called = True
            return 2

    result = _native_result()
    result["n_iter"] = CallbackInt(2)

    with pytest.raises(RuntimeError, match="invalid RSM Rust result payload"):
        _fit_with_result(monkeypatch, result)
    assert called is False
