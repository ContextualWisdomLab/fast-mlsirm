"""Fail-closed admission tests for Rating Scale Model native results."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import fit_rsm


def _native_result(*, converged: bool = True) -> dict[str, object]:
    trace = [-10.0, -9.0] if converged else [-10.0, -9.0, -8.5]
    return {
        "item_location": [0.1, -0.1],
        "thresholds": [0.0],
        "theta": [-0.2, 0.2],
        "loglik_trace": trace,
        "n_iter": 2,
        "converged": converged,
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
    """Returned evidence must not alias provider-owned PyO3 list vectors."""

    result = _native_result()
    fitted = _fit_with_result(monkeypatch, result)

    expected_item_location = fitted.item_location.copy()
    expected_thresholds = fitted.thresholds.copy()
    expected_theta = fitted.theta.copy()
    expected_loglik = fitted.loglik_trace.copy()

    result["item_location"][0] = 99.0
    result["thresholds"][0] = 99.0
    result["theta"][0] = 99.0
    result["loglik_trace"][0] = 99.0

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
    result["item_location"] = [0.1]

    with pytest.raises(RuntimeError, match="invalid RSM Rust result payload"):
        _fit_with_result(monkeypatch, result)


def test_fit_rsm_rejects_callback_bearing_native_scalar(monkeypatch):
    """Native scalar admission must not execute provider conversion callbacks."""

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


def test_fit_rsm_accepts_native_max_iteration_trace_contract(monkeypatch):
    """An unconverged native fit carries one final returned-parameter likelihood."""

    fitted = _fit_with_result(monkeypatch, _native_result(converged=False))

    assert fitted.converged is False
    assert fitted.n_iter == 2
    assert fitted.loglik_trace.shape == (3,)


def test_fit_rsm_compiled_core_converged_trace_contract():
    """The real PyO3 producer is accepted on a deterministic converged path."""

    responses = np.array([[0, 1], [1, 0]], dtype=np.float64)
    fitted = fit_rsm(responses, n_cat=2, max_iter=2, tol=1e9)

    assert fitted.converged is True
    assert fitted.n_iter == 2
    assert fitted.loglik_trace.shape == (2,)


def test_fit_rsm_compiled_core_max_iteration_trace_contract():
    """The real PyO3 producer is accepted when the iteration cap is reached."""

    responses = np.array([[0, 1], [1, 0]], dtype=np.float64)
    fitted = fit_rsm(responses, n_cat=2, max_iter=1, tol=1e-14)

    assert fitted.converged is False
    assert fitted.n_iter == 1
    assert fitted.loglik_trace.shape == (2,)
