"""Trust-boundary regressions for public LLTM input admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import lltm


class _ArrayProbe:
    """Fail if caller-owned data are materialized before semantic controls."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("caller-owned data materialized before LLTM control admission")


class _TruthProvider:
    """Fail if an untrusted Boolean protocol is invoked."""

    def __bool__(self):
        raise AssertionError("caller Boolean callback executed")


class _IntegerProvider:
    """Fail if an untrusted integer protocol is invoked."""

    def __int__(self):
        raise AssertionError("caller integer callback executed")


class _RealProvider:
    """Fail if an untrusted real-number protocol is invoked."""

    def __float__(self):
        raise AssertionError("caller real-number callback executed")


def _unexpected_core() -> object:
    """Fail if native capability discovery happens before Python admission."""

    raise AssertionError("compiled core discovered before LLTM input admission")


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core LLTM result."""

    return {
        "eta": [0.0],
        "intercept": 0.0,
        "b": [0.0, 0.0],
        "theta": [0.0, 0.0],
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": 2,
        "loglik_rasch": 0.0,
        "lr_stat": 0.0,
        "lr_df": 0,
        "lr_p": float("nan"),
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """Imaginary observed-response evidence must never be projected onto the real axis."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0.0 + 0.0j, 1.0 + 1.0j]], dtype=np.complex128)
    q_design = np.array([[0.0], [1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        lltm.fit_lltm(responses, q_design)


def test_complex_design_fails_before_lossy_cast_or_native_discovery(monkeypatch):
    """Imaginary explanatory weights must never be projected onto a different design."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0.0, 1.0]], dtype=np.float64)
    q_design = np.array([[0.0 + 0.0j], [1.0 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="q_design must be real-valued"):
        lltm.fit_lltm(responses, q_design)


def test_untrusted_controls_fail_before_callbacks_data_or_native_discovery(monkeypatch):
    """Control admission must reject protocol providers before any caller or native work."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        ({"fit_intercept": _TruthProvider()}, "fit_intercept must be a boolean"),
        ({"compute_lr": _TruthProvider()}, "compute_lr must be a boolean"),
        ({"max_iter": _IntegerProvider()}, "max_iter must be a positive integer"),
        ({"tol": _RealProvider()}, "tol must be finite and non-negative"),
    )
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            lltm.fit_lltm(_ArrayProbe(), _ArrayProbe(), **kwargs)


def test_invalid_control_domains_fail_before_data_or_native_discovery(monkeypatch):
    """Rust-owned LLTM control domains should fail at the Python marshalling boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"tol": -1.0}, "tol must be finite and non-negative"),
        ({"tol": float("nan")}, "tol must be finite and non-negative"),
        ({"tol": float("inf")}, "tol must be finite and non-negative"),
    )
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            lltm.fit_lltm(_ArrayProbe(), _ArrayProbe(), **kwargs)


def test_lossy_tolerance_fails_before_data_or_native_discovery(monkeypatch):
    """The exact tolerance identity must survive the Rust binary64 boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    with pytest.raises(ValueError, match="tol must be finite and non-negative"):
        lltm.fit_lltm(_ArrayProbe(), _ArrayProbe(), tol=2**53 + 1)


def test_trusted_numpy_controls_preserve_native_marshalling(monkeypatch):
    """Concrete NumPy scalar controls normalize to inert built-in Rust arguments."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        """Capture the LLTM PyO3 argument vector without performing arithmetic."""

        def fit_lltm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    q_design = np.array([[0.0], [1.0]], dtype=np.float32)

    fitted = lltm.fit_lltm(
        responses,
        q_design,
        fit_intercept=np.bool_(True),
        compute_lr=np.bool_(False),
        max_iter=np.int64(1),
        tol=np.float32(0.0),
    )

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.0, 1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(args[1], np.array([True, True, True, True]))
    np.testing.assert_array_equal(args[2], np.array([0.0, 1.0]))
    assert args[3:6] == (2, 2, 1)
    assert type(args[6]) is bool and args[6] is True
    assert type(args[7]) is bool and args[7] is False
    assert type(args[8]) is int and args[8] == 1
    assert type(args[9]) is float and args[9] == 0.0
    assert fitted.n_iter == 1


def test_exact_large_integer_tolerance_preserves_native_marshalling(monkeypatch):
    """An exactly representable integer tolerance remains a package-owned Rust float."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_lltm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    lltm.fit_lltm(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[0.0], [1.0]], dtype=np.float64),
        compute_lr=False,
        max_iter=1,
        tol=2**53,
    )

    assert type(captured["args"][9]) is float
    assert captured["args"][9] == float(2**53)


def test_untrusted_scientific_carriers_fail_before_numpy_or_native_discovery(monkeypatch):
    """Observed responses and explanatory design must reject caller array protocols."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = [[0.0, 1.0], [1.0, 0.0]]
    q_design = [[0.0], [1.0]]

    with pytest.raises(ValueError, match="responses must be an exact NumPy array or built-in matrix"):
        lltm.fit_lltm(_ArrayProbe(), q_design)
    with pytest.raises(ValueError, match="q_design must be an exact NumPy array or built-in matrix"):
        lltm.fit_lltm(responses, _ArrayProbe())


def test_nested_real_provider_fails_without_conversion_callback(monkeypatch):
    """Nested scientific evidence must be inert before float64 marshalling."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    with pytest.raises(ValueError, match="responses must contain real-valued numeric evidence"):
        lltm.fit_lltm([[_RealProvider(), 1.0], [1.0, 0.0]], [[0.0], [1.0]])
    with pytest.raises(ValueError, match="q_design must contain real-valued numeric evidence"):
        lltm.fit_lltm([[0.0, 1.0], [1.0, 0.0]], [[_RealProvider()], [1.0]])


def test_trusted_builtin_and_exact_numpy_rows_preserve_marshalling(monkeypatch):
    """Callback-free built-in matrices and exact NumPy rows remain supported."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_lltm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = [[np.float32(0.0), np.int16(1)], np.array([1.0, 0.0], dtype=np.float32)]
    q_design = [np.array([0.0], dtype=np.float32), [np.uint8(1)]]

    lltm.fit_lltm(responses, q_design, max_iter=1, tol=0.0)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.0, 1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(args[2], np.array([0.0, 1.0]))
    assert args[3:6] == (2, 2, 1)


def test_response_infinity_is_not_reclassified_as_missing(monkeypatch):
    """Only NaN is LLTM missingness; infinities are invalid observed evidence."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    q_design = np.array([[0.0], [1.0]], dtype=np.float64)
    for invalid in (float("inf"), float("-inf")):
        responses = np.array([[invalid, 1.0], [0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="responses must contain only finite values or NaN missingness"):
            lltm.fit_lltm(responses, q_design)


def test_nonfinite_design_fails_before_native_discovery(monkeypatch):
    """The public boundary should replay Rust's finite explanatory-design contract."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    for invalid in (float("nan"), float("inf"), float("-inf")):
        q_design = np.array([[0.0], [invalid]], dtype=np.float64)
        with pytest.raises(ValueError, match="q_design entries must be finite"):
            lltm.fit_lltm(responses, q_design)


def test_nan_response_remains_explicit_missingness(monkeypatch):
    """NaN remains the sole supported public missing-response sentinel."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_lltm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[np.nan, 1.0], [0.0, 1.0]], dtype=np.float64)
    q_design = np.array([[0.0], [1.0]], dtype=np.float64)

    lltm.fit_lltm(responses, q_design, max_iter=1, tol=0.0)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(args[1], np.array([False, True, True, True]))
