"""Trust-boundary regressions for Rasch CML public controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.rasch_cml as rasch_cml
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


class _HostileArray:
    """Array-protocol provider that native result validation must never execute."""

    calls = 0

    def __array__(self, *args, **kwargs):
        del args, kwargs
        type(self).calls += 1
        raise AssertionError("caller-owned __array__ executed")


def _unexpected_core_discovery():
    """Fail if rejected public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid public input")


def _lossy_longdouble() -> np.longdouble:
    """Return a finite positive long double that cannot round-trip through binary64."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double is not wider than float64")
    value = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert np.longdouble(float(value)) != value
    return value


def _distinct_longdouble_group_labels() -> tuple[np.longdouble, np.longdouble]:
    """Return adjacent integral labels that collapse through binary64."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double is not wider than float64")
    lower = np.longdouble(2**53)
    upper = lower + np.longdouble(1)
    assert upper != lower
    assert float(upper) == float(lower)
    return lower, upper


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


def test_rasch_cml_rejects_lossy_extended_precision_tolerance_before_core(monkeypatch):
    """A wider finite tolerance cannot silently change at the Rust f64 boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    tol = _lossy_longdouble()

    with pytest.raises(ValueError, match="tol"):
        fit_rasch_cml(_binary(), tol=tol)
    with pytest.raises(ValueError, match="tol"):
        andersen_lr_test(_binary(), np.array([0, 0, 1, 1]), tol=tol)


def test_andersen_preserves_distinct_longdouble_group_identity(monkeypatch):
    """Extended-precision integral labels remain distinct at the Rust boundary."""

    lower, upper = _distinct_longdouble_group_labels()
    captured: dict[str, object] = {}

    class _Core:
        def andersen_lr_test(
            self,
            yy,
            gid,
            n_groups,
            n_persons,
            n_items,
            max_iter,
            tol,
        ):
            captured["gid"] = np.array(gid, copy=True)
            captured["n_groups"] = n_groups
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": [2, 2],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = andersen_lr_test(_binary(), [lower, upper, lower, upper])

    assert result["converged"] is True
    assert captured["n_groups"] == 2
    np.testing.assert_array_equal(captured["gid"], np.array([0, 1, 0, 1], dtype=np.int64))


def test_andersen_numpy_longdouble_group_array_preserves_identity(monkeypatch):
    """NumPy long-double arrays keep distinct integral group identities."""

    lower, upper = _distinct_longdouble_group_labels()
    captured: dict[str, object] = {}

    class _Core:
        def andersen_lr_test(
            self,
            yy,
            gid,
            n_groups,
            n_persons,
            n_items,
            max_iter,
            tol,
        ):
            captured["gid"] = np.array(gid, copy=True)
            captured["n_groups"] = n_groups
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": [2, 2],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = andersen_lr_test(
        _binary(),
        np.array([lower, upper, lower, upper], dtype=np.longdouble),
    )

    assert result["converged"] is True
    assert captured["n_groups"] == 2
    np.testing.assert_array_equal(captured["gid"], np.array([0, 1, 0, 1], dtype=np.int64))


def test_andersen_numpy_group_array_bulk_converts_scalars(monkeypatch):
    """Ordinary NumPy groups avoid one NumPy scalar boxing operation per person."""

    captured: dict[str, object] = {}
    scalar_types: list[type[object]] = []
    original = rasch_cml._normalized_group_label

    def recording_normalizer(label: object) -> int:
        scalar_types.append(type(label))
        return original(label)

    class _Core:
        def andersen_lr_test(
            self,
            yy,
            gid,
            n_groups,
            n_persons,
            n_items,
            max_iter,
            tol,
        ):
            captured["gid"] = np.array(gid, copy=True)
            captured["n_groups"] = n_groups
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": [2, 2],
                "converged": True,
            }

    monkeypatch.setattr(rasch_cml, "_normalized_group_label", recording_normalizer)
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = andersen_lr_test(
        _binary(),
        np.array([0, 1, 0, 1], dtype=np.int64),
    )

    assert result["converged"] is True
    assert scalar_types == [int, int, int, int]
    assert captured["n_groups"] == 2
    np.testing.assert_array_equal(captured["gid"], np.array([0, 1, 0, 1], dtype=np.int64))


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


def test_exact_longdouble_tolerance_preserves_compatibility(monkeypatch):
    """An exactly representable long double remains supported at the Rust boundary."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)

    with pytest.raises(RuntimeError, match="fit_rasch_cml requires the compiled Rust core"):
        fit_rasch_cml(_binary(), tol=np.longdouble(0.5))

    assert calls == 1


def test_fit_rasch_cml_rejects_hostile_native_result_before_array_callback(monkeypatch):
    """A stale/foreign native result cannot run array protocols during marshalling."""

    _HostileArray.calls = 0

    class _Core:
        def fit_rasch_cml(self, *args):
            del args
            return {
                "beta": _HostileArray(),
                "se": [0.1, 0.1, 0.1],
                "loglik": -1.0,
                "n_iter": 1,
                "converged": True,
                "n_used": 4,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    with pytest.raises(RuntimeError, match="invalid Rasch CML Rust result payload"):
        fit_rasch_cml(_binary())

    assert _HostileArray.calls == 0


def test_fit_rasch_cml_rejects_malformed_native_result_shape_and_finiteness(monkeypatch):
    """Native vectors and scalar evidence are replayed before public marshalling."""

    payloads = [
        {
            "beta": [0.0, 0.0],
            "se": [0.1, 0.1, 0.1],
            "loglik": -1.0,
            "n_iter": 1,
            "converged": True,
            "n_used": 4,
        },
        {
            "beta": [0.0, 0.0, 0.0],
            "se": [0.1, 0.1, 0.1],
            "loglik": float("nan"),
            "n_iter": 1,
            "converged": True,
            "n_used": 4,
        },
    ]

    class _Core:
        def __init__(self, payload):
            self.payload = payload

        def fit_rasch_cml(self, *args):
            del args
            return self.payload

    for payload in payloads:
        monkeypatch.setattr(fitstats, "_core_module", lambda payload=payload: _Core(payload))
        with pytest.raises(RuntimeError, match="invalid Rasch CML Rust result payload"):
            fit_rasch_cml(_binary())


def test_andersen_rejects_hostile_native_result_before_array_callback(monkeypatch):
    """Andersen result vectors cannot invoke array protocols before replay."""

    _HostileArray.calls = 0

    class _Core:
        def andersen_lr_test(self, *args):
            del args
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": _HostileArray(),
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    with pytest.raises(RuntimeError, match="invalid Andersen Rust result payload"):
        andersen_lr_test(_binary(), [0, 0, 1, 1])

    assert _HostileArray.calls == 0


def test_native_result_replay_preserves_valid_current_shaped_payloads(monkeypatch):
    """Current Rust-shaped built-in payloads preserve public return compatibility."""

    class _Core:
        def fit_rasch_cml(self, *args):
            del args
            return {
                "beta": [-0.2, 0.0, 0.2],
                "se": [0.1, float("nan"), 0.1],
                "loglik": -2.5,
                "n_iter": 3,
                "converged": True,
                "n_used": 4,
            }

        def andersen_lr_test(self, *args):
            del args
            return {
                "lr": 1.25,
                "df": 2,
                "p_value": 0.5,
                "n_used": [2, 2],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    fit = fit_rasch_cml(_binary())
    np.testing.assert_array_equal(fit["beta"], np.array([-0.2, 0.0, 0.2]))
    assert np.isnan(fit["se"][1])
    assert fit["n_iter"] == 3
    assert fit["n_used"] == 4

    lr = andersen_lr_test(_binary(), [0, 0, 1, 1])
    assert lr["lr"] == 1.25
    assert lr["df"] == 2
    assert lr["p_value"] == 0.5
    np.testing.assert_array_equal(lr["n_used"], np.array([2, 2], dtype=np.int64))
    assert lr["converged"] is True
