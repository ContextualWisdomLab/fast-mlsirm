"""Trust-boundary regressions for many-facet response evidence admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.facets import fit_facets


@dataclass
class _CallbackCounter:
    """Record caller-controlled evidence conversion attempts."""

    calls: int = 0

    def hit(self) -> None:
        """Record and fail if package validation executes caller conversion."""

        self.calls += 1
        raise AssertionError("caller evidence conversion executed")


class _HostileFloat:
    """Object-array element whose real conversion must never execute."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __float__(self) -> float:
        self._counter.hit()
        return 0.0


class _HostileArrayProvider:
    """Top-level provider whose NumPy array callback must never execute."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __array__(self, dtype=None, copy=None):
        self._counter.hit()
        return np.zeros((1, 1, 1), dtype=np.float64)


def _unexpected_core_discovery():
    """Fail if invalid evidence reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


def test_fit_facets_rejects_complex_responses_before_lossy_coercion(monkeypatch):
    """Imaginary rating evidence is not projected onto real response categories."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array(
        [[[0.0 + 1.0j], [1.0]], [[1.0], [0.0]]], dtype=np.complex128
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_rejects_object_storage_before_element_coercion(monkeypatch):
    """Object ratings fail before caller-provided per-element conversions."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    counter = _CallbackCounter()
    responses = np.array(
        [[[_HostileFloat(counter)], [1]], [[1], [0]]], dtype=object
    )

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)

    assert counter.calls == 0


def test_fit_facets_rejects_array_provider_before_numpy_callback(monkeypatch):
    """Arbitrary array providers cannot synthesize the observed rating cube."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    counter = _CallbackCounter()

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit_facets(
            _HostileArrayProvider(counter),
            n_cat=2,
            q_theta=41,
            max_iter=10,
            tol=1e-6,
        )

    assert counter.calls == 0


def test_fit_facets_valid_real_evidence_reaches_dispatch_boundary(monkeypatch):
    """Ordinary real ratings retain their historical native-dispatch contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array(
        [[[0.0], [1.0]], [[1.0], [np.nan]]], dtype=np.float64
    )

    with pytest.raises(RuntimeError, match="fit_facets requires the compiled Rust core"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)

    assert calls == 1


def test_fit_facets_trusted_builtin_sequence_evidence_reaches_dispatch(monkeypatch):
    """Exact list/tuple trees with trusted real scalars remain compatible."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = (
        ([np.int8(0)], [np.float32(1.0)]),
        ([np.uint8(1)], [np.nan]),
    )

    with pytest.raises(RuntimeError, match="fit_facets requires the compiled Rust core"):
        fit_facets(responses, n_cat=np.int8(2), q_theta=np.int16(41), max_iter=10, tol=1e-6)

    assert calls == 1
