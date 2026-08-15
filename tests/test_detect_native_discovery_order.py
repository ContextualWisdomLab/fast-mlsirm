"""Regression tests for validation-before-native-discovery in DETECT adapters."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.detect import detect_analysis, dimtest


def _unexpected_core_discovery():
    """Fail if invalid public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid public input")


def test_detect_rejects_invalid_response_shape_before_core_discovery(monkeypatch):
    """Malformed DETECT responses fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        detect_analysis(np.zeros(6), np.array([0, 1]))


def test_detect_rejects_nonbinary_responses_before_core_discovery(monkeypatch):
    """Out-of-domain DETECT responses fail before native discovery."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        detect_analysis(responses, np.array([0, 1]))


def test_dimtest_rejects_invalid_partition_before_core_discovery(monkeypatch):
    """Malformed DIMTEST item indices fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.zeros((40, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="at1 indices must be non-empty integers"):
        dimtest(responses, np.array([], dtype=np.int64), np.array([2, 3]))


def test_detect_valid_input_discovers_core_only_at_dispatch_boundary(monkeypatch):
    """A valid request still discovers the compiled core exactly when dispatch is needed."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="detect_analysis requires the compiled Rust core"):
        detect_analysis(responses, np.array([0, 1], dtype=np.int64))

    assert calls == 1


def test_dimtest_valid_input_discovers_core_only_at_dispatch_boundary(monkeypatch):
    """A valid DIMTEST request still reaches core discovery after validation."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.zeros((40, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="dimtest requires the compiled Rust core"):
        dimtest(responses, np.array([0, 1]), np.array([2, 3]))

    assert calls == 1
