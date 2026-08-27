"""Ordering regressions for optional KSIRT bandwidth admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import ksirt


def test_invalid_bandwidth_fails_before_response_value_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent bandwidth controls fail after shape but before response values."""
    original_real_float_array = ksirt._real_float_array
    admitted_names: list[str] = []

    def _guarded_real_float_array(value: object, name: str) -> np.ndarray:
        admitted_names.append(name)
        if name == "responses":
            raise AssertionError("response values were marshalled before bandwidth")
        return original_real_float_array(value, name)

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(ksirt, "_real_float_array", _guarded_real_float_array)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    responses = np.broadcast_to(np.array([[0.0]], dtype=np.float64), (2, 1))
    with pytest.raises(ValueError, match="bandwidths must be finite and positive"):
        ksirt.ksirt_analysis(responses, bandwidth=np.array([0.0], dtype=np.float64))

    assert admitted_names == ["bandwidth"]
