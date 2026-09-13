"""Binding-integrity regressions for KSIRT Rust result marshalling."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import ksirt


class _HostileArrayPayload:
    """Array-like payload whose conversion callback must never execute."""

    calls = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).calls += 1
        raise AssertionError("hostile Rust-result __array__ callback executed")


def _valid_result() -> dict[str, object]:
    """Return the package-shaped result produced by the current Rust binding."""
    return {
        "theta": [-0.5, 0.5],
        "grid": [-1.0, 0.0, 1.0],
        "bandwidth": [0.5],
        "options": [[0.0, 1.0]],
        "occ": [[0.8, 0.5, 0.2, 0.2, 0.5, 0.8]],
        "expected": [[0.2, 0.5, 0.8]],
        "expected_total": [0.2, 0.5, 0.8],
    }


def _core_with_result(result: dict[str, object]) -> object:
    """Return a package-owned fake core for result-boundary tests."""

    class _Core:
        def ksirt_occ(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            kernel: str,
            nevalpoints: int,
            bandwidth: list[float] | None,
        ) -> dict[str, object]:
            del responses, n_persons, n_items, kernel, nevalpoints, bandwidth
            return result

    return _Core()


def _call_with_result(monkeypatch: pytest.MonkeyPatch, result: dict[str, object]):
    """Run one minimal valid KSIRT request against a controlled native result."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_with_result(result))
    return ksirt.ksirt_analysis(
        np.array([[0.0], [1.0]], dtype=np.float64),
        nevalpoints=3,
        bandwidth=np.array([0.5], dtype=np.float64),
    )


def test_ksirt_rejects_hostile_native_payload_before_numpy_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale/foreign native result cannot execute caller array protocols."""
    result = _valid_result()
    _HostileArrayPayload.calls = 0
    result["grid"] = _HostileArrayPayload()

    with pytest.raises(RuntimeError, match="invalid KSIRT Rust result payload"):
        _call_with_result(monkeypatch, result)

    assert _HostileArrayPayload.calls == 0


def test_ksirt_rejects_native_vector_length_mismatch_before_reshape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native result lengths must match the exact request before marshalling."""
    result = _valid_result()
    result["occ"] = [[0.8, 0.5, 0.2]]

    with pytest.raises(RuntimeError, match="invalid KSIRT Rust result payload"):
        _call_with_result(monkeypatch, result)


def test_ksirt_rejects_nonfinite_native_result_before_public_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite Rust-returned diagnostics fail closed at the binding boundary."""
    result = _valid_result()
    result["expected_total"] = [0.2, float("nan"), 0.8]

    with pytest.raises(RuntimeError, match="invalid KSIRT Rust result payload"):
        _call_with_result(monkeypatch, result)


def test_ksirt_accepts_current_rust_shaped_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current Rust result shape remains compatible with public marshalling."""
    result = _call_with_result(monkeypatch, _valid_result())

    assert result.theta.shape == (2,)
    assert result.grid.shape == (3,)
    assert result.bandwidth.tolist() == [0.5]
    assert len(result.options) == 1
    assert result.options[0].tolist() == [0.0, 1.0]
    assert result.occ[0].shape == (2, 3)
    assert result.expected[0].tolist() == [0.2, 0.5, 0.8]
    assert result.expected_total.tolist() == [0.2, 0.5, 0.8]
