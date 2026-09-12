"""Default-workspace ordering regressions for Horn parallel analysis."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm.parallel_analysis import parallel_analysis


class _RecordingCore:
    """Capture a valid default-workspace call without running Rust numerics."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def parallel_analysis(self, *args):
        self.calls.append(args)
        return {
            "retained": 1,
            "eigenvalues": [1.5, 0.5],
            "random_eigenvalues": [1.1, 0.9],
            "bias": [0.1, -0.1],
            "adjusted_eigenvalues": [1.4, 0.6],
        }


def _reject_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make compiled-core discovery observable and forbidden."""
    import fast_mlsirm.fitstats as fitstats

    calls: list[bool] = []

    def discover_core() -> object:
        calls.append(True)
        raise AssertionError("native core discovered before workspace validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    return calls


def test_default_workspace_fails_before_dense_observed_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known default-workspace overflow must precede observed dense marshalling."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    monkeypatch.setattr(
        module,
        "_MAX_PARALLEL_RANDOM_WORKSPACE_BYTES",
        959,
        raising=False,
    )
    discovery_calls = _reject_core_discovery(monkeypatch)
    dense_calls: list[bool] = []

    def reject_dense_conversion(raw: np.ndarray) -> np.ndarray:
        dense_calls.append(True)
        raise AssertionError("dense observed conversion executed before workspace validation")

    monkeypatch.setattr(module, "_lossless_float64_matrix", reject_dense_conversion)
    data = np.broadcast_to(np.array([[0.25]], dtype=np.float64), (3, 2))

    with pytest.raises(ValueError, match="workspace"):
        parallel_analysis(data)

    assert dense_calls == []
    assert discovery_calls == []


def test_default_workspace_at_boundary_reaches_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid default workspace retains the established public contract."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    monkeypatch.setattr(
        module,
        "_MAX_PARALLEL_RANDOM_WORKSPACE_BYTES",
        960,
        raising=False,
    )
    core = _RecordingCore()
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    data = np.array(
        [[0.1, 1.0], [0.4, 0.5], [0.9, -0.2]],
        dtype=np.float64,
    )

    result = parallel_analysis(data)

    assert result.retained == 1
    assert len(core.calls) == 1
    assert core.calls[0][-3:] == (60, 0, 1)
