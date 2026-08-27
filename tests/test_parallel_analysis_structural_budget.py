"""Fail-first resource and shape contracts for parallel-analysis matrix traversal."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


class _RecordingCore:
    """Capture accepted evidence without running the numerical kernel."""

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


def test_empty_row_fanout_is_bounded_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell row fan-out cannot bypass a bounded Python traversal budget."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    monkeypatch.setattr(
        module,
        "_MAX_PARALLEL_DATA_STRUCTURE_NODES",
        2,
        raising=False,
    )
    dense_calls: list[bool] = []
    discovery_calls: list[bool] = []

    def reject_dense_conversion(raw: np.ndarray) -> np.ndarray:
        dense_calls.append(True)
        raise AssertionError("dense conversion executed before structural validation")

    def reject_core_discovery() -> object:
        discovery_calls.append(True)
        raise AssertionError("native core discovered before structural validation")

    monkeypatch.setattr(module, "_lossless_float64_matrix", reject_dense_conversion)
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", reject_core_discovery)

    with pytest.raises(ValueError, match="structural"):
        module.parallel_analysis([[], [], []], n_iterations=1)

    assert dense_calls == []
    assert discovery_calls == []


def test_valid_matrix_at_structural_budget_reaches_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural ceiling preserves a valid non-empty 2-D matrix."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    monkeypatch.setattr(
        module,
        "_MAX_PARALLEL_DATA_STRUCTURE_NODES",
        6,
        raising=False,
    )
    core = _RecordingCore()
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    data = [[0.1, 1.0], [0.4, 0.5]]

    result = module.parallel_analysis(data, n_iterations=1)

    assert result.retained == 1
    assert len(core.calls) == 1
    assert core.calls[0][0] == pytest.approx([0.1, 1.0, 0.4, 0.5])


def test_ragged_builtin_matrix_fails_before_numpy_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known ragged row widths fail during bounded preflight, not NumPy inference."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    numpy_calls: list[bool] = []
    discovery_calls: list[bool] = []

    def reject_numpy_materialization(*args, **kwargs):
        numpy_calls.append(True)
        raise AssertionError("NumPy materialization executed before rectangularity replay")

    def reject_core_discovery() -> object:
        discovery_calls.append(True)
        raise AssertionError("native core discovered before rectangularity replay")

    monkeypatch.setattr(module.np, "asarray", reject_numpy_materialization)
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", reject_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        module.parallel_analysis([[0.1, 1.0], [0.4]], n_iterations=1)

    assert numpy_calls == []
    assert discovery_calls == []


def test_flat_builtin_sequence_fails_before_numpy_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known 1-D built-in carrier fails before dense shape inference."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    numpy_calls: list[bool] = []
    discovery_calls: list[bool] = []

    def reject_numpy_materialization(*args, **kwargs):
        numpy_calls.append(True)
        raise AssertionError("NumPy materialization executed before 2-D preflight")

    def reject_core_discovery() -> object:
        discovery_calls.append(True)
        raise AssertionError("native core discovered before 2-D preflight")

    monkeypatch.setattr(module.np, "asarray", reject_numpy_materialization)
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", reject_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        module.parallel_analysis([0.1, 0.4], n_iterations=1)

    assert numpy_calls == []
    assert discovery_calls == []


def test_equal_width_builtin_and_numpy_rows_reach_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rectangularity replay preserves mixed trusted row carriers."""
    module = importlib.import_module("fast_mlsirm.parallel_analysis")
    core = _RecordingCore()
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    data = [[0.1, 1.0], np.array([0.4, 0.5], dtype=np.float32)]

    result = module.parallel_analysis(data, n_iterations=1)

    assert result.retained == 1
    assert len(core.calls) == 1
    assert core.calls[0][0] == pytest.approx([0.1, 1.0, 0.4, 0.5])
