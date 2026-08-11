"""Contracts for the safe marginal-distance engineering benchmark."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    """Load the repository benchmark without requiring it to be a Python package."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_marginal_distance_workspaces.py"
    spec = importlib.util.spec_from_file_location("marginal_distance_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_stable_kernel_and_exact_pairwise_peak() -> None:
    """Benchmark metadata must describe the final coordinate-subtraction kernel."""
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        n_left=4,
        n_right=5,
        latent_dim=2,
        warmups=1,
        repetitions=1,
        seed=563,
        output=None,
    )

    report = benchmark.build_report(args)

    assert "bounded_coordinate_subtraction" in report
    assert "bounded_squared_norm" not in report
    assert (
        "eliminates_the_identified_high_offset_cancellation_mechanism"
        in report["claims"]
    )
    assert report["inputs"]["pairwise_kernel_peak_bytes"] == 2 * 4 * 5 * np.dtype(np.float64).itemsize
    assert report["inputs"]["pairwise_output_bytes"] == 4 * 5 * np.dtype(np.float64).itemsize
    assert report["inputs"]["private_ceiling_bytes"] == benchmark.MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES


def test_benchmark_reference_comparison_remains_below_safety_ceiling() -> None:
    """Small benchmark runs compare stable and explicit-broadcast distances exactly enough."""
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        n_left=3,
        n_right=7,
        latent_dim=3,
        warmups=1,
        repetitions=1,
        seed=563,
        output=None,
    )

    report = benchmark.build_report(args)

    assert report["legacy_broadcast"] is not None
    assert report["legacy_broadcast"]["maximum_absolute_difference"] <= 1e-12
