"""Contracts for marginal-reduction parity and allocation evidence."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_benchmark_module():
    """Load the executable benchmark from the repository."""
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "benchmark_marginal_reductions.py"
    )
    assert path.exists(), "marginal-reduction benchmark is missing"
    spec = importlib.util.spec_from_file_location(
        "marginal_reduction_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_parity_and_observed_peak_allocations() -> None:
    """The report must bind every changed reduction to values and memory evidence."""
    benchmark = _load_benchmark_module()
    report = benchmark.build_report(
        argparse.Namespace(
            repetitions=1,
            seed=2310,
            clusters=256,
            contexts=64,
            groups=4,
            items=12,
            theta_nodes=12,
            latent_nodes=16,
        )
    )

    assert report["parity"] == {
        "multilevel_second_moment": True,
        "covariate_score_information": True,
        "weighted_population_moments": True,
    }
    for reduction in report["measurements"].values():
        assert reduction["optimized"]["traced_peak_bytes"][0] < reduction["former"][
            "traced_peak_bytes"
        ][0]
