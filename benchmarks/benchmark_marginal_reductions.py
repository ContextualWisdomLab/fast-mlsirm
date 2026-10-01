#!/usr/bin/env python3
"""Measure parity and traced allocations for three marginal reductions.

The report compares the current production helpers with their former broadcast
equations on exactly representable float64 fixtures. Results are specific to
the recorded environment and do not imply a universal speedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from typing import Callable

import numpy as np

from fast_mlsirm.estimators.marginal import (
    _covariate_score_information,
    _multilevel_second_moment,
    _weighted_population_moments,
)


def _positive_integer(value: str) -> int:
    """Return a positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _measure(operation: Callable[[], object], repetitions: int) -> dict[str, object]:
    """Return elapsed and Python-traced peak observations for one operation."""
    operation()
    elapsed_seconds: list[float] = []
    traced_peak_bytes: list[int] = []
    for _ in range(repetitions):
        tracemalloc.start()
        started = time.perf_counter()
        operation()
        elapsed_seconds.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_peak_bytes.append(int(peak))
    return {
        "elapsed_seconds": elapsed_seconds,
        "traced_peak_bytes": traced_peak_bytes,
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    """Build one deterministic parity and allocation report."""
    random_generator = np.random.default_rng(args.seed)

    cluster_post = random_generator.integers(
        0,
        5,
        size=(args.clusters, args.contexts),
    ).astype(np.float64)
    u_nodes = random_generator.integers(
        -3,
        4,
        size=args.contexts,
    ).astype(np.float64)
    current_multilevel = lambda: _multilevel_second_moment(cluster_post, u_nodes)
    former_multilevel = lambda: float(
        (cluster_post * u_nodes[None, :] ** 2).sum()
    )

    reduction_shape = (
        args.groups,
        args.items,
        args.theta_nodes,
        args.latent_nodes,
    )
    residual = random_generator.integers(0, 5, size=reduction_shape).astype(
        np.float64
    )
    expected_count = random_generator.integers(1, 5, size=reduction_shape).astype(
        np.float64
    )
    probability = np.full(reduction_shape, 0.5, dtype=np.float64)
    covariate = random_generator.integers(
        -3,
        4,
        size=(args.groups, args.items),
    ).astype(np.float64)
    current_covariate = lambda: _covariate_score_information(
        residual,
        expected_count,
        probability,
        covariate,
    )

    def former_covariate() -> tuple[float, float]:
        """Return the former broadcast score and information equations."""
        covariate_broadcast = covariate[:, :, None, None]
        return (
            float((residual * covariate_broadcast).sum()),
            float(
                (
                    expected_count
                    * probability
                    * (1.0 - probability)
                    * covariate_broadcast
                    * covariate_broadcast
                ).sum()
            ),
        )

    population_weights = random_generator.integers(
        0,
        5,
        size=(args.theta_nodes, args.latent_nodes),
    ).astype(np.float64)
    theta_nodes = random_generator.integers(
        -3,
        4,
        size=args.theta_nodes,
    ).astype(np.float64)
    current_population = lambda: _weighted_population_moments(
        population_weights,
        theta_nodes,
    )

    def former_population() -> tuple[float, float, float]:
        """Return the former broadcast population-moment equations."""
        return (
            float(population_weights.sum()),
            float((population_weights * theta_nodes[:, None]).sum()),
            float((population_weights * (theta_nodes**2)[:, None]).sum()),
        )

    current_values = {
        "multilevel_second_moment": current_multilevel(),
        "covariate_score_information": current_covariate(),
        "weighted_population_moments": current_population(),
    }
    former_values = {
        "multilevel_second_moment": former_multilevel(),
        "covariate_score_information": former_covariate(),
        "weighted_population_moments": former_population(),
    }
    operations = {
        "multilevel_second_moment": (current_multilevel, former_multilevel),
        "covariate_score_information": (current_covariate, former_covariate),
        "weighted_population_moments": (current_population, former_population),
    }
    return {
        "scope": "environment_specific_marginal_reduction_evidence",
        "interpretation": "No universal speed or memory claim.",
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "configuration": vars(args),
        "parity": {
            name: current_values[name] == former_values[name]
            for name in current_values
        },
        "measurements": {
            name: {
                "optimized": _measure(current, args.repetitions),
                "former": _measure(former, args.repetitions),
            }
            for name, (current, former) in operations.items()
        },
    }


def main() -> int:
    """Run the benchmark and print machine-readable JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=_positive_integer, default=7)
    parser.add_argument("--seed", type=int, default=2310)
    parser.add_argument("--clusters", type=_positive_integer, default=2_000)
    parser.add_argument("--contexts", type=_positive_integer, default=64)
    parser.add_argument("--groups", type=_positive_integer, default=8)
    parser.add_argument("--items", type=_positive_integer, default=32)
    parser.add_argument("--theta-nodes", type=_positive_integer, default=32)
    parser.add_argument("--latent-nodes", type=_positive_integer, default=64)
    report = build_report(parser.parse_args())
    if not all(report["parity"].values()):
        raise AssertionError("marginal reduction parity failed")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
