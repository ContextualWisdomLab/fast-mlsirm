#!/usr/bin/env python3
"""Report environment-specific timing and memory for infit/outfit reductions.

The benchmark compares the historical NumPy fallback equations with the
reviewed in-place reduction contract. Results characterize only the current
runtime, dimensions, dtype, and linked numerical libraries; they are not a
universal performance claim.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from typing import Callable

import numpy as np


def _positive_integer(value: str) -> int:
    """Return one positive command-line integer or raise an argparse error."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _rss_bytes() -> int:
    """Return maximum resident-set size in platform-normalized bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _numpy_configuration() -> str:
    """Return bounded NumPy and linked-library build information."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue().strip()[:32_768]


def _legacy(
    residual_squared: np.ndarray,
    variance: np.ndarray,
    observed: np.ndarray,
    observed_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the historical fallback equations."""
    outfit = (residual_squared / variance * observed).sum(axis=0) / observed_count
    infit = residual_squared.sum(axis=0) / np.maximum(
        (variance * observed).sum(axis=0),
        1e-12,
    )
    return infit, outfit


def _bounded(
    residual_squared: np.ndarray,
    variance: np.ndarray,
    observed: np.ndarray,
    observed_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reviewed in-place fallback equations."""
    working = residual_squared.copy()
    numerator = working.sum(axis=0)
    denominator = np.sum(variance, axis=0, where=observed)
    np.divide(working, variance, out=working)
    outfit = working.sum(axis=0) / observed_count
    infit = numerator / np.maximum(denominator, 1e-12)
    return infit, outfit


def _measure(
    callback: Callable[[], tuple[np.ndarray, np.ndarray]],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    """Measure elapsed distribution and traced/RSS memory for one callback."""
    for _ in range(warmups):
        callback()
    elapsed: list[float] = []
    traced: list[int] = []
    rss_before = _rss_bytes()
    checksum = 0.0
    for _ in range(repetitions):
        tracemalloc.start()
        started = time.perf_counter()
        infit, outfit = callback()
        elapsed.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced.append(int(peak))
        checksum += float(infit[0] + outfit[0])
    rss_after = _rss_bytes()
    return {
        "elapsed_seconds": {
            "minimum": min(elapsed),
            "median": statistics.median(elapsed),
            "maximum": max(elapsed),
            "values": elapsed,
        },
        "tracemalloc_peak_bytes": {
            "minimum": min(traced),
            "median": statistics.median(traced),
            "maximum": max(traced),
        },
        "maximum_rss_before_bytes": rss_before,
        "maximum_rss_after_bytes": rss_after,
        "maximum_rss_growth_bytes": max(rss_after - rss_before, 0),
        "checksum": checksum,
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    """Build one deterministic report for validated benchmark dimensions."""
    rng = np.random.default_rng(args.seed)
    probability = np.clip(
        rng.uniform(1e-4, 1.0 - 1e-4, (args.n_persons, args.n_items)),
        1e-12,
        1.0 - 1e-12,
    )
    response = (rng.random(probability.shape) < probability).astype(np.float64)
    observed = rng.random(probability.shape) >= args.missing_rate
    variance = probability * (1.0 - probability)
    residual_squared = np.subtract(response, probability)
    np.square(residual_squared, out=residual_squared)
    np.multiply(residual_squared, observed, out=residual_squared)
    observed_count = np.maximum(observed.sum(axis=0), 1)

    legacy_values = _legacy(residual_squared, variance, observed, observed_count)
    bounded_values = _bounded(residual_squared, variance, observed, observed_count)
    maximum_difference = max(
        float(np.max(np.abs(left - right), initial=0.0))
        for left, right in zip(legacy_values, bounded_values, strict=True)
    )
    return {
        "scope": "environment_specific_fitstats_infit_outfit_reduction_evidence",
        "claims": [
            "reuses_one_residual_squared_work_buffer_for_outfit_division",
            "uses_numpy_where_reduction_without_a_numeric_mask_copy",
            "does_not_claim_universal_speedup_or_capacity",
        ],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "operating_system": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy_configuration": _numpy_configuration(),
        },
        "inputs": {
            "n_persons": args.n_persons,
            "n_items": args.n_items,
            "missing_rate": args.missing_rate,
            "dtype": "float64",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "maximum_absolute_difference": maximum_difference,
        "legacy": _measure(
            lambda: _legacy(
                residual_squared,
                variance,
                observed,
                observed_count,
            ),
            warmups=args.warmups,
            repetitions=args.repetitions,
        ),
        "bounded": _measure(
            lambda: _bounded(
                residual_squared,
                variance,
                observed,
                observed_count,
            ),
            warmups=args.warmups,
            repetitions=args.repetitions,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write one machine-readable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-persons", type=_positive_integer, default=20_000)
    parser.add_argument("--n-items", type=_positive_integer, default=200)
    parser.add_argument("--missing-rate", type=float, default=0.15)
    parser.add_argument("--warmups", type=_positive_integer, default=2)
    parser.add_argument("--repetitions", type=_positive_integer, default=7)
    parser.add_argument("--seed", type=int, default=570)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("--missing-rate must be in [0, 1)")
    rendered = json.dumps(build_report(args), indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
