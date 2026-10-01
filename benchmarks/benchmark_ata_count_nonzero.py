"""Benchmark boolean array masking optimization in ATA."""

from __future__ import annotations

import timeit

import numpy as np


def benchmark_ata_boolean_count() -> None:
    """Compare np.sum vs np.count_nonzero on a realistic ATA workload.

    In automated test assembly (ATA) with content constraints, evaluating
    feasibility requires counting available items matching a specific label
    over a large boolean mask.
    """
    n_items = 200_000
    rng = np.random.default_rng(42)

    # 5 different labels uniformly distributed
    labels = rng.choice(["A", "B", "C", "D", "E"], size=n_items)

    # Eligible mask where roughly 50% are eligible
    eligible_now = rng.choice([True, False], size=n_items)

    lbl = "A"

    def use_sum() -> int:
        return int(np.sum(labels[eligible_now] == lbl))

    def use_count_nonzero() -> int:
        return int(np.count_nonzero(labels[eligible_now] == lbl))

    assert use_sum() == use_count_nonzero()

    sum_time = timeit.timeit(use_sum, number=100)
    count_nonzero_time = timeit.timeit(use_count_nonzero, number=100)

    print(f"ATA boolean counting benchmark (n={n_items}):")
    print(f"  np.sum:           {sum_time:.5f}s (100 iterations)")
    print(f"  np.count_nonzero: {count_nonzero_time:.5f}s (100 iterations)")
    print(f"  Speedup:          {sum_time / count_nonzero_time:.2f}x")


if __name__ == "__main__":
    benchmark_ata_boolean_count()
