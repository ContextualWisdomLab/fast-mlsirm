"""Bound the NumPy GPCM parity-reference expected-count workspace."""

from __future__ import annotations

import runpy
import tracemalloc
from pathlib import Path

import numpy as np

MARGINAL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "fast_mlsirm"
    / "estimators"
    / "marginal.py"
)


def test_gpcm_expected_counts_use_bounded_category_workspace() -> None:
    """Expected counts must not allocate a persons-by-categories matrix."""
    namespace = runpy.run_path(str(MARGINAL_SOURCE))
    expected_counts = namespace["_gpcm_expected_category_counts"]
    n_persons, n_categories, n_nodes = 50_000, 64, 7
    responses = np.arange(n_persons, dtype=np.int64) % (n_categories - 1)
    posterior = np.full((n_persons, n_nodes), 1.0 / n_nodes)

    tracemalloc.start()
    try:
        counts = expected_counts(responses, posterior, n_categories)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 4 * 1024 * 1024
    assert counts.shape == (n_nodes, n_categories)
    assert np.all(counts[:, -1] == 0.0)
    assert np.allclose(counts.sum(axis=1), posterior.sum(axis=0))
