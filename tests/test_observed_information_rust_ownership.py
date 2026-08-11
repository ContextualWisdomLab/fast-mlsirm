"""Fail-first ownership contract for public observed-information arithmetic.

The public inference API may validate and marshal in Python, but Hessian
construction and symmetrization determine reported uncertainty and therefore
belong to the compiled Rust numerical core.  The test installs an unmistakable
compiled-core sentinel and requires the public API to transport that exact
matrix instead of independently finite-differencing the likelihood in Python.
"""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
import fast_mlsirm.inference as inference
from fast_mlsirm.config import FitConfig
from fast_mlsirm.types import MLSIRMParams


def _small_mirt_problem() -> tuple[np.ndarray, np.ndarray, MLSIRMParams]:
    """Return a tiny identified-shape MIRT fixture for ownership delegation."""
    responses = np.array([[1.0], [0.0]], dtype=np.float64)
    factor_id = np.array([0], dtype=np.int64)
    params = MLSIRMParams(
        theta=np.array([[0.25], [-0.25]], dtype=np.float64),
        alpha=np.array([0.1], dtype=np.float64),
        b=np.array([0.0], dtype=np.float64),
        xi=np.zeros((2, 1), dtype=np.float64),
        zeta=np.zeros((1, 1), dtype=np.float64),
        tau=-2.0,
    )
    return responses, factor_id, params


def test_public_observed_information_delegates_matrix_to_rust(monkeypatch) -> None:
    """The exact public Hessian result must come from the Rust numerical owner."""
    responses, factor_id, params = _small_mirt_problem()
    config = FitConfig(model="MIRT", backend="rust")
    packed_dimension = int(inference._pack(params, config.normalized_model()).size)
    sentinel = np.diag(np.arange(1, packed_dimension + 1, dtype=np.float64))
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_observed_information(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel.copy()

    monkeypatch.setattr(
        core,
        "observed_information",
        fake_observed_information,
        raising=False,
    )

    result = inference.observed_information(
        responses,
        factor_id,
        params,
        config=config,
        backend="rust",
        device="cpu",
        step=1e-4,
    )

    assert len(calls) == 1
    assert np.array_equal(result, sentinel)
