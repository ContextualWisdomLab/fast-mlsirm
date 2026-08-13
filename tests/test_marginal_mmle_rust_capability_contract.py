"""Fail-first ABI compatibility contract for Rust marginal MMLE dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.config import FitConfig


fit_module = importlib.import_module("fast_mlsirm.fit")
marginal_module = importlib.import_module("fast_mlsirm.estimators.marginal")


def test_public_spatial_mmle_rejects_stale_rust_marginal_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable from an older marginal ABI must fail before dispatch."""
    responses = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    stale_calls: list[tuple[object, ...]] = []
    numpy_calls: list[tuple[object, ...]] = []

    def stale_fit_marginal(
        y: object,
        observed: object,
        factors: object,
        n_persons: object,
        n_items: object,
        n_dims: object,
        latent_dim: object,
        model: object,
        eps_distance: object,
    ) -> None:
        stale_calls.append(
            (
                y,
                observed,
                factors,
                n_persons,
                n_items,
                n_dims,
                latent_dim,
                model,
                eps_distance,
            )
        )

    def reject_numpy_reference(*args: object, **_kwargs: object) -> dict[str, object]:
        numpy_calls.append(args)
        raise AssertionError("public marginal MMLE entered NumPy production arithmetic")

    stale_core = SimpleNamespace(
        fit_marginal=stale_fit_marginal,
        MARGINAL_CAPABILITY_VERSION=0,
    )
    monkeypatch.setattr(fast_mlsirm, "_core", stale_core, raising=False)
    monkeypatch.setattr(fit_module, "resolve_backend", lambda _backend: "rust")
    monkeypatch.setattr(marginal_module, "fit_marginal_numpy", reject_numpy_reference)

    with pytest.raises(RuntimeError, match=r"compiled Rust core.*marginal"):
        fit_module.fit(
            responses,
            factor_id,
            FitConfig(
                estimator="mmle",
                model="MLS2PLM",
                backend="rust",
                latent_dim=1,
                max_iter=1,
                n_restarts=1,
            ),
        )

    assert stale_calls == []
    assert numpy_calls == []
