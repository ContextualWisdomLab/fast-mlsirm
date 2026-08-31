"""Fail-first provenance contract for the legacy Rust MMLE result boundary."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.config import FitConfig


fit_module = importlib.import_module("fast_mlsirm.fit")


def test_mmle_seals_native_item_vector_before_numpy_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained PyO3 vectors must not redefine already-returned fit evidence."""
    native_a = [1.25, 0.75]
    native_b = [-0.2, 0.2]
    native_theta = [-0.5, 0.0, 0.5]
    native_trace = [-4.0, -3.0]

    fake_core = types.ModuleType("fast_mlsirm._core")

    def fake_fit_mmle_2pl(*_args: object) -> tuple[object, ...]:
        return native_a, native_b, native_theta, native_trace, True

    fake_core.fit_mmle_2pl = fake_fit_mmle_2pl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake_core)
    monkeypatch.setattr(fast_mlsirm, "_core", fake_core, raising=False)

    real_asarray = np.asarray
    seam_calls = 0

    def mutate_provider_before_conversion(
        value: object,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        nonlocal seam_calls
        seam_calls += 1
        if seam_calls == 1:
            native_a[0] = 99.0
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(fit_module.np, "asarray", mutate_provider_before_conversion)

    responses = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    observed = np.ones_like(responses, dtype=bool)
    result = fit_module._fit_mmle(
        responses,
        observed,
        "ULS2PLM",
        FitConfig(estimator="mmle", max_iter=2, n_restarts=1),
        "rust",
    )

    assert native_a[0] == 99.0
    np.testing.assert_allclose(np.exp(result.params.alpha), [1.25, 0.75])
