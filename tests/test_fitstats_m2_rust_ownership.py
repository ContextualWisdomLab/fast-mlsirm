"""Fail-first ownership contract for public limited-information M2."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _mirt_case() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Return one ordinary identified MIRT fixture that reaches public M2."""
    rng = np.random.default_rng(62701)
    n_persons = 300
    n_items = 5
    alpha = np.log(np.linspace(0.8, 1.4, n_items))
    b = np.linspace(-1.0, 1.0, n_items)
    theta = rng.standard_normal((n_persons, 1))
    eta = np.exp(alpha)[None, :] * theta + b[None, :]
    responses = (
        rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))
    ).astype(float)
    params = SimpleNamespace(
        alpha=alpha,
        b=b,
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
        theta=theta,
        xi=np.zeros((n_persons, 1)),
    )
    return responses, np.zeros(n_items, dtype=np.int64), params


def test_public_m2_requires_compiled_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary public M2 must not select the private NumPy reference implicitly."""
    responses, factor_id, params = _mirt_case()
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    with pytest.raises(
        RuntimeError,
        match="fit statistics require the compiled Rust core",
    ):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MIRT",
            q_theta=7,
            q_xi=7,
        )
