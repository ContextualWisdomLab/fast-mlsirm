"""Fail-first Rust ownership contract for multigroup M2 (issue #627)."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def test_multigroup_m2_rejects_missing_core_before_reference_projection(monkeypatch):
    """Public multigroup M2 must not enter private NumPy projection arithmetic."""
    rng = np.random.default_rng(627)
    n_persons = 120
    n_items = 6
    group_id = np.repeat(np.arange(2, dtype=np.int64), n_persons // 2)
    factor_id = np.zeros(n_items, dtype=np.int64)
    group_mean = np.array([[0.0], [0.35]], dtype=np.float64)
    group_sd = np.ones((2, 1), dtype=np.float64)
    theta = rng.standard_normal(n_persons) + group_mean[group_id, 0]
    intercept = np.linspace(-0.9, 0.9, n_items)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] + intercept[None, :])))
    responses = (rng.random((n_persons, n_items)) < probability).astype(np.float64)
    params = SimpleNamespace(alpha=np.zeros(n_items), b=intercept, zeta=np.zeros((n_items, 1)), tau=-30.0)

    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    def reject_reference_projection(*_args, **_kwargs):
        raise AssertionError("public multigroup M2 entered NumPy projection arithmetic")

    monkeypatch.setattr(fitstats, "_projected_m2_numpy", reject_reference_projection)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fitstats.m2_multigroup(
            responses, factor_id, params, "MIRT", group_id, group_mean, group_sd, q_theta=7, q_xi=7
        )
