"""Fail-first Rust ownership contract for multilevel M2 (issue #627)."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _multilevel_case():
    """Return a bounded random-intercept MIRT case with enough clusters."""
    rng = np.random.default_rng(6271)
    n_items = 6
    n_clusters = 40
    cluster_size = 3
    cluster_id = np.repeat(np.arange(n_clusters, dtype=np.int64), cluster_size)
    intercept = np.linspace(-0.9, 0.9, n_items)
    random_intercept = np.repeat(0.5 * rng.standard_normal(n_clusters), cluster_size)
    probability = 1.0 / (
        1.0 + np.exp(-(random_intercept[:, None] + intercept[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(np.float64)
    params = SimpleNamespace(
        alpha=np.zeros(n_items),
        b=intercept,
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    return responses, np.zeros(n_items, dtype=np.int64), params, cluster_id


def _run_multilevel(case):
    """Call the public multilevel M2 path with bounded quadrature."""
    responses, factor_id, params, cluster_id = case
    return fitstats.m2_multilevel(
        responses,
        factor_id,
        params,
        "MIRT",
        cluster_id,
        0.5,
        q_theta=7,
        q_u=7,
        q_xi=7,
    )


def test_multilevel_m2_rejects_missing_core_before_reference_projection(monkeypatch):
    """Public multilevel M2 fails closed without Rust projection ownership."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    monkeypatch.setattr(
        fitstats,
        "_projected_m2_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multilevel M2 entered NumPy projection arithmetic")
        ),
    )

    with pytest.raises(
        RuntimeError, match="fit statistics require the compiled Rust core"
    ):
        _run_multilevel(_multilevel_case())


def test_multilevel_m2_delegates_target_and_null_projection_to_rust(monkeypatch):
    """Both multilevel target and independence projections cross the Rust boundary."""
    calls = []

    class ProjectionCore:
        @staticmethod
        def chi2_sf(_x, _df):
            return 0.5

        @staticmethod
        def projected_m2(residual, delta, xi, n):
            calls.append(
                (
                    np.asarray(residual).shape,
                    np.asarray(delta).shape,
                    np.asarray(xi).shape,
                    float(n),
                )
            )
            return 8.0 if len(calls) == 1 else 12.0

    monkeypatch.setattr(fitstats, "_core_module", lambda: ProjectionCore())
    monkeypatch.setattr(
        fitstats,
        "_projected_m2_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multilevel M2 entered NumPy projection arithmetic")
        ),
    )

    result = _run_multilevel(_multilevel_case())

    assert result.m2 == 8.0
    assert result.null_m2 == 12.0
    assert len(calls) == 2
    for residual_shape, delta_shape, xi_shape, n in calls:
        assert residual_shape == (21,)
        assert xi_shape == (21, 21)
        assert n == 120.0
    assert calls[0][1] == (21, 13)
    assert calls[1][1] == (21, 6)
