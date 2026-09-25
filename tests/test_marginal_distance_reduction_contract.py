"""Behavioral binary64 contract for the NumPy marginal distance reduction."""

from __future__ import annotations

import numpy as np

import fast_mlsirm.estimators.marginal as marginal


_DIFF = np.array(
    [[
        float.fromhex("0x1.9a4c408bd8417p+0"),
        float.fromhex("0x1.7f38d14a8fe3bp-14"),
        float.fromhex("0x1.b02b980a9c79cp+2"),
    ]],
    dtype=np.float64,
)


def test_rowwise_squared_distance_keeps_established_binary64_reduction() -> None:
    """The proposed reassociation changes this exact finite binary64 fixture."""
    established = np.sum(_DIFF * _DIFF, axis=1)
    proposed = np.einsum("ij,ij->i", _DIFF, _DIFF)

    assert established[0].hex() == "0x1.815656f4f071ap+5"
    assert proposed[0].hex() == "0x1.815656f4f0719p+5"
    assert not np.array_equal(established, proposed)


def test_distance_reassociation_changes_the_live_estimator_path(monkeypatch) -> None:
    """The one-ULP reduction change must propagate through the production M-step."""
    x_grid = np.ascontiguousarray(_DIFF, dtype=np.float64)
    x_logw = np.zeros(1, dtype=np.float64)

    def one_point_xi_grid(*_args, **_kwargs):
        return x_grid.copy(), x_logw.copy()

    monkeypatch.setattr(marginal, "_xi_nodes", one_point_xi_grid)

    y = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    observed = np.ones_like(y, dtype=bool)
    factor_id = np.zeros(1, dtype=np.int64)
    fit_kwargs = {
        "model": "ULS2PLM",
        "n_dims": 1,
        "latent_dim": 3,
        "pop": {"kind": "single"},
        "q_theta": 7,
        "q_xi": 7,
        "max_iter": 1,
        "m_steps": 1,
        "init_zeta_radius": 0.0,
        "xi_rule": "qmc",
        "xi_points": 1,
        "eps_distance": float.fromhex("0x0.0000000000001p-1022"),
    }

    baseline = marginal.fit_marginal_numpy(y, observed, factor_id, **fit_kwargs)

    original_sum = np.sum
    proposed_distance = np.einsum("ij,ij->i", _DIFF, _DIFF)
    target_squared = _DIFF * _DIFF
    reassociation_seen = False

    def reassociated_sum(values, *args, **kwargs):
        nonlocal reassociation_seen
        axis = kwargs.get("axis", args[0] if args else None)
        array = np.asarray(values)
        if axis == 1 and array.shape == target_squared.shape and np.array_equal(
            array, target_squared
        ):
            reassociation_seen = True
            return proposed_distance.copy()
        return original_sum(values, *args, **kwargs)

    monkeypatch.setattr(marginal.np, "sum", reassociated_sum)
    reassociated = marginal.fit_marginal_numpy(y, observed, factor_id, **fit_kwargs)

    assert reassociation_seen, "fixture must exercise the production zeta-distance reduction"
    assert not np.array_equal(
        baseline["zeta"].view(np.uint64), reassociated["zeta"].view(np.uint64)
    ), "the one-ULP distance reassociation must be observable in estimator output"
