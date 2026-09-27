"""Actual grouped producer composition; synthetic choices are not study settings."""
import numpy as np
import pytest
from fast_mlsirm.two_tier_focal import (
    fit_score_two_tier_groups, score_two_tier_grm_orthogonal,
)
from test_two_tier_focal_gaussian_recovery_native import _continuous_six_latent_fixture


def test_group_pipeline_preserves_slots_priors_and_nonconvergence():
    """Exercise Cai2010 reference/focal/EAP composition on actual M1 GPU.

    Reuse the bridge fixture and source contracts, with reference label1 and
    interleaved slots. Seven fit/nine score nodes are synthetic execution
    choices; no study sensitivity, population recovery or intervals claimed.
    """
    ref, ap, _, sm, _, _ = _continuous_six_latent_fixture(
        512, np.zeros(6), np.ones(6), 20260928,
    )
    focal, _, _, _, _, _ = _continuous_six_latent_fixture(
        512, np.array([.30, -.25, .20, -.30, .35, -.15]),
        np.array([1.15, .85, .80, 1.20, 1.10, .90]), 20260929,
    )
    y = np.stack([focal, ref], axis=1).reshape(1024, 16)
    groups = np.tile([0, 1], 512)
    kw = dict(
        responses=y, group_ids=groups, primary_map=ap != 0, specific_map=sm,
        n_groups=2, reference_group=1, n_cat=4, n_primary=2, n_specific=4,
        focal_primary=0, initial_mean=np.zeros((2, 6)), initial_sd=np.ones((2, 6)),
        fit_q_primary=7, fit_q_specific=7, score_q_primary=9, score_q_specific=9,
        q_nuisance=121, max_iter=2000, tol=1e-6, n_starts=2, seed=20260928,
        device="gpu", gpu_memory_budget_bytes=1 << 30, cache_item_tables=True,
    )
    with pytest.raises(ValueError, match="group_ids"):
        fit_score_two_tier_groups(**(kw | {"group_ids": groups + .5}))
    with pytest.raises(RuntimeError) as failed:
        fit_score_two_tier_groups(**(kw | {"max_iter": 1}))
    assert failed.value.group_id == 1
    assert not failed.value.fit.converged
    out = fit_score_two_tier_groups(**kw)
    assert all(f.converged for f in out["group_fits"].values())
    assert all(s.backend == "gpu" for s in out["group_scores"].values())
    np.testing.assert_array_equal(out["group_rows"][1], np.arange(1, 1024, 2))
    bank = out["reference_fit"]
    f = out["group_fits"][0]
    independent = score_two_tier_grm_orthogonal(
        focal, ap != 0, sm, bank.a_primary, bank.a_specific, bank.threshold,
        f.latent_mean, f.latent_sd, n_cat=4, n_primary=2, n_specific=4,
        q_primary=9, q_specific=9, device="gpu", gpu_memory_budget_bytes=1 << 30,
        cache_item_tables=True,
    )
    np.testing.assert_allclose(out["theta"][::2], independent.mean[:, 0], atol=1e-10)
    np.testing.assert_array_equal(out["theta"][1::2], out["group_scores"][1].mean[:, 0])
    assert np.isfinite(out["expected_total"]).all()
    assert out["settings"]["score_q_primary"] == 9
