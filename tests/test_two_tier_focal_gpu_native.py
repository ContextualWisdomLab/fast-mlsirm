"""Actual hardware parity for the P2/S4 fixed-bank focal GPU route.

Cai (2010), pp.608-609 Appendices A/B, DOI 10.1007/s11336-010-9178-0,
defines shared-product and shared-specific joint posterior moments. Reuse the
existing Rust full-product fixture, including missing persons/blocks. GPU
weights are WGSL f32 (https://www.w3.org/TR/WGSL/#floating-point-types), with
f64 host contraction. Absolute 1e-5 moment/parameter and 1e-10 certified likelihood bounds
are numerical regression choices, not scientific cutoffs or a coverage claim.
A one-update comparison is not convergence or continuous population recovery.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fast_mlsirm import (
    fit_two_tier_grm_focal_orthogonal,
    score_two_tier_grm_orthogonal,
)


@pytest.mark.skipif(
    os.environ.get("FOCAL_GPU_NATIVE") != "1",
    reason="requires actual GPU; run with FOCAL_GPU_NATIVE=1",
)
def test_actual_gpu_six_latent_score_and_focal_update():
    """Check every latent moment and a common Gaussian EM update on hardware.

    See Cai Appendix B and WGSL precision limits in the module docstring.
    The explicit GPU route rejects software adapters and unavailable devices.
    """
    f = json.loads(
        (
            Path(__file__).parent / "fixtures/two_tier_focal/six_latent_same_node.json"
        ).read_text()
    )
    y = np.array(f["responses"], dtype=float).reshape(3, 16)
    y[~np.array(f["observed"], dtype=bool).reshape(3, 16)] = np.nan
    kwargs = {
        "responses": y,
        "primary_map": np.array(f["primary_map"], dtype=bool).reshape(16, 2),
        "specific_map": np.array(f["specific_map"], dtype=np.int64),
        "a_primary": np.array(f["a_primary"]).reshape(16, 2),
        "a_specific": np.array(f["a_specific"]),
        "threshold": np.array(f["threshold"]).reshape(16, 3),
        "latent_mean": np.array(f["latent_mean"]),
        "latent_sd": np.array(f["latent_sd"]),
        "n_cat": 4,
        "n_primary": 2,
        "n_specific": 4,
        "q_primary": 5,
        "q_specific": 5,
    }
    cpu = score_two_tier_grm_orthogonal(**kwargs, device="cpu")
    gpu = score_two_tier_grm_orthogonal(**kwargs, device="gpu", gpu_memory_budget_bytes=1 << 30)
    # Resource budgets are test choices, not scientific accuracy settings.
    # 40 KiB admits fixed inputs plus one person, forcing multiple batches.
    batched = score_two_tier_grm_orthogonal(
        **kwargs, device="gpu", gpu_memory_budget_bytes=40 * 1024
    )
    for name in ("mean", "second", "sd"):
        np.testing.assert_allclose(getattr(batched, name), getattr(gpu, name),
                                   rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="GPU fixed inputs exceed"):
        score_two_tier_grm_orthogonal(
            **kwargs, device="gpu", gpu_memory_budget_bytes=1
        )
    assert cpu.backend == "cpu"
    assert gpu.backend == "gpu"
    assert gpu.gpu_memory_budget_bytes == 1 << 30
    assert batched.gpu_memory_budget_bytes == 40 * 1024
    for name in ("mean", "second", "sd"):
        np.testing.assert_allclose(
            getattr(gpu, name), getattr(cpu, name), rtol=0, atol=1e-5
        )
    assert abs(cpu.loglik - gpu.loglik) < 1e-10
    fit_cpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, max_iter=1, tol=1e-6, device="cpu"
    )
    fit_gpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, max_iter=1, tol=1e-6, device="gpu", gpu_memory_budget_bytes=1 << 30
    )
    assert fit_gpu.scores.backend == "gpu"
    assert fit_cpu.n_iter == fit_gpu.n_iter == 1
    np.testing.assert_allclose(
        fit_gpu.latent_mean, fit_cpu.latent_mean, rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(fit_gpu.latent_sd, fit_cpu.latent_sd, rtol=0, atol=1e-5)
    print(
        json.dumps(
            {
                "backend": gpu.backend,
                "gpu_memory_budget_bytes": 1 << 30,
                "batched_memory_budget_bytes": 40 * 1024,
                "score_loglik_delta": gpu.loglik - cpu.loglik,
                "fit_mean_max_delta": float(
                    np.max(np.abs(fit_gpu.latent_mean - fit_cpu.latent_mean))
                ),
                "fit_sd_max_delta": float(
                    np.max(np.abs(fit_gpu.latent_sd - fit_cpu.latent_sd))
                ),
            }
        )
    )


@pytest.mark.skipif(
    os.environ.get("FOCAL_GPU_RECOVERY") != "1",
    reason="prespecified continuous six-latent GPU recovery; FOCAL_GPU_RECOVERY=1",
)
def test_actual_gpu_six_latent_continuous_population_recovery():
    """Reuse the committed Gaussian generator and unchanged recovery criteria.

    Cai (2010), Appendices A/B and Python 3.12 random sources are documented
    in the shared fixture. GPU mixed precision follows this module's WGSL
    citation. Preserve nonconvergence/failures; never tune after an outcome.
    """
    from test_two_tier_focal_gaussian_recovery_native import (
        test_native_six_latent_continuous_gaussian_recovery,
    )

    test_native_six_latent_continuous_gaussian_recovery(device="gpu", gpu_memory_budget_bytes=1 << 30)


@pytest.mark.skipif(
    os.environ.get("FOCAL_GPU_NATIVE") != "1",
    reason="requires actual GPU; run with FOCAL_GPU_NATIVE=1",
)
def test_actual_gpu_reference_focal_expected_score_pipeline():
    """Reuse the reference→fixed-bank focal→reference-curve contract.

    Cai (2010), Appendices A/B; source and limits are in the reused test.
    """
    from test_two_tier_focal_gaussian_recovery_native import (
        test_native_reference_focal_expected_score_pipeline,
    )
    test_native_reference_focal_expected_score_pipeline(device="gpu")


@pytest.mark.skipif(
    os.environ.get("FOCAL_GPU_NATIVE") != "1",
    reason="requires actual GPU; run with FOCAL_GPU_NATIVE=1",
)
def test_actual_gpu_primary_only_matches_unidimensional_and_focal_prior():
    """Verify the primary-only reduction and estimated-prior GPU route.

    Cai (2010), p.587 eq.7 and p.589 eqs.11-12 remove the specific term.
    Compare the standard-normal case with the separate unidimensional Rust
    EAP implementation, then compare nonstandard-prior CPU/GPU moments and
    one EM update (not convergence/recovery). Bounds are measured regression
    tolerances inherited from this module, not research acceptance settings.

    References
    ----------
    Cai, L. (2010). A two-tier full-information item factor analysis model
    with applications. Psychometrika, 75(4), 581-612.
    https://doi.org/10.1007/s11336-010-9178-0
    """
    from fast_mlsirm.polytomous import PolytomousFit, score_polytomous

    y = np.array([[0, 1, 2, 1], [2, 0, 1, 2], [1, np.nan, 0, 2],
                  [np.nan, np.nan, np.nan, np.nan]], dtype=float)
    a = np.array([0.8, 1.2, 0.9, 1.1])
    d = np.array([[0.8, -0.7], [0.4, -1.0], [1.2, -0.2], [0.5, -0.9]])
    kwargs = dict(responses=y, primary_map=np.ones((4, 1), dtype=bool),
                  specific_map=np.full(4, -1, dtype=np.int64),
                  a_primary=a[:, None], a_specific=np.zeros(4), threshold=d,
                  n_cat=3, n_primary=1, n_specific=0,
                  q_primary=15, q_specific=1)
    standard = score_two_tier_grm_orthogonal(
        **kwargs, latent_mean=np.zeros(1), latent_sd=np.ones(1), device="cpu")
    independent = score_polytomous(
        y, PolytomousFit(model="grm", slope=a, cat_params=d,
                         loglik=0.0, n_iter=0), q_theta=15)
    np.testing.assert_allclose(standard.mean[:, 0], independent["theta_eap"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(standard.sd[:, 0], independent["theta_sd"],
                               rtol=0, atol=1e-12)
    prior = dict(latent_mean=np.array([0.4]), latent_sd=np.array([1.2]))
    cpu = score_two_tier_grm_orthogonal(**kwargs, **prior, device="cpu")
    gpu = score_two_tier_grm_orthogonal(
        **kwargs, **prior, device="gpu", gpu_memory_budget_bytes=1 << 20)
    assert gpu.backend == "gpu"
    assert gpu.mean.shape == (4, 1)
    for name in ("mean", "second", "sd"):
        np.testing.assert_allclose(getattr(gpu, name), getattr(cpu, name),
                                   rtol=0, atol=1e-5)
    assert abs(cpu.loglik - gpu.loglik) < 1e-10
    assert not np.allclose(cpu.mean, standard.mean)
    np.testing.assert_allclose(gpu.mean[-1], prior["latent_mean"], atol=1e-5)
    np.testing.assert_allclose(gpu.sd[-1], prior["latent_sd"], atol=1e-5)
    fit_cpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, **prior, max_iter=1, tol=1e-6, device="cpu")
    fit_gpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, **prior, max_iter=1, tol=1e-6, device="gpu",
        gpu_memory_budget_bytes=1 << 20)
    assert fit_gpu.scores.backend == "gpu"
    assert fit_cpu.n_iter == fit_gpu.n_iter == 1
    np.testing.assert_allclose(fit_gpu.latent_mean, fit_cpu.latent_mean, atol=1e-5)
    np.testing.assert_allclose(fit_gpu.latent_sd, fit_cpu.latent_sd, atol=1e-5)
    with pytest.raises(ValueError, match="specific_map"):
        score_two_tier_grm_orthogonal(
            **{**kwargs, "specific_map": np.zeros(4, dtype=np.int64)},
            **prior, device="gpu", gpu_memory_budget_bytes=1 << 20)
    with pytest.raises(ValueError, match="fixed pattern positions"):
        score_two_tier_grm_orthogonal(
            **{**kwargs, "a_specific": np.ones(4)},
            **prior, device="gpu", gpu_memory_budget_bytes=1 << 20)
    print(json.dumps({"backend": gpu.backend, "n_primary": 1, "n_specific": 0,
                      "q_primary": 15, "q_specific": 1,
                      "scope": "score reduction and one Gaussian update"}))


@pytest.mark.skipif(
    os.environ.get("FOCAL_GPU_NATIVE") != "1",
    reason="requires actual GPU; run with FOCAL_GPU_NATIVE=1",
)
def test_actual_gpu_primary_only_group_pipeline():
    """Execute the reference-bank/focal-prior/score composition with S=0.

    Synthetic response generation uses Cai (2010), p.589 eqs.11-12.
    This numerical execution check uses 121 primary nodes, explicit settings
    and no nuisance dimensions; it is not acceptance of any study scale.

    References
    ----------
    Cai, L. (2010). A two-tier full-information item factor analysis model
    with applications. Psychometrika, 75(4), 581-612.
    https://doi.org/10.1007/s11336-010-9178-0
    """
    from fast_mlsirm.two_tier_focal import fit_score_two_tier_groups
    from fast_mlsirm.two_tier_grm import expected_total_score_two_tier_from_fit
    rng = np.random.default_rng(20260928)
    groups = np.tile(np.array([1, 0], dtype=np.int64), 512)
    theta = rng.normal(np.where(groups == 1, 0.0, 0.4),
                       np.where(groups == 1, 1.0, 1.1))
    a = np.array([0.8, 1.2, 0.9, 1.1])
    d = np.array([[0.8, -0.7], [0.4, -1.0], [1.2, -0.2], [0.5, -0.9]])
    cumulative = 1.0 / (1.0 + np.exp(-(theta[:, None, None] * a[None, :, None]
                                      + d[None, :, :])))
    uniforms = rng.random((1024, 4, 1))
    y = (uniforms < cumulative).sum(axis=2)
    pm = np.ones((4, 1), dtype=bool)
    sm = np.full(4, -1, dtype=np.int64)
    result = fit_score_two_tier_groups(
        y, groups, pm, sm, n_groups=2, reference_group=1,
        n_cat=3, n_primary=1, n_specific=0, focal_primary=0,
        initial_mean=np.zeros((2, 1)), initial_sd=np.ones((2, 1)),
        fit_q_primary=121, fit_q_specific=1,
        score_q_primary=121, score_q_specific=1, q_nuisance=1,
        max_iter=2000, tol=1e-6, n_starts=2, seed=20260928,
        device="gpu", gpu_memory_budget_bytes=1 << 28, cache_item_tables=True)
    for group in (0, 1):
        assert result["group_fits"][group].converged
        assert result["group_scores"][group].backend == "gpu"
        np.testing.assert_array_equal(result["group_rows"][group],
                                      np.flatnonzero(groups == group))
    focal = result["group_fits"][0]
    reference = result["reference_fit"]
    rescored = score_two_tier_grm_orthogonal(
        y[groups == 0], pm, sm, reference.a_primary,
        reference.a_specific, reference.threshold,
        focal.latent_mean, focal.latent_sd,
        n_cat=3, n_primary=1, n_specific=0,
        q_primary=121, q_specific=1, device="gpu",
        gpu_memory_budget_bytes=1 << 28)
    np.testing.assert_allclose(result["theta"][groups == 0], rescored.mean[:, 0],
                               rtol=0, atol=1e-12)
    assert np.isfinite(result["expected_total"]).all()
    curve = expected_total_score_two_tier_from_fit(
        reference, result["theta"], specific_map=sm, focal_primary=0,
        q_nuisance=1, orthogonal_primary_identification=True,
        primary_ref_mean=np.zeros(1), primary_ref_sd=np.ones(1),
        specific_ref_mean=np.empty(0), specific_ref_sd=np.empty(0))
    np.testing.assert_allclose(result["expected_total"], curve.expected_total,
                               rtol=0, atol=1e-12)
    print(json.dumps({"backend": "gpu", "n_specific": 0,
                      "reference_group": 1, "primary_nodes": 121,
                      "reference_iterations": reference.n_iter,
                      "focal_iterations": focal.n_iter}))
