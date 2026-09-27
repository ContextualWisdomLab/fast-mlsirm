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
    gpu = score_two_tier_grm_orthogonal(**kwargs, device="gpu")
    assert cpu.backend == "cpu"
    assert gpu.backend == "gpu"
    for name in ("mean", "second", "sd"):
        np.testing.assert_allclose(
            getattr(gpu, name), getattr(cpu, name), rtol=0, atol=1e-5
        )
    assert abs(cpu.loglik - gpu.loglik) < 1e-10
    fit_cpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, max_iter=1, tol=1e-6, device="cpu"
    )
    fit_gpu = fit_two_tier_grm_focal_orthogonal(
        **kwargs, max_iter=1, tol=1e-6, device="gpu"
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

    test_native_six_latent_continuous_gaussian_recovery(device="gpu")
