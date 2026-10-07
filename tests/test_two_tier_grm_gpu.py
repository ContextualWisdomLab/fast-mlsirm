# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU/GPU E-step equivalence for the two-tier GRM EM fit (#2282).

The two-tier reduced E-step (Cai, 2010, pp. 583-584) reuses the bifactor
WGSL kernels with the general node generalized to the flattened primary
product grid, so the tolerance derivation of ``tests/test_bifactor_gpu.py``
carries over: f32 device accumulation against the f64 CPU reference, with
1e-4 absolute agreement for slopes, thresholds and the primary correlation
and 1e-3 absolute for the loglik. The M-step runs on the CPU for both, so
the convergence path (iterations, flags) must agree exactly. Without a GPU
adapter the ``'gpu'`` fit falls back to CPU and the assertions hold
trivially.

References (APA 7th ed.): Cai, L. (2010). A two-tier full-information item
factor analysis model with applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0; Gibbons, R. D., Bock, R. D.,
Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik, D. K., Kupfer, D. J., Frank,
E., Grochocinski, V. J., & Stover, A. (2007). Full-information item bifactor
analysis of graded response data. *Applied Psychological Measurement,
31*(1), 4-19. https://doi.org/10.1177/0146621606289485
"""

import os
import time

import numpy as np
import pytest

from fast_mlsirm.two_tier_grm import fit_two_tier_grm

ATOL = 1e-4
LOGLIK_ATOL = 1e-3


def _fixture(n_persons=300, n_items=8, n_cat=3, seed=7):
    """Graded-response draws from a two-tier model (Cai, 2010, eq. 1)."""
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], n_persons)
    spec = rng.normal(size=(n_persons, 2))
    pmap = np.zeros((n_items, 2), dtype=bool)
    pmap[: n_items // 2, 0] = True
    pmap[n_items // 2 :, 1] = True
    smap = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    a = rng.uniform(0.8, 1.6, n_items)
    a_s = rng.uniform(0.4, 0.9, n_items)
    b = np.sort(rng.normal(size=(n_items, n_cat - 1)), axis=1)
    eta = (theta @ pmap.T.astype(float)) * a + spec[:, smap] * a_s
    p_ge = 1.0 / (1.0 + np.exp(-(eta[:, :, None] - b[None, :, :])))
    responses = (rng.uniform(size=p_ge.shape[:2])[:, :, None] < p_ge).sum(axis=2)
    return responses.astype(float), pmap, smap, n_cat


@pytest.mark.parametrize("primary_correlation", ["estimate", "identity"])
def test_two_tier_gpu_fit_matches_cpu(capfd, primary_correlation):
    responses, pmap, smap, n_cat = _fixture()
    kw = dict(
        n_cat=n_cat,
        n_primary=2,
        n_specific=2,
        q_primary=11,
        q_specific=11,
        max_iter=200,
        tol=1e-5,
        n_starts=1,
        seed=3,
        primary_correlation=primary_correlation,
    )
    t0 = time.perf_counter()
    cpu = fit_two_tier_grm(responses, pmap, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    gpu = fit_two_tier_grm(responses, pmap, smap, **kw, device="gpu")
    t2 = time.perf_counter()
    err = capfd.readouterr().err
    if os.environ.get("MLSIRM_REQUIRE_GPU") == "1":
        assert "falling back" not in err, err
    print(
        f"\n[two-tier P=2 q=11 {primary_correlation}] CPU {t1 - t0:.3f}s vs "
        f"GPU {t2 - t1:.3f}s (gpu_executed={'falling back' not in err}); "
        f"max|Δa|={np.max(np.abs(cpu.a_primary - gpu.a_primary)):.3e} "
        f"|Δloglik|={abs(cpu.loglik_trace[-1] - gpu.loglik_trace[-1]):.3e}"
    )
    assert cpu.converged == gpu.converged
    assert cpu.n_iter == gpu.n_iter
    np.testing.assert_allclose(cpu.a_primary, gpu.a_primary, atol=ATOL)
    np.testing.assert_allclose(cpu.a_specific, gpu.a_specific, atol=ATOL)
    np.testing.assert_allclose(cpu.threshold, gpu.threshold, atol=ATOL)
    np.testing.assert_allclose(cpu.phi, gpu.phi, atol=ATOL)
    assert abs(cpu.loglik_trace[-1] - gpu.loglik_trace[-1]) <= LOGLIK_ATOL


def _missing_mask(responses, smap, seed=11):
    """MAR mask with whole-block-missing rows.

    Persons ``4k + 1`` miss every item of specific block 0 and persons
    ``4k + 2`` miss block 1, so the GPU ``anyobs`` path sees blocks with
    no observation; the rest drop about 15 % of responses at random.
    """
    rng = np.random.default_rng(seed)
    y = responses.copy()
    y[rng.uniform(size=y.shape) < 0.15] = np.nan
    y[1::4, smap == 0] = np.nan
    y[2::4, smap == 1] = np.nan
    return y


@pytest.mark.parametrize("specific_free", [False, True])
def test_two_tier_gpu_fit_matches_cpu_with_missing_blocks(capfd, specific_free):
    """GPU/CPU fit agreement with missing blocks and specific-free items.

    ``specific_map == -1`` items contribute through the primaries only
    (Cai, 2010, pp. 583-584), so the GPU counts kernels must keep their
    expected counts; missing responses are dropped MAR on both devices.
    """
    responses, pmap, smap, n_cat = _fixture()
    if specific_free:
        smap = smap.copy()
        smap[[3, 7]] = -1
    y = _missing_mask(responses, smap)
    kw = dict(
        n_cat=n_cat,
        n_primary=2,
        n_specific=2,
        q_primary=11,
        q_specific=11,
        max_iter=200,
        tol=1e-5,
        n_starts=1,
        seed=3,
    )
    cpu = fit_two_tier_grm(y, pmap, smap, **kw, device="cpu")
    gpu = fit_two_tier_grm(y, pmap, smap, **kw, device="gpu")
    err = capfd.readouterr().err
    if os.environ.get("MLSIRM_REQUIRE_GPU") == "1":
        assert "falling back" not in err, err
    assert cpu.converged == gpu.converged
    assert cpu.n_iter == gpu.n_iter
    np.testing.assert_allclose(cpu.a_primary, gpu.a_primary, atol=ATOL)
    np.testing.assert_allclose(cpu.a_specific, gpu.a_specific, atol=ATOL)
    np.testing.assert_allclose(cpu.threshold, gpu.threshold, atol=ATOL)
    np.testing.assert_allclose(cpu.phi, gpu.phi, atol=ATOL)
    assert abs(cpu.loglik_trace[-1] - gpu.loglik_trace[-1]) <= LOGLIK_ATOL


def test_two_tier_rejects_unknown_device():
    responses, pmap, smap, n_cat = _fixture(n_persons=30)
    with pytest.raises(ValueError, match="device must be one of"):
        fit_two_tier_grm(
            responses, pmap, smap, n_cat, 2, 2, 5, 5, 2, 1e-3, 1, 0, device="tpu"
        )
