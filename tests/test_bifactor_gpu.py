# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU/GPU E-step equivalence for the polytomous bifactor GRM.

Tolerance derivation (documented, from float precision)
-------------------------------------------------------
The WGSL kernels accumulate in f32 (machine epsilon ≈ 1.19e-7) while the CPU
reference path is f64. Expected counts are reductions over persons of
posterior weights in [0, 1], so the worst-case absolute error per count entry
scales with ``n_persons`` (≈ 200 × 1.2e-7 ≈ 2.4e-5); the marginal log-likelihood
sums per-person terms every EM iteration, adding an iteration factor (≈ 10
here, hence the looser loglik bound). Fit-level slope/threshold agreement is
asserted at 1e-4 (about three orders of magnitude above the measured
f32-level differences of ~1e-7) and loglik agreement at 1e-2 absolute
(relative ~2e-6 on the fixture scale); both bounds were confirmed by
measurement on the fixture below (CPU vs GPU: slope 1.6e-7, threshold
7.8e-8, loglik 4.0e-5 at qmc_draws=241).

Quadrature is precision: equivalence is demonstrated at 241 and 481 Halton
draws (above the 121-draw floor for study settings); node counts are caller
arguments with no upper cap.

Implementation basis: Gibbons, R. D., & Hedeker, D. R. (1992).
Full-information item bi-factor analysis. *Psychometrika, 57*(3), 423–436.
https://doi.org/10.1007/BF02295430; Cai, L., Yang, J. S., & Hansen, M.
(2011). Generalized full-information item bifactor analysis. *Psychological
Methods, 16*(3), 221–248. https://doi.org/10.1037/a0023350
"""

import time

import numpy as np
import pytest

from fast_mlsirm.polytomous_bifactor import fit_polytomous_bifactor

SLOPE_ATOL = 1e-4
THRESHOLD_ATOL = 1e-4
LOGLIK_ATOL = 1e-2


def _fixture():
    n_persons = 200
    n_items = 10
    n_cat = 3
    n_dims = 3
    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1
        loading_pattern[i, 1 + (i % 2)] = 1

    rng = np.random.default_rng(42)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)

    # Reverse-keyed items (measurement-relevant fixture asymmetry)
    responses[:, 1] = n_cat - 1 - responses[:, 1]
    responses[:, 3] = n_cat - 1 - responses[:, 3]

    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1
    return responses, loading_pattern, n_cat, group_ids


@pytest.mark.parametrize("qmc_draws", [241, 481])
def test_bifactor_gpu_equivalence(qmc_draws, capfd):
    """CPU and GPU E-steps agree within f32-derived tolerance (multigroup)."""
    responses, loading_pattern, n_cat, group_ids = _fixture()

    t0 = time.perf_counter()
    fit_cpu = fit_polytomous_bifactor(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        max_iter=10,
        qmc_draws=qmc_draws,
        compute_oakes_se=False,
        device="cpu",
    )
    t1 = time.perf_counter()

    fit_gpu = fit_polytomous_bifactor(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        max_iter=10,
        qmc_draws=qmc_draws,
        compute_oakes_se=False,
        device="gpu",
    )
    t2 = time.perf_counter()

    cpu_time, gpu_time = t1 - t0, t2 - t1
    slope_diff = float(np.max(np.abs(fit_cpu.slope - fit_gpu.slope)))
    threshold_diff = float(np.max(np.abs(fit_cpu.threshold - fit_gpu.threshold)))
    loglik_diff = abs(float(fit_cpu.loglik) - float(fit_gpu.loglik))

    err = capfd.readouterr().err
    gpu_executed = "falling back" not in err
    print(
        f"\n[qn={qmc_draws}] CPU {cpu_time:.3f}s vs GPU {gpu_time:.3f}s "
        f"(gpu_executed={gpu_executed}); max|Δslope|={slope_diff:.3e} "
        f"max|Δthreshold|={threshold_diff:.3e} |Δloglik|={loglik_diff:.3e}"
    )

    # Convergence path must agree exactly (same iterations, same outcome).
    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    np.testing.assert_allclose(fit_cpu.slope, fit_gpu.slope, atol=SLOPE_ATOL)
    np.testing.assert_allclose(
        fit_cpu.threshold, fit_gpu.threshold, atol=THRESHOLD_ATOL
    )
    assert loglik_diff <= LOGLIK_ATOL
    # Group-moment agreement at the same single-precision level.
    np.testing.assert_allclose(
        fit_cpu.group_means, fit_gpu.group_means, atol=SLOPE_ATOL
    )
    np.testing.assert_allclose(
        fit_cpu.group_variances, fit_gpu.group_variances, atol=SLOPE_ATOL
    )
