import time
import numpy as np
import pytest
from fast_mlsirm.polytomous_bifactor import fit_polytomous_bifactor

def test_bifactor_gpu_equivalence():
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
    
    # Reverse-keyed items (simulate)
    responses[:, 1] = n_cat - 1 - responses[:, 1]
    responses[:, 3] = n_cat - 1 - responses[:, 3]

    t0 = time.perf_counter()
    fit_cpu = fit_polytomous_bifactor(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        max_iter=10,
        device="cpu",
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    fit_gpu = fit_polytomous_bifactor(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        max_iter=10,
        device="gpu",
    )
    t3 = time.perf_counter()

    print(f"\nCPU Time: {t1 - t0:.3f}s")
    print(f"GPU Time: {t3 - t2:.3f}s")

    np.testing.assert_allclose(fit_cpu.slope, fit_gpu.slope, atol=1e-5)
    np.testing.assert_allclose(fit_cpu.threshold, fit_gpu.threshold, atol=1e-5)
    np.testing.assert_allclose(fit_cpu.loglik, fit_gpu.loglik, atol=1e-5)

