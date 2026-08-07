import inspect
import numpy as np
import tracemalloc
from fast_mlsirm import fitstats
from fast_mlsirm.types import MLSIRMParams

def test_fallback_source_reuses_residual_buffer_without_numeric_mask_copy() -> None:
    source = inspect.getsource(fitstats.infit_outfit)
    assert "observed.astype" not in source

def test_infit_outfit_allocation_regression(monkeypatch):
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: None)

    np.random.seed(42)
    N, J = 500, 100
    y = np.random.randint(0, 2, size=(N, J)).astype(float)
    observed = np.random.rand(N, J) > 0.1
    # include all missing items
    observed[:, 0] = False

    # include explicit masked out probabilities at boundary limits
    y[0, 1] = 1.0
    y[0, 2] = 0.0

    params = MLSIRMParams(
        theta=np.random.rand(N, 1),
        alpha=np.ones(J),
        b=np.zeros(J),
        xi=np.random.rand(N, 2),
        zeta=np.random.rand(J, 2),
        tau=0.5
    )

    # Run once to warm up JIT / import overheads
    fitstats.infit_outfit(y, np.zeros(J, dtype=int), params, "mlsirm", mask=observed)

    tracemalloc.start()
    res = fitstats.infit_outfit(y, np.zeros(J, dtype=int), params, "mlsirm", mask=observed)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert peak memory is within bounds of intermediate probability matrices
    # but strictly less than 1.5 extra N x J buffers
    assert peak < (N * J * 8) * 8 # < 3.2MB

    assert "infit" in res
    assert "outfit" in res

    assert not np.isnan(res["infit"][1:]).all()
    assert not np.isnan(res["outfit"][1:]).all()

    # Infit denominator was sum(v, where=observed)
    # The sum of resid2 for missing column will be 0.
    # 0 / n_obs (which is max(sum(obs), 1)) = 0 / 1 = 0
    # For infit, 0 / max(infit_denominator, 1e-12) = 0
    assert res["outfit"][0] == 0.0
    assert res["infit"][0] == 0.0

    assert not np.isnan(res["outfit"][1])
    assert not np.isnan(res["infit"][1])
