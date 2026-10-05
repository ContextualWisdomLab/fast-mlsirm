"""Bifactor FIPC device transport without numerical execution.

Kim (2006, pp. 360–363) supplies the fixed-parameter calibration context;
these tests characterize implementation dispatch, not a statistical theorem.

Reference: Kim, S. (2006). A comparative study of IRT fixed parameter
calibration methods. Journal of Educational Measurement, 43(4), 355–381.
https://doi.org/10.1111/j.1745-3984.2006.00021.x
"""
from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import fit_bifactor_grm_fipc
import fast_mlsirm.fitstats as fitstats


def test_fipc_device_controls_reach_native_without_changing_numerics(monkeypatch):
    """Characterize transport with fixed anchors, Q121/241 and a finite return."""
    y = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], dtype=np.int64)
    smap = np.array([0, 0, 1, 1], dtype=np.int64)
    anchor = np.array([True, False, True, False])
    slope = np.ones(4)
    threshold = np.tile([1.0, -1.0], (4, 1))
    before = [a.copy() for a in (y, smap, anchor, slope, threshold)]
    calls = []
    fields = {}

    class Recorder:
        def fit_bifactor_grm_fipc(self, *args, **kwargs):
            calls.append((args, kwargs))
            result = dict(
                a_general=slope.tolist(), a_specific=slope.tolist(),
                threshold=threshold.reshape(-1).tolist(), general_mean=0.0,
                general_sd=1.0, specific_sd=[1.0, 1.0], theta_g_eap=[0.0]*3,
                theta_g_sd=[1.0]*3, category_counts=[1]*12,
                loglik_trace=[-2.0, -1.0], n_iter=1, converged=False,
                termination_reason="max_iter_reached", final_loglik_change=1.0,
                n_parameters=8, gpu_execution_used=kwargs.get("device") == "gpu",
                gpu_backend="fixture-Metal" if kwargs.get("device") == "gpu" else None,
                gpu_device_name="fixture-device" if kwargs.get("device") == "gpu" else None,
                cpu_fallback_reason=None,
            )
            result.update(fields)
            return result

    monkeypatch.setattr(fitstats, "_core_module", lambda: Recorder())
    args = (y, smap, 3, 2, anchor, slope, slope, threshold)
    controls = dict(q_general=121, q_specific=241, max_iter=100, tol=1e-5,
                    newton_iter=5, ridge=1e-8, estimate_specific_vars=False)
    cpu = fit_bifactor_grm_fipc(*args, **controls)
    assert not cpu.converged and not cpu.gpu_execution_used
    assert calls[-1][1] == {}
    cpu_args = calls[-1][0]
    gpu = fit_bifactor_grm_fipc(*args, **controls, device="gpu")
    assert gpu.gpu_execution_used and not gpu.converged
    assert gpu.gpu_backend == "fixture-Metal" and gpu.gpu_device_name == "fixture-device"
    assert calls[-1][1] == {"device": "gpu"}
    assert calls[-1][0][3:7] == cpu_args[3:7]
    assert calls[-1][0][11:] == cpu_args[11:] == (121, 241, 100, 1e-5, 5, 1e-8, False)
    for index in (0, 1, 2, 7, 8, 9, 10):
        np.testing.assert_array_equal(calls[-1][0][index], cpu_args[index])

    fields.update(gpu_execution_used=False, gpu_backend=None, gpu_device_name=None)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        fit_bifactor_grm_fipc(*args, **controls, device="gpu")
    fields.update(gpu_execution_used=True, gpu_backend="fixture-Metal",
                  gpu_device_name="fixture-device", cpu_fallback_reason="fixture-failed")
    with pytest.raises(RuntimeError, match="CPU fallback"):
        fit_bifactor_grm_fipc(*args, **controls, device="gpu")
    fields.update(gpu_execution_used=1)
    with pytest.raises(ValueError, match="boolean"):
        fit_bifactor_grm_fipc(*args, **controls, device="gpu")
    fields.update(gpu_execution_used=True, gpu_backend=None, gpu_device_name=None)
    with pytest.raises(ValueError, match="identity"):
        fit_bifactor_grm_fipc(*args, **controls, device="gpu")
    fields.update(gpu_execution_used=False, gpu_backend=None, gpu_device_name=None,
                  cpu_fallback_reason="fixture-unavailable")
    auto = fit_bifactor_grm_fipc(*args, **controls, device="auto")
    assert not auto.gpu_execution_used and not auto.converged
    assert auto.cpu_fallback_reason == "fixture-unavailable"
    for original, saved in zip((y, smap, anchor, slope, threshold), before):
        np.testing.assert_array_equal(original, saved)
    count = len(calls)
    with pytest.raises(ValueError, match="device"):
        fit_bifactor_grm_fipc(*args, **controls, device="GPU")
    assert len(calls) == count
