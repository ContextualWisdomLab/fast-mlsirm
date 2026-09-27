"""Admission and Rust-dispatch checks, independent of numerical Rust tests."""
import importlib.util
from pathlib import Path

import fast_mlsirm
import numpy as np
import pytest


def _wrapper():
    path = Path(__file__).parents[1] / "python/fast_mlsirm/report_transforms.py"
    spec = importlib.util.spec_from_file_location("fast_mlsirm.report_transforms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.orthogonal_grm_report


def test_wrapper_only_marshals_and_preserves_native_report(monkeypatch):
    from fast_mlsirm import _core
    seen = []

    def native(slopes, intercepts, metric, scale):
        seen.append((slopes, intercepts, metric, scale))
        return dict(loadings=[0.1], thresholds=[-0.2], communality=0.01,
                    uniqueness=0.99, approximate=True, input_metric=metric, scale=scale)

    monkeypatch.setattr(_core, "orthogonal_grm_report", native, raising=False)
    result = _wrapper()(np.array([2], dtype=np.int32), np.array([3.]),
                        input_metric="logistic_approximation", scale=1.702)
    assert len(seen) == 1
    assert seen[0][0].dtype == np.float64 and seen[0][0][0] == 2.
    assert seen[0][2:] == ("logistic_approximation", 1.702)
    assert result["loadings"][0] == 0.1 and result["communality"] == 0.01
    assert result["approximate"] is True


def test_admission_rejects_callbacks_before_native_or_array_conversion(monkeypatch):
    from fast_mlsirm import _core

    def forbidden(*args):
        raise AssertionError("native must not run")

    class Callback:
        def __array__(self, *args, **kwargs):
            raise AssertionError("conversion callback must not run")
        def __float__(self):
            raise AssertionError("scalar callback must not run")

    monkeypatch.setattr(_core, "orthogonal_grm_report", forbidden, raising=False)
    wrapper = _wrapper()
    object_array = np.empty(1, dtype=object)
    object_array[0] = Callback()
    for slopes, scale, metric in (
        (Callback(), 1., "normal_ogive"),
        (np.array([1.]), Callback(), "normal_ogive"),
        (np.array([1.]), True, "normal_ogive"),
        (np.array([1.]), 1., Callback()),
        (np.array([1.]), 1., "correlated"),
        (np.array([1j]), 1., "normal_ogive"),
        (np.array([[1.]]), 1., "normal_ogive"),
        (object_array, 1., "normal_ogive"),
    ):
        with pytest.raises(ValueError):
            wrapper(slopes, np.array([0.]), input_metric=metric, scale=scale)


def test_wrapper_propagates_native_rejection(monkeypatch):
    from fast_mlsirm import _core

    def reject(*args):
        raise ValueError("boundary intercepts must be strictly decreasing")

    monkeypatch.setattr(_core, "orthogonal_grm_report", reject, raising=False)
    with pytest.raises(ValueError, match="strictly decreasing"):
        _wrapper()(np.array([1.]), np.array([0., 1.]), input_metric="normal_ogive", scale=1.)
