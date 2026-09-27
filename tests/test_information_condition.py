"""Condition reports use the same Rust decomposition as second-order checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_condition_wrapper_preserves_native_fields(monkeypatch):
    from fast_mlsirm import _core

    def native(matrix, tol):
        return dict(passed=False, min_eigenvalue=-2., eigenvalues=[-2., 1.],
                    condition_number_2=2., reciprocal_condition_number_2=0.5,
                    condition_matrix="symmetrized_input")

    monkeypatch.setattr(_core, "second_order_test", native)
    path = Path(__file__).parents[1] / "python/fast_mlsirm/inference.py"
    spec = importlib.util.spec_from_file_location("fast_mlsirm.inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.second_order_test(np.eye(2))
    assert report["condition_number_2"] == 2.
    assert report["reciprocal_condition_number_2"] == 0.5
    assert report["condition_matrix"] == "symmetrized_input"
    assert report["passed"] is False


def test_installed_condition_api_scale_rotation_and_singular_cases():
    from fast_mlsirm.inference import second_order_test

    for scale in (1., 1e-300, 1e300):
        report = second_order_test(np.array([[2., 1.], [1., 2.]]) * scale, tol=0.)
        assert report["condition_number_2"] == pytest.approx(3.)
        assert report["reciprocal_condition_number_2"] == pytest.approx(1 / 3)
        assert report["condition_matrix"] == "symmetrized_input"
        assert report["passed"] is True
    report = second_order_test(np.diag([-4., 2.]))
    assert report["condition_number_2"] == 2. and report["passed"] is False
    report = second_order_test(np.diag([0., 2.]))
    assert np.isinf(report["condition_number_2"])
    assert report["reciprocal_condition_number_2"] == 0.
