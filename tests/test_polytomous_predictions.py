"""Public Rust-owned GRM/GPCM prediction contract."""

import numpy as np
import pytest

import fast_mlsirm._polytomous_prediction_admission as prediction_admission
import fast_mlsirm.polytomous as polytomous_module
from fast_mlsirm import (
    PolytomousFit,
    polytomous_category_probabilities,
    polytomous_expected_response,
)


class _ArrayProvider:
    def __array__(self, *args, **kwargs):
        raise AssertionError("caller __array__ callback executed")


@pytest.mark.parametrize(
    ("model", "cat_params"),
    [
        ("grm", np.array([[1.0, -1.0], [0.5, -0.5]])),
        ("gpcm", np.array([[0.2, -0.4], [-0.1, 0.3]])),
    ],
)
def test_public_polytomous_predictions_are_normalized_and_consistent(model, cat_params):
    fit = PolytomousFit(model, np.array([1.0, 0.8]), cat_params, 0.0, 0)
    theta = np.array([-1000.0, 0.0, 1000.0])

    probabilities = polytomous_category_probabilities(fit, theta)
    expected = polytomous_expected_response(fit, theta)

    assert probabilities.shape == (3, 2, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    assert np.allclose(expected, probabilities @ np.arange(3, dtype=np.float64))


def test_public_polytomous_predictions_reject_invalid_grm_thresholds():
    fit = PolytomousFit("grm", np.array([1.0]), np.array([[-1.0, 1.0]]), 0.0, 0)
    with pytest.raises(ValueError, match="strictly decreasing"):
        polytomous_category_probabilities(fit, np.array([0.0]))


def test_public_polytomous_predictions_bound_output_before_native_dispatch(monkeypatch):
    """Reject oversized logical prediction grids before Rust/output allocation."""

    fit = PolytomousFit("grm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)
    theta = np.broadcast_to(np.array([0.0]), (10_000_001,))

    def unexpected_core_discovery():
        raise AssertionError("oversized prediction grid reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="20,000,000.*prediction"):
        polytomous_category_probabilities(fit, theta)


def test_public_polytomous_predictions_bound_output_before_dense_copy(monkeypatch):
    """Output-cell admission must precede contiguous materialization of broadcast theta."""

    fit = PolytomousFit("grm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)
    theta = np.broadcast_to(np.array([0.0]), (10_000_001,))

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("oversized prediction grid reached contiguous materialization")

    def unexpected_core_discovery():
        raise AssertionError("oversized prediction grid reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "ascontiguousarray", unexpected_copy)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="20,000,000.*prediction"):
        polytomous_category_probabilities(fit, theta)


@pytest.mark.parametrize("field", ["theta", "slope", "cat_params"])
def test_public_polytomous_predictions_reject_array_providers_before_callbacks(
    monkeypatch, field
):
    """Caller array protocols must not run while prediction evidence is admitted."""

    theta = np.array([0.0])
    slope = np.array([1.0])
    cat_params = np.array([[0.0]])
    if field == "theta":
        theta = _ArrayProvider()
    elif field == "slope":
        slope = _ArrayProvider()
    else:
        cat_params = _ArrayProvider()
    fit = PolytomousFit("grm", slope, cat_params, 0.0, 0)

    def unexpected_core_discovery():
        raise AssertionError("untrusted evidence reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="trusted NumPy array or built-in sequence"):
        polytomous_category_probabilities(fit, theta)


def test_public_polytomous_predictions_reject_complex_before_native_dispatch(monkeypatch):
    """Do not silently project complex theta evidence onto the real line."""

    fit = PolytomousFit("gpcm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)

    def unexpected_core_discovery():
        raise AssertionError("complex evidence reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="real-valued"):
        polytomous_expected_response(fit, np.array([0.0 + 1.0j]))
