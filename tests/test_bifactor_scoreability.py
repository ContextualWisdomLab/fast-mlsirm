"""Public Python contracts for Rust-native bifactor scoreability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm._bifactor_core_loader import bifactor_core
from fast_mlsirm.bifactor_scoreability import (
    BifactorScoreabilityResult,
    bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes,
)


def _loadings() -> np.ndarray:
    """Return one standardized strict-bifactor loading matrix."""
    return np.asarray(
        [
            [0.70, 0.40, 0.00],
            [0.70, 0.30, 0.00],
            [0.70, 0.00, 0.50],
            [0.70, 0.00, 0.60],
        ],
        dtype=np.float64,
    )


def _uniquenesses() -> np.ndarray:
    """Return residual variances satisfying the standardized identity."""
    return np.asarray([0.35, 0.42, 0.26, 0.15], dtype=np.float64)


def _logit_slopes() -> np.ndarray:
    """Invert the documented logistic latent-response standardization."""
    loadings = _loadings()
    uniquenesses = _uniquenesses()
    logistic_sd = np.pi / np.sqrt(3.0)
    return loadings * (logistic_sd / np.sqrt(uniquenesses))[:, None]


def _assert_result_matches_raw(result: BifactorScoreabilityResult, raw: dict) -> None:
    """Assert exact field-by-field wrapper parity with the compiled module."""
    assert result.factor_item_counts == tuple(raw["factor_item_counts"])
    assert result.is_strict_bifactor is raw["is_strict_bifactor"]
    assert result.puc == pytest.approx(raw["puc"])
    np.testing.assert_allclose(result.ecv_ss, raw["ecv_ss"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.ecv_sg, raw["ecv_sg"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.ecv_gs, raw["ecv_gs"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.item_ecv, raw["item_ecv"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.omega_total, raw["omega_total"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.omega_hierarchical,
        raw["omega_hierarchical"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.construct_replicability,
        raw["construct_replicability"],
        rtol=0.0,
        atol=0.0,
    )


def test_standardized_loading_wrapper_has_exact_rust_parity():
    """The Python layer only marshals the compiled standardized-loading result."""
    loadings = _loadings()
    uniquenesses = _uniquenesses()
    result = bifactor_scoreability(loadings, uniquenesses, general_factor=0)
    raw = bifactor_core().bifactor_indices(
        loadings,
        uniquenesses,
        0,
        0.0,
    )

    assert isinstance(result, BifactorScoreabilityResult)
    assert result.puc == pytest.approx(2.0 / 3.0)
    _assert_result_matches_raw(result, raw)


def test_logit_slope_wrapper_has_exact_rust_parity():
    """Fitted logistic slopes are standardized and scored only in Rust."""
    slopes = _logit_slopes()
    result = bifactor_scoreability_from_logit_slopes(slopes, general_factor=0)
    raw = bifactor_core().bifactor_indices_from_logit_slopes(slopes, 0, 0.0)

    _assert_result_matches_raw(result, raw)


def test_package_root_exports_the_public_bifactor_api():
    """Buyers can discover the typed diagnostics from the package root."""
    assert fast_mlsirm.BifactorScoreabilityResult is BifactorScoreabilityResult
    assert fast_mlsirm.bifactor_scoreability is bifactor_scoreability
    assert (
        fast_mlsirm.bifactor_scoreability_from_logit_slopes
        is bifactor_scoreability_from_logit_slopes
    )


@pytest.mark.parametrize(
    ("loadings", "uniquenesses", "message"),
    [
        (np.ones(4), _uniquenesses(), "loadings must be a 2-D"),
        (_loadings(), np.ones((2, 2)), "uniquenesses must be a 1-D"),
        (_loadings(), np.ones(3), "uniquenesses length"),
    ],
)
def test_public_wrapper_validates_array_shapes(loadings, uniquenesses, message):
    """Shape mismatches fail before crossing the compiled boundary."""
    with pytest.raises(ValueError, match=message):
        bifactor_scoreability(loadings, uniquenesses)


def test_missing_general_factor_fails_through_rust():
    """The Python wrapper preserves the Rust fail-closed bifactor definition."""
    loadings = _loadings()
    loadings[0, 0] = 0.0
    uniquenesses = 1.0 - np.square(loadings).sum(axis=1)
    with pytest.raises(ValueError, match="general factor"):
        bifactor_scoreability(loadings, uniquenesses, general_factor=0)


def test_loader_caches_the_secondary_extension_module():
    """Repeated calls reuse one initialized `_bifactor_core` module."""
    assert bifactor_core() is bifactor_core()
