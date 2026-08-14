"""End-to-end tests for the Rust longitudinal state estimator."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.multilevel import (
    LongitudinalStateKind,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
    fit_longitudinal_state,
)


def _occasion(respondent: str, occasion: str, sequence: int, offset: int):
    """Build one sealed occasion with a unique revision identity."""
    occasion_id = f"occasion_{occasion}"
    return build_temporal_occasion(
        respondent_id=respondent,
        occasion_id=occasion_id,
        sequence_index=sequence,
        time_offset_milliseconds=offset,
        occasion_revision_fingerprint=(occasion_id + " revision").encode().hex().ljust(64, "0"),
    )


def test_rust_state_fit_recovers_slopes_and_missing_values() -> None:
    """A two-person synthetic recovery fixture exercises multithreading."""
    occasions = [
        _occasion("respondent_a", "a0", 0, 0),
        _occasion("respondent_a", "a1", 1, 86_400_000),
        _occasion("respondent_a", "a2", 2, 172_800_000),
        _occasion("respondent_b", "b0", 0, 0),
        _occasion("respondent_b", "b1", 1, 86_400_000),
        _occasion("respondent_b", "b2", 2, 172_800_000),
    ]
    design = build_longitudinal_design(
        occasions=occasions,
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )
    result = fit_longitudinal_state(
        design,
        {"occasion_a0": 2.0, "occasion_a1": 3.5, "occasion_a2": 5.0, "occasion_b0": -1.0, "occasion_b1": -3.0},
        worker_count=4,
    )
    np.testing.assert_allclose(result["intercepts"], [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(result["slopes"], [1.5, -2.0], atol=1e-12)
    assert result["observed_count"] == 5
    assert result["engine"] == "rust_cpu_multithreaded"
    assert result["state_kind"] == "random_intercept_slope"
    assert len(result["design_fingerprint"]) == 64
    assert result["occasion_records"][0]["sequence_index"] == 0


def test_rust_state_fit_preserves_discrete_ar_and_irregular_time() -> None:
    """An AR fixture verifies the explicit discrete-step contract."""
    design = build_longitudinal_design(
        occasions=[
            _occasion("respondent_a", "a0", 0, 0),
            _occasion("respondent_a", "a1", 1, 86_400_000),
            _occasion("respondent_a", "a2", 4, 259_200_000),
        ],
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE,
            autoregressive_coefficient=0.5,
        ),
    )
    result = fit_longitudinal_state(
        design,
        {"occasion_a0": 1.0, "occasion_a1": 0.5, "occasion_a2": 0.125},
    )
    assert result["ar_coefficient"] == 0.5
    assert result["transition_count"] == 2
    # The third declared occasion is sequence three steps after the second;
    # the discrete-step AR(1) therefore predicts 0.5**3 * 0.5 = 0.0625.
    np.testing.assert_allclose(result["state"], [1.0, 0.5, 0.0625], atol=1e-12)


def test_state_fit_rejects_invalid_worker_count_and_foreign_design() -> None:
    """The public estimator validates execution controls before Rust dispatch."""
    design = build_longitudinal_design(
        occasions=[_occasion("respondent_a", "guard", 0, 0)],
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )
    with pytest.raises(ValueError, match="worker_count"):
        fit_longitudinal_state(design, {}, worker_count=0)

    class ForeignDesign:
        """Represent an object that was not produced by the package factory."""

    with pytest.raises(ValueError, match="LongitudinalDesign"):
        fit_longitudinal_state(ForeignDesign(), {})  # type: ignore[arg-type]
