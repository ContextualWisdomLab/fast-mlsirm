"""Tests for the Rust-owned finite-population sampling-design contract."""

from __future__ import annotations

import pytest
from fast_mlsirm import (
    SAMPLING_DESIGN_SCHEMA_VERSION,
    SamplingStratum,
    finite_population_proportion_design,
)


def test_finite_population_design_matches_rust_reference() -> None:
    """The public API returns the versioned Rust result without Python arithmetic."""
    design = finite_population_proportion_design(
        1_000,
        0.95,
        0.05,
        [SamplingStratum(1_000, 0.5)],
        allocation_method="proportional",
    )

    assert design.schema_version == SAMPLING_DESIGN_SCHEMA_VERSION
    assert design.sample_size == 278
    assert design.stratum_sample_sizes == (278,)
    assert design.finite_population_correction == pytest.approx(
        (722.0 / 999.0) ** 0.5
    )


def test_neyman_allocation_uses_caller_supplied_stratum_variability() -> None:
    """Neyman allocation differs only through declared stratum evidence."""
    design = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(60, 0.1), SamplingStratum(40, 0.5)],
        allocation_method="neyman",
    )

    assert design.sample_size == 43
    assert design.stratum_sample_sizes == (20, 23)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": True},
        {"confidence_level": 1.0},
        {"margin_of_error": float("nan")},
        {"strata": [SamplingStratum(100, 0.0)]},
        {"allocation_method": "equal"},
    ],
)
def test_sampling_design_rejects_implicit_or_invalid_evidence(
    kwargs: dict[str, object],
) -> None:
    """The Python seam rejects coercion and unsupported allocation rules."""
    arguments: dict[str, object] = {
        "population_size": 100,
        "confidence_level": 0.95,
        "margin_of_error": 0.1,
        "strata": [SamplingStratum(100, 0.5)],
        "allocation_method": "proportional",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        finite_population_proportion_design(**arguments)  # type: ignore[arg-type]
