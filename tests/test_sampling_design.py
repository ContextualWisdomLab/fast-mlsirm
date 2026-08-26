"""Tests for the Rust-owned finite-population sampling-design contract."""

from __future__ import annotations

import fast_mlsirm.sampling_design as sampling_design_module
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
    assert design.source_identity == "fast-mlsirm.mlsirm-core.sampling-design"
    assert design.algorithm_version == "1.0.0"
    assert design.strata == (SamplingStratum(1_000, 0.5),)
    assert design.sample_size == 278
    assert design.stratum_sample_sizes == (278,)
    assert design.finite_population_correction == pytest.approx(
        (722.0 / 999.0) ** 0.5
    )
    assert all(
        len(identity) == 64
        for identity in (
            design.source_sha256,
            design.input_sha256,
            design.output_sha256,
            design.artifact_sha256,
        )
    )


def test_sampling_artifact_replay_has_stable_content_identity() -> None:
    """Exact canonical inputs reproduce every Rust-owned artifact identity."""
    first = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(60, 0.1), SamplingStratum(40, 0.5)],
        allocation_method="neyman",
    )
    replay = finite_population_proportion_design(
        first.population_size,
        first.confidence_level,
        first.margin_of_error,
        list(first.strata),
        allocation_method=first.allocation_method,  # type: ignore[arg-type]
    )

    assert replay == first
    assert first.input_sha256 == (
        "c4c9a1dd09b3bd59fc14e4ac2824cd8a2169b3c94f5a5080bacfd2e5a964d786"
    )
    assert first.output_sha256 == (
        "0729aeadbb08fe0cf1c0231841453516557dd85db9146f1be5ce87e35e225cda"
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


def test_sampling_resource_bounds_fail_before_rust_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust resource/domain bounds are replayed before caller evidence or core work."""
    core_calls = 0

    def forbidden_core(*args: object, **kwargs: object) -> object:
        nonlocal core_calls
        core_calls += 1
        raise AssertionError("Rust dispatch must not run for invalid resource controls")

    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_proportion_design",
        forbidden_core,
    )

    with pytest.raises(ValueError, match=r"population_size must be between 1 and 2\^53"):
        finite_population_proportion_design(
            (1 << 53) + 1,
            0.95,
            0.1,
            [SamplingStratum((1 << 53) + 1, 0.5)],
            allocation_method="proportional",
        )

    with pytest.raises(
        ValueError,
        match=r"strata\[0\]\.population_size must be between 1 and 2\^53",
    ):
        finite_population_proportion_design(
            100,
            0.95,
            0.1,
            [SamplingStratum((1 << 53) + 1, 0.5)],
            allocation_method="proportional",
        )

    oversized_strata = [SamplingStratum(1, 0.5)] * 100_001
    with pytest.raises(ValueError, match="strata must contain between 1 and 100000 entries"):
        finite_population_proportion_design(
            100_001,
            0.95,
            0.1,
            oversized_strata,
            allocation_method="proportional",
        )

    assert core_calls == 0


def test_sampling_schema_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A foreign Rust result schema is rejected before result marshalling."""
    assert (
        sampling_design_module.SAMPLING_DESIGN_SCHEMA_VERSION
        == sampling_design_module._core.SAMPLING_DESIGN_SCHEMA_VERSION
    )

    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_proportion_design",
        lambda *args, **kwargs: {"schema_version": "fast-mlsirm.sampling-design.v2"},
    )

    with pytest.raises(ValueError, match="unsupported sampling-design schema version"):
        finite_population_proportion_design(
            100,
            0.95,
            0.1,
            [SamplingStratum(100, 0.5)],
            allocation_method="proportional",
        )


def test_giant_integer_probability_fails_with_package_value_error() -> None:
    """Strict probability controls do not overflow through unnecessary float coercion."""
    with pytest.raises(ValueError, match="confidence_level must be finite"):
        finite_population_proportion_design(
            100,
            10**400,
            0.1,
            [SamplingStratum(100, 0.5)],
            allocation_method="proportional",
        )
