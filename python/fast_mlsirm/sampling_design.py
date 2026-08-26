"""Versioned Rust-backed finite-population proportion sampling designs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from . import _core

SAMPLING_DESIGN_SCHEMA_VERSION: str = _core.SAMPLING_DESIGN_SCHEMA_VERSION
_MAX_EXACT_F64_INTEGER = 1 << 53
_MAX_SAMPLING_STRATA = 100_000


@dataclass(frozen=True)
class SamplingStratum:
    """One disjoint population stratum with prior- or pilot-derived prevalence."""

    population_size: int
    expected_proportion: float


@dataclass(frozen=True)
class ProportionSamplingDesign:
    """Auditable finite-population sample size, FPC, and stratum allocation."""

    schema_version: str
    source_identity: str
    source_sha256: str
    algorithm_version: str
    population_size: int
    expected_proportion: float
    confidence_level: float
    critical_value: float
    margin_of_error: float
    uncorrected_sample_size: float
    sample_size: int
    finite_population_correction: float
    allocation_method: str
    strata: tuple[SamplingStratum, ...]
    stratum_sample_sizes: tuple[int, ...]
    input_sha256: str
    output_sha256: str
    artifact_sha256: str


def _exact_positive_integer(name: str, value: object) -> int:
    """Admit one exact positive built-in integer inside the Rust f64 identity domain."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    if value > _MAX_EXACT_F64_INTEGER:
        raise ValueError(f"{name} must be between 1 and 2^53")
    return value


def _exact_probability(name: str, value: object) -> float:
    """Admit one exact built-in finite probability strictly inside the unit interval."""
    if type(value) is int:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    if type(value) is not float:
        raise ValueError(f"{name} must be a built-in real number")
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return value


def finite_population_proportion_design(
    population_size: int,
    confidence_level: float,
    margin_of_error: float,
    strata: Sequence[SamplingStratum],
    *,
    allocation_method: Literal["proportional", "neyman"],
) -> ProportionSamplingDesign:
    """Return the NIST/ABS sampling design computed entirely by the Rust core.

    Each expected proportion must come from declared prior or pilot evidence.
    The API deliberately has no implicit ``0.5`` or allocation-weight default.
    """
    normalized_population_size = _exact_positive_integer(
        "population_size", population_size
    )
    normalized_confidence_level = _exact_probability(
        "confidence_level", confidence_level
    )
    normalized_margin_of_error = _exact_probability(
        "margin_of_error", margin_of_error
    )
    if type(strata) not in (list, tuple) or not strata:
        raise ValueError("strata must be a non-empty built-in list or tuple")
    if len(strata) > _MAX_SAMPLING_STRATA:
        raise ValueError(
            f"strata must contain between 1 and {_MAX_SAMPLING_STRATA} entries"
        )
    if type(allocation_method) is not str or allocation_method not in (
        "proportional",
        "neyman",
    ):
        raise ValueError("allocation_method must be proportional or neyman")
    population_sizes: list[int] = []
    expected_proportions: list[float] = []
    for index, stratum in enumerate(strata):
        if type(stratum) is not SamplingStratum:
            raise ValueError(f"strata[{index}] must be a SamplingStratum")
        population_sizes.append(
            _exact_positive_integer(
                f"strata[{index}].population_size", stratum.population_size
            )
        )
        expected_proportions.append(
            _exact_probability(
                f"strata[{index}].expected_proportion",
                stratum.expected_proportion,
            )
        )

    result = _core.finite_population_proportion_design(
        normalized_population_size,
        normalized_confidence_level,
        normalized_margin_of_error,
        population_sizes,
        expected_proportions,
        allocation_method,
    )
    schema_version = result["schema_version"]
    if schema_version != SAMPLING_DESIGN_SCHEMA_VERSION:
        raise ValueError(
            "unsupported sampling-design schema version: "
            f"{schema_version!r}"
        )
    return ProportionSamplingDesign(
        schema_version=schema_version,
        source_identity=result["source_identity"],
        source_sha256=result["source_sha256"],
        algorithm_version=result["algorithm_version"],
        population_size=result["population_size"],
        expected_proportion=result["expected_proportion"],
        confidence_level=result["confidence_level"],
        critical_value=result["critical_value"],
        margin_of_error=result["margin_of_error"],
        uncorrected_sample_size=result["uncorrected_sample_size"],
        sample_size=result["sample_size"],
        finite_population_correction=result["finite_population_correction"],
        allocation_method=result["allocation_method"],
        strata=tuple(
            SamplingStratum(population, proportion)
            for population, proportion in zip(
                result["stratum_population_sizes"],
                result["stratum_expected_proportions"],
                strict=True,
            )
        ),
        stratum_sample_sizes=tuple(result["stratum_sample_sizes"]),
        input_sha256=result["input_sha256"],
        output_sha256=result["output_sha256"],
        artifact_sha256=result["artifact_sha256"],
    )
