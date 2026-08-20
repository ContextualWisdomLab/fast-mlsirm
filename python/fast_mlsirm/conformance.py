"""Source-free contracts for reproducible cross-engine conformance evidence.

The contract records what was compared and how it can be reproduced.  It does
not import, execute, or depend on an external statistical engine; numerical
equations, estimators, and comparison harnesses remain separate bounded
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re

SCHEMA_VERSION = "1.0"
MAX_COLLECTION_VALUES = 64
MAX_TEXT_LENGTH = 4_096

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ConformanceStatus(str, Enum):
    """Explicit coverage states that cannot be collapsed into success."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NO_INDEPENDENT_ENGINE = "no_independent_engine"
    NOT_COMPARABLE = "not_comparable"
    PLANNED = "planned"


class ConformanceLayer(str, Enum):
    """Comparison layer or output family represented by one capability row."""

    EQUATION = "equation"
    FITTED_RESULT = "fitted_result"
    SCORING = "scoring"
    FIT_STATISTIC = "fit_statistic"
    LINKING = "linking"
    SERIALIZATION = "serialization"


def _text(value: object, name: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Normalize bounded exact built-in text without caller callbacks."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _identifier(value: object, name: str) -> str:
    """Normalize one lower-snake-case identifier with at least two tokens."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must use two-or-more-token lower snake_case")
    return normalized


def _fingerprint(value: object, name: str) -> str:
    """Validate one lowercase SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _optional_fingerprint(value: object | None, name: str) -> str | None:
    """Validate an optional lowercase SHA-256 fingerprint."""
    return None if value is None else _fingerprint(value, name)


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    """Admit an exact enum member or exact built-in serialized value."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be a supported {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


def _record_tuple(values: object, record_type: type, name: str) -> tuple:
    """Normalize a bounded exact list or tuple of package-owned records."""
    if type(values) not in {tuple, list}:
        raise ValueError(f"{name} must be a list or tuple")
    if len(values) > MAX_COLLECTION_VALUES:
        raise ValueError(
            f"{name} must contain at most {MAX_COLLECTION_VALUES} values"
        )
    for index, value in enumerate(values):
        if type(value) is not record_type:
            raise ValueError(f"{name}[{index}] must be a {record_type.__name__}")
    return tuple(values)


def _finite_nonnegative(value: object, name: str) -> float:
    """Normalize one finite non-negative built-in numeric control."""
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _canonical_json(payload: object) -> str:
    """Serialize normalized manifest content deterministically."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: object) -> str:
    """Return SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EngineReference:
    """Identify one optional, independently versioned comparison engine."""

    engine_id: str
    version: str
    source_url: str
    license_classification: str

    def __post_init__(self) -> None:
        """Normalize engine identity without importing or executing the engine."""
        if type(self) is not EngineReference:
            raise ValueError("EngineReference must be an exact package record")
        object.__setattr__(self, "engine_id", _identifier(self.engine_id, "engine_id"))
        for field_name in ("version", "source_url", "license_classification"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))

    def to_manifest(self) -> dict[str, str]:
        """Return the source-free engine identity projection."""
        return {
            "engine_id": self.engine_id,
            "license_classification": self.license_classification,
            "source_url": self.source_url,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ConformanceTolerance:
    """Record a preregistered tolerance for one named estimand."""

    estimand_id: str
    absolute: float
    relative: float
    rationale: str

    def __post_init__(self) -> None:
        """Normalize a non-negative tolerance and its scientific rationale."""
        if type(self) is not ConformanceTolerance:
            raise ValueError("ConformanceTolerance must be an exact package record")
        object.__setattr__(self, "estimand_id", _identifier(self.estimand_id, "estimand_id"))
        object.__setattr__(self, "absolute", _finite_nonnegative(self.absolute, "absolute"))
        object.__setattr__(self, "relative", _finite_nonnegative(self.relative, "relative"))
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("absolute or relative tolerance must be positive")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))

    def to_manifest(self) -> dict[str, object]:
        """Return the JSON-compatible tolerance projection."""
        return {
            "absolute": self.absolute,
            "estimand_id": self.estimand_id,
            "rationale": self.rationale,
            "relative": self.relative,
        }


@dataclass(frozen=True, slots=True)
class ConformanceCapability:
    """Describe one public capability and its independent comparison scope."""

    capability_id: str
    public_entry_point: str
    estimand: str
    layer: ConformanceLayer
    status: ConformanceStatus
    engines: tuple[EngineReference, ...]
    mapping_fingerprint: str | None
    tolerances: tuple[ConformanceTolerance, ...]

    def __post_init__(self) -> None:
        """Normalize capability metadata and reject false coverage claims."""
        if type(self) is not ConformanceCapability:
            raise ValueError("ConformanceCapability must be an exact package record")
        object.__setattr__(self, "capability_id", _identifier(self.capability_id, "capability_id"))
        object.__setattr__(self, "public_entry_point", _text(self.public_entry_point, "public_entry_point"))
        object.__setattr__(self, "estimand", _text(self.estimand, "estimand"))
        object.__setattr__(self, "layer", _enum_value(self.layer, ConformanceLayer, "layer"))
        status = _enum_value(self.status, ConformanceStatus, "status")
        object.__setattr__(self, "status", status)
        engines = _record_tuple(self.engines, EngineReference, "engines")
        if len({engine.engine_id for engine in engines}) != len(engines):
            raise ValueError("engine_id values must be unique")
        object.__setattr__(self, "engines", engines)
        object.__setattr__(
            self,
            "mapping_fingerprint",
            _optional_fingerprint(self.mapping_fingerprint, "mapping_fingerprint"),
        )
        tolerances = _record_tuple(
            self.tolerances,
            ConformanceTolerance,
            "tolerances",
        )
        if len({item.estimand_id for item in tolerances}) != len(tolerances):
            raise ValueError("tolerance estimand_id values must be unique")
        object.__setattr__(self, "tolerances", tolerances)
        if status in {
            ConformanceStatus.COVERED,
            ConformanceStatus.PARTIALLY_COVERED,
        } and not engines:
            raise ValueError("covered capabilities require an independent engine")
        if status is ConformanceStatus.COVERED and self.mapping_fingerprint is None:
            raise ValueError("covered capabilities require a mapping_fingerprint")

    def to_manifest(self) -> dict[str, object]:
        """Return one deterministic capability row without engine output."""
        return {
            "capability_id": self.capability_id,
            "engines": [engine.to_manifest() for engine in self.engines],
            "estimand": self.estimand,
            "layer": self.layer.value,
            "mapping_fingerprint": self.mapping_fingerprint,
            "public_entry_point": self.public_entry_point,
            "status": self.status.value,
            "tolerances": [item.to_manifest() for item in self.tolerances],
        }


@dataclass(frozen=True, slots=True)
class ConformanceProvenance:
    """Bind a conformance profile to immutable code, fixture, and environment IDs."""

    protected_main_sha: str
    harness_sha: str
    environment_fingerprint: str
    fixture_fingerprint: str
    mapping_schema: str
    mapping_fingerprint: str
    rng_algorithm: str
    rng_seeds: tuple[int, ...]
    raw_output_fingerprint: str | None
    normalized_output_fingerprint: str | None
    license_classification: str

    def __post_init__(self) -> None:
        """Normalize provenance and retain exact reproducibility controls."""
        if type(self) is not ConformanceProvenance:
            raise ValueError("ConformanceProvenance must be an exact package record")
        for field_name in (
            "protected_main_sha",
            "harness_sha",
            "environment_fingerprint",
            "fixture_fingerprint",
            "mapping_fingerprint",
        ):
            object.__setattr__(self, field_name, _fingerprint(getattr(self, field_name), field_name))
        for field_name in ("mapping_schema", "rng_algorithm", "license_classification"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        if type(self.rng_seeds) not in {tuple, list}:
            raise ValueError("rng_seeds must be a list or tuple")
        if len(self.rng_seeds) > MAX_COLLECTION_VALUES:
            raise ValueError(f"rng_seeds must contain at most {MAX_COLLECTION_VALUES} values")
        seeds = tuple(self.rng_seeds)
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("rng_seeds must contain non-negative built-in integers")
        object.__setattr__(self, "rng_seeds", seeds)
        for field_name in ("raw_output_fingerprint", "normalized_output_fingerprint"):
            object.__setattr__(self, field_name, _optional_fingerprint(getattr(self, field_name), field_name))

    def to_manifest(self) -> dict[str, object]:
        """Return source-free reproducibility provenance."""
        return {
            "environment_fingerprint": self.environment_fingerprint,
            "fixture_fingerprint": self.fixture_fingerprint,
            "harness_sha": self.harness_sha,
            "license_classification": self.license_classification,
            "mapping_fingerprint": self.mapping_fingerprint,
            "mapping_schema": self.mapping_schema,
            "normalized_output_fingerprint": self.normalized_output_fingerprint,
            "protected_main_sha": self.protected_main_sha,
            "raw_output_fingerprint": self.raw_output_fingerprint,
            "rng_algorithm": self.rng_algorithm,
            "rng_seeds": list(self.rng_seeds),
        }


@dataclass(frozen=True, slots=True)
class ConformanceManifest:
    """Content-addressed inventory for one isolated conformance run or plan."""

    conformance_manifest_id: str
    protected_main_sha: str
    capabilities: tuple[ConformanceCapability, ...]
    provenance: ConformanceProvenance
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize capabilities and ensure the manifest binds one main revision."""
        if type(self) is not ConformanceManifest:
            raise ValueError("ConformanceManifest must be an exact package record")
        object.__setattr__(
            self,
            "conformance_manifest_id",
            _identifier(self.conformance_manifest_id, "conformance_manifest_id"),
        )
        main_sha = _fingerprint(self.protected_main_sha, "protected_main_sha")
        object.__setattr__(self, "protected_main_sha", main_sha)
        if type(self.provenance) is not ConformanceProvenance:
            raise ValueError("provenance must be a ConformanceProvenance")
        if self.provenance.protected_main_sha != main_sha:
            raise ValueError("provenance protected_main_sha must match the manifest")
        capabilities = _record_tuple(
            self.capabilities,
            ConformanceCapability,
            "capabilities",
        )
        if not capabilities:
            raise ValueError("capabilities must not be empty")
        if len({item.capability_id for item in capabilities}) != len(capabilities):
            raise ValueError("capability_id values must be unique")
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(capabilities, key=lambda item: item.capability_id)),
        )
        version = _text(self.schema_version, "schema_version", maximum=16)
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
        object.__setattr__(self, "schema_version", version)

    def _content(self) -> dict[str, object]:
        """Return canonical manifest content without the derived fingerprint."""
        return {
            "capabilities": [item.to_manifest() for item in self.capabilities],
            "conformance_manifest_id": self.conformance_manifest_id,
            "protected_main_sha": self.protected_main_sha,
            "provenance": self.provenance.to_manifest(),
            "schema_version": self.schema_version,
        }

    @property
    def manifest_fingerprint(self) -> str:
        """Return the immutable SHA-256 identity of normalized manifest content."""
        return _sha256(self._content())

    def to_manifest(self) -> dict[str, object]:
        """Return deterministic JSON-compatible conformance metadata."""
        payload = self._content()
        payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload


__all__ = [
    "ConformanceCapability",
    "ConformanceLayer",
    "ConformanceManifest",
    "ConformanceProvenance",
    "ConformanceStatus",
    "ConformanceTolerance",
    "EngineReference",
]
