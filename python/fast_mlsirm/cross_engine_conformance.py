"""Provider-neutral contracts for independent numerical conformance evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
import unicodedata

SCHEMA_VERSION = "1.0"
MAX_TEXT_LENGTH = 4_096
MAX_COLLECTION_VALUES = 128

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


class ConformanceLayer(str, Enum):
    """Independent comparison layers required by the conformance protocol."""

    FIXED_PARAMETER_EQUATION = "fixed_parameter_equation"
    FITTED_RESULT = "fitted_result"
    NEUTRAL_TRUTH_ADVERSARIAL = "neutral_truth_adversarial"


class ConformanceCoverageStatus(str, Enum):
    """Coverage states for one advertised numerical capability."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NO_INDEPENDENT_ENGINE = "no_independent_engine"
    NOT_COMPARABLE = "not_comparable"
    PLANNED = "planned"


class ConformanceExecutionStatus(str, Enum):
    """Execution/verdict states that must remain semantically distinct."""

    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_EXECUTED = "not_executed"
    NOT_APPLICABLE = "not_applicable"


def _text(value: object, name: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Normalize bounded exact built-in text without caller callback dispatch."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _identifier(value: object, name: str) -> str:
    """Normalize a two-or-more-token lower-snake-case public identifier."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must use two-or-more-token lower snake_case")
    return normalized


def _fingerprint(value: object, name: str) -> str:
    """Validate a lowercase SHA-256 hexadecimal identity."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _optional_fingerprint(value: object | None, name: str) -> str | None:
    """Validate an optional lowercase SHA-256 hexadecimal identity."""
    return None if value is None else _fingerprint(value, name)


def _git_sha(value: object, name: str) -> str:
    """Validate one immutable full Git SHA-1 or SHA-256 identity."""
    normalized = _text(value, name, maximum=64)
    if _GIT_SHA_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{name} must be a full lowercase 40- or 64-character Git SHA"
        )
    return normalized


def _semantic_version(value: object, name: str) -> str:
    """Validate a canonical numeric semantic version."""
    normalized = _text(value, name, maximum=64)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a canonical semantic version")
    return normalized


def _schema_version(value: object) -> str:
    """Accept only the conformance schema implemented by this package slice."""
    normalized = _text(value, "schema_version", maximum=16)
    if normalized != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
    return normalized


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    """Admit an exact enum member or an exact built-in serialized value."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be a supported {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


def _canonical_json(payload: object) -> str:
    """Serialize source-free conformance content deterministically."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: object) -> str:
    """Return SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonEngine:
    """Versioned identity of one isolated independent comparison engine."""

    engine_id: str
    engine_version: str
    source_reference: str
    license_classification: str

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned engine record."""
        if type(self) is not ComparisonEngine:
            raise ValueError("ComparisonEngine must be an exact package record")
        object.__setattr__(self, "engine_id", _identifier(self.engine_id, "engine_id"))
        object.__setattr__(
            self,
            "engine_version",
            _text(self.engine_version, "engine_version", maximum=128),
        )
        object.__setattr__(
            self,
            "source_reference",
            _text(self.source_reference, "source_reference", maximum=512),
        )
        object.__setattr__(
            self,
            "license_classification",
            _identifier(self.license_classification, "license_classification"),
        )

    def to_manifest(self) -> dict[str, str]:
        """Return the JSON-compatible external-engine identity."""
        return {
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "license_classification": self.license_classification,
            "source_reference": self.source_reference,
        }


def _engine(value: object, name: str = "engine") -> ComparisonEngine:
    """Revalidate one exact package-owned comparison-engine record."""
    if type(value) is not ComparisonEngine:
        raise ValueError(f"{name} must be a ComparisonEngine")
    return ComparisonEngine(
        engine_id=value.engine_id,
        engine_version=value.engine_version,
        source_reference=value.source_reference,
        license_classification=value.license_classification,
    )


@dataclass(frozen=True, slots=True)
class ConformanceEvidence:
    """Immutable provenance for one independent conformance execution or plan."""

    evidence_id: str
    engine: ComparisonEngine
    layer: ConformanceLayer
    execution_status: ConformanceExecutionStatus
    parameter_mapping_version: str
    parameter_mapping_sha256: str
    fixture_sha256: str
    environment_sha256: str
    artifact_sha256: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned evidence record."""
        if type(self) is not ConformanceEvidence:
            raise ValueError("ConformanceEvidence must be an exact package record")
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "engine", _engine(self.engine))
        object.__setattr__(
            self,
            "layer",
            _enum_value(self.layer, ConformanceLayer, "layer"),
        )
        object.__setattr__(
            self,
            "execution_status",
            _enum_value(
                self.execution_status,
                ConformanceExecutionStatus,
                "execution_status",
            ),
        )
        object.__setattr__(
            self,
            "parameter_mapping_version",
            _semantic_version(
                self.parameter_mapping_version,
                "parameter_mapping_version",
            ),
        )
        for field_name in (
            "parameter_mapping_sha256",
            "fixture_sha256",
            "environment_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "artifact_sha256",
            _optional_fingerprint(self.artifact_sha256, "artifact_sha256"),
        )
        if self.limitation is not None:
            object.__setattr__(
                self,
                "limitation",
                _text(self.limitation, "limitation"),
            )

        executed = self.execution_status in {
            ConformanceExecutionStatus.PASSED,
            ConformanceExecutionStatus.FAILED,
            ConformanceExecutionStatus.INDETERMINATE,
        }
        if executed and self.artifact_sha256 is None:
            raise ValueError("artifact_sha256 is required for an executed status")
        if not executed and self.artifact_sha256 is not None:
            raise ValueError("artifact_sha256 must be omitted for a nonexecuted status")

    def to_manifest(self) -> dict[str, object]:
        """Return the JSON-compatible evidence projection."""
        return {
            "artifact_sha256": self.artifact_sha256,
            "engine": self.engine.to_manifest(),
            "environment_sha256": self.environment_sha256,
            "evidence_id": self.evidence_id,
            "execution_status": self.execution_status.value,
            "fixture_sha256": self.fixture_sha256,
            "layer": self.layer.value,
            "limitation": self.limitation,
            "parameter_mapping_sha256": self.parameter_mapping_sha256,
            "parameter_mapping_version": self.parameter_mapping_version,
        }


def _evidence_values(values: object) -> tuple[ConformanceEvidence, ...]:
    """Normalize bounded exact package-owned evidence records."""
    if type(values) not in {tuple, list}:
        raise ValueError("evidence must be a list or tuple")
    if len(values) > MAX_COLLECTION_VALUES:
        raise ValueError(
            f"evidence must contain at most {MAX_COLLECTION_VALUES} values"
        )
    normalized: list[ConformanceEvidence] = []
    for index, value in enumerate(values):
        if type(value) is not ConformanceEvidence:
            raise ValueError(f"evidence[{index}] must be a ConformanceEvidence")
        normalized.append(
            ConformanceEvidence(
                evidence_id=value.evidence_id,
                engine=value.engine,
                layer=value.layer,
                execution_status=value.execution_status,
                parameter_mapping_version=value.parameter_mapping_version,
                parameter_mapping_sha256=value.parameter_mapping_sha256,
                fixture_sha256=value.fixture_sha256,
                environment_sha256=value.environment_sha256,
                artifact_sha256=value.artifact_sha256,
                limitation=value.limitation,
            )
        )
    if len({row.evidence_id for row in normalized}) != len(normalized):
        raise ValueError("evidence_id values must be unique")
    return tuple(sorted(normalized, key=lambda row: row.evidence_id))


@dataclass(frozen=True, slots=True)
class ConformanceCapability:
    """One public numerical estimand and its independent comparison coverage."""

    capability_id: str
    public_entrypoint: str
    estimand: str
    likelihood_family: str
    parameterization: str
    identification: str
    comparison_scope: str
    coverage_status: ConformanceCoverageStatus
    evidence: tuple[ConformanceEvidence, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned capability record."""
        if type(self) is not ConformanceCapability:
            raise ValueError("ConformanceCapability must be an exact package record")
        object.__setattr__(
            self,
            "capability_id",
            _identifier(self.capability_id, "capability_id"),
        )
        for field_name in (
            "public_entrypoint",
            "estimand",
            "likelihood_family",
            "parameterization",
            "identification",
            "comparison_scope",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "coverage_status",
            _enum_value(
                self.coverage_status,
                ConformanceCoverageStatus,
                "coverage_status",
            ),
        )
        evidence = _evidence_values(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if self.coverage_status in {
            ConformanceCoverageStatus.COVERED,
            ConformanceCoverageStatus.PARTIALLY_COVERED,
        } and not evidence:
            raise ValueError("covered capability requires evidence")
        if self.coverage_status in {
            ConformanceCoverageStatus.COVERED,
            ConformanceCoverageStatus.PARTIALLY_COVERED,
        } and not any(
            row.execution_status
            in {
                ConformanceExecutionStatus.PASSED,
                ConformanceExecutionStatus.FAILED,
                ConformanceExecutionStatus.INDETERMINATE,
            }
            for row in evidence
        ):
            raise ValueError("covered capability requires executed evidence")
        if self.coverage_status in {
            ConformanceCoverageStatus.NO_INDEPENDENT_ENGINE,
            ConformanceCoverageStatus.NOT_COMPARABLE,
        } and evidence:
            raise ValueError(
                f"{self.coverage_status.value} must not contain comparison evidence"
            )
        if self.coverage_status is ConformanceCoverageStatus.PLANNED and any(
            row.execution_status
            not in {
                ConformanceExecutionStatus.NOT_EXECUTED,
                ConformanceExecutionStatus.NOT_APPLICABLE,
            }
            for row in evidence
        ):
            raise ValueError("planned coverage may contain only nonexecuted evidence")

    def to_manifest(self) -> dict[str, object]:
        """Return the JSON-compatible capability projection."""
        return {
            "capability_id": self.capability_id,
            "comparison_scope": self.comparison_scope,
            "coverage_status": self.coverage_status.value,
            "estimand": self.estimand,
            "evidence": [row.to_manifest() for row in self.evidence],
            "identification": self.identification,
            "likelihood_family": self.likelihood_family,
            "parameterization": self.parameterization,
            "public_entrypoint": self.public_entrypoint,
            "schema_version": self.schema_version,
        }


def _capability_values(values: object) -> tuple[ConformanceCapability, ...]:
    """Normalize a bounded non-empty set of exact package-owned capabilities."""
    if type(values) not in {tuple, list}:
        raise ValueError("capabilities must be a list or tuple")
    if not 1 <= len(values) <= MAX_COLLECTION_VALUES:
        raise ValueError(
            f"capabilities must contain between 1 and {MAX_COLLECTION_VALUES} values"
        )
    normalized: list[ConformanceCapability] = []
    for index, value in enumerate(values):
        if type(value) is not ConformanceCapability:
            raise ValueError(f"capabilities[{index}] must be a ConformanceCapability")
        normalized.append(
            ConformanceCapability(
                capability_id=value.capability_id,
                public_entrypoint=value.public_entrypoint,
                estimand=value.estimand,
                likelihood_family=value.likelihood_family,
                parameterization=value.parameterization,
                identification=value.identification,
                comparison_scope=value.comparison_scope,
                coverage_status=value.coverage_status,
                evidence=value.evidence,
                schema_version=value.schema_version,
            )
        )
    if len({row.capability_id for row in normalized}) != len(normalized):
        raise ValueError("capability_id values must be unique")
    return tuple(sorted(normalized, key=lambda row: row.capability_id))


@dataclass(frozen=True, slots=True)
class ConformanceInventory:
    """Content-addressed inventory of independent conformance coverage."""

    package_version: str
    source_commit: str
    capabilities: tuple[ConformanceCapability, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned inventory record."""
        if type(self) is not ConformanceInventory:
            raise ValueError("ConformanceInventory must be an exact package record")
        object.__setattr__(
            self,
            "package_version",
            _semantic_version(self.package_version, "package_version"),
        )
        object.__setattr__(
            self,
            "source_commit",
            _git_sha(self.source_commit, "source_commit"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _capability_values(self.capabilities),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    def _manifest_without_fingerprint(self) -> dict[str, object]:
        """Return normalized content used to derive immutable inventory identity."""
        return {
            "capabilities": [row.to_manifest() for row in self.capabilities],
            "package_version": self.package_version,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
        }

    @property
    def inventory_fingerprint(self) -> str:
        """Return SHA-256 identity of the normalized source-free inventory."""
        return _sha256(self._manifest_without_fingerprint())

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible source-free inventory."""
        payload = self._manifest_without_fingerprint()
        payload["inventory_fingerprint"] = self.inventory_fingerprint
        return payload


__all__ = [
    "ComparisonEngine",
    "ConformanceCapability",
    "ConformanceCoverageStatus",
    "ConformanceEvidence",
    "ConformanceExecutionStatus",
    "ConformanceInventory",
    "ConformanceLayer",
]
