"""Immutable assessment and scoring-policy contracts.

This module defines the provider-neutral contract that binds exact rubric
fingerprints to scoring, calibration, validation, adjudication, monitoring, and
reporting policies.  It performs no scoring or psychometric arithmetic; later
orchestration layers must delegate numerical work to the existing Rust-backed
estimators.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import json
import math
import operator
import re
from types import MappingProxyType
from typing import Any

from fast_mlsirm.rubric.models import (
    SCHEMA_VERSION,
    RubricSpecification,
    _bounded_values,
    _identifier,
    _schema_version,
    _semantic_version,
    _text,
)

MAX_METADATA_COLLECTION_VALUES = 64
MAX_METADATA_DEPTH = 8
MAX_METADATA_NODES = 1_024

_MAX_ASSESSMENT_CONSTRUCTS = 32
_MAX_ASSESSMENT_RUBRICS = 64
_MAX_POLICY_REFERENCES = 64
_MAX_RATERS_PER_RESPONSE = 64
_MAX_METADATA_KEY_LENGTH = 128
_MAX_METADATA_TEXT_LENGTH = 8_192
_MIN_SIGNED_INTEGER = -(1 << 63)
_MAX_SIGNED_INTEGER = (1 << 63) - 1
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSESSMENT_SPEC_TOKEN = object()


class AssessmentSpecError(ValueError):
    """Structured fail-closed assessment-graph validation error."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store stable error metadata without embedding caller content."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


class AssessmentResponseType(str, Enum):
    """Granularity at which one assessment records scoring observations."""

    CRITERION_LEVEL = "criterion_level"
    HOLISTIC = "holistic"
    MIXED = "mixed"


def _response_type(value: AssessmentResponseType | str) -> AssessmentResponseType:
    """Normalize one supported assessment-response representation."""
    if isinstance(value, AssessmentResponseType):
        return value
    try:
        return AssessmentResponseType(value)
    except (TypeError, ValueError) as exc:
        choices = [member.value for member in AssessmentResponseType]
        raise ValueError(f"response_type must be one of {choices}") from exc


def _strict_boolean(value: Any, name: str) -> bool:
    """Return a real Boolean without accepting integer coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _bounded_positive_integer(value: Any, name: str, maximum: int) -> int:
    """Return a bounded positive integer while rejecting booleans and fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}") from exc
    if not 1 <= normalized <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return int(normalized)


def _fingerprint(value: Any, name: str) -> str:
    """Return a validated lowercase SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character lower hexadecimal digest")
    return normalized


def _sorted_identifiers(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = _MAX_POLICY_REFERENCES,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique descriptive identifiers."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _identifier(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _sorted_fingerprints(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = _MAX_ASSESSMENT_RUBRICS,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique SHA-256 fingerprints."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _fingerprint(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _metadata_key(value: Any, path: str) -> str:
    """Return one bounded printable metadata key without hidden whitespace."""
    if not isinstance(value, str):
        raise ValueError(f"{path} metadata keys must be strings")
    if not value or value != value.strip():
        raise ValueError(f"{path} metadata keys must be non-empty and trimmed")
    if len(value) > _MAX_METADATA_KEY_LENGTH:
        raise ValueError(
            f"{path} metadata keys must contain at most {_MAX_METADATA_KEY_LENGTH} characters"
        )
    if not value.isprintable():
        raise ValueError(f"{path} metadata keys must not contain control characters")
    return value


def _freeze_json_value(
    value: Any,
    path: str,
    *,
    depth: int,
    node_count: list[int],
) -> Any:
    """Validate and deeply freeze one bounded JSON-compatible value."""
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(
            f"{path} exceeds the maximum metadata depth of {MAX_METADATA_DEPTH}"
        )
    node_count[0] += 1
    if node_count[0] > MAX_METADATA_NODES:
        raise ValueError(
            f"{path} exceeds the maximum metadata node count of {MAX_METADATA_NODES}"
        )

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not _MIN_SIGNED_INTEGER <= value <= _MAX_SIGNED_INTEGER:
            raise ValueError(f"{path} integer metadata must fit signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numeric metadata must be finite")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_METADATA_TEXT_LENGTH:
            raise ValueError(
                f"{path} string metadata must contain at most "
                f"{_MAX_METADATA_TEXT_LENGTH} characters"
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_COLLECTION_VALUES:
            raise ValueError(
                f"{path} mappings must contain at most "
                f"{MAX_METADATA_COLLECTION_VALUES} values"
            )
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = _metadata_key(raw_key, path)
            normalized[key] = _freeze_json_value(
                value[raw_key],
                f"{path}.{key}",
                depth=depth + 1,
                node_count=node_count,
            )
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_METADATA_COLLECTION_VALUES:
            raise ValueError(
                f"{path} collections must contain at most "
                f"{MAX_METADATA_COLLECTION_VALUES} values"
            )
        return tuple(
            _freeze_json_value(
                entry,
                f"{path}[{index}]",
                depth=depth + 1,
                node_count=node_count,
            )
            for index, entry in enumerate(value)
        )
    raise ValueError(f"{path} contains an unsupported metadata value")


def _freeze_metadata(value: Any) -> MappingProxyType:
    """Return one deeply immutable bounded metadata mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    frozen = _freeze_json_value(value, "$.metadata", depth=0, node_count=[0])
    if not isinstance(frozen, MappingProxyType):
        raise RuntimeError("metadata normalization did not produce an immutable mapping")
    return frozen


def _thaw_json_value(value: Any) -> Any:
    """Return ordinary JSON-compatible containers from immutable domain values."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(entry) for entry in value]
    return value


def _canonical_payload(value: Any) -> Any:
    """Return the authoritative content used to serialize and address an artifact."""
    content_method = getattr(value, "_content_dict", None)
    if callable(content_method):
        return content_method()
    dictionary_method = getattr(value, "to_dict", None)
    if callable(dictionary_method):
        return dictionary_method()
    return value


def canonical_json(value: Any) -> str:
    """Serialize bounded JSON-compatible content deterministically as UTF-8 text."""
    payload = _canonical_payload(value)
    frozen = _freeze_json_value(payload, "$", depth=0, node_count=[0])
    return json.dumps(
        _thaw_json_value(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def artifact_digest(value: Any) -> str:
    """Return the SHA-256 identity of one canonical bounded artifact."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConstructSpec:
    """One declared construct and the exact rubrics that operationalize it."""

    construct_id: str
    construct_definition: str
    rubric_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize the construct and immutable rubric references."""
        object.__setattr__(
            self,
            "construct_id",
            _identifier(self.construct_id, "construct_id"),
        )
        object.__setattr__(
            self,
            "construct_definition",
            _text(self.construct_definition, "construct_definition"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _sorted_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible construct representation."""
        return {
            "construct_id": self.construct_id,
            "construct_definition": self.construct_definition,
            "rubric_fingerprints": list(self.rubric_fingerprints),
        }


@dataclass(frozen=True)
class EnginePolicy:
    """Allowed human and automated rater boundary for an assessment."""

    policy_id: str
    engine_ids: tuple[str, ...] = ()
    allow_human_raters: bool = True
    allow_automated_raters: bool = False
    minimum_raters_per_response: int = 1

    def __post_init__(self) -> None:
        """Normalize engine identities and reject contradictory rater policies."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        engines = _sorted_identifiers(self.engine_ids, "engine_ids", minimum=0)
        object.__setattr__(self, "engine_ids", engines)
        human = _strict_boolean(self.allow_human_raters, "allow_human_raters")
        automated = _strict_boolean(
            self.allow_automated_raters,
            "allow_automated_raters",
        )
        object.__setattr__(self, "allow_human_raters", human)
        object.__setattr__(self, "allow_automated_raters", automated)
        object.__setattr__(
            self,
            "minimum_raters_per_response",
            _bounded_positive_integer(
                self.minimum_raters_per_response,
                "minimum_raters_per_response",
                _MAX_RATERS_PER_RESPONSE,
            ),
        )
        if not human and not automated:
            raise ValueError("engine policy must allow at least one rater kind")
        if automated and not engines:
            raise ValueError("automated scoring requires at least one engine")
        if not automated and engines:
            raise ValueError("engine_ids must be empty when automated raters are disabled")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible engine policy."""
        return {
            "policy_id": self.policy_id,
            "engine_ids": list(self.engine_ids),
            "allow_human_raters": self.allow_human_raters,
            "allow_automated_raters": self.allow_automated_raters,
            "minimum_raters_per_response": self.minimum_raters_per_response,
        }


@dataclass(frozen=True)
class CalibrationPolicy:
    """Declared calibration model and the constructs included in that model."""

    policy_id: str
    model_id: str
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize calibration-model and construct references."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "construct_ids",
            _sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible calibration policy."""
        return {
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "construct_ids": list(self.construct_ids),
        }


@dataclass(frozen=True)
class ValidationPolicy:
    """Validation metrics and construct scopes required before score use."""

    policy_id: str
    metric_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize metric and construct references."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "metric_ids",
            _sorted_identifiers(self.metric_ids, "metric_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            _sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible validation policy."""
        return {
            "policy_id": self.policy_id,
            "metric_ids": list(self.metric_ids),
            "construct_ids": list(self.construct_ids),
        }


@dataclass(frozen=True)
class AdjudicationPolicy:
    """Transparent human-review triggers and their construct scopes."""

    policy_id: str
    trigger_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize adjudication-trigger and construct references."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "trigger_ids",
            _sorted_identifiers(self.trigger_ids, "trigger_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            _sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible adjudication policy."""
        return {
            "policy_id": self.policy_id,
            "trigger_ids": list(self.trigger_ids),
            "construct_ids": list(self.construct_ids),
        }


@dataclass(frozen=True)
class MonitoringPolicy:
    """Versioned drift metrics and construct scopes for operational monitoring."""

    policy_id: str
    metric_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize monitoring-metric and construct references."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "metric_ids",
            _sorted_identifiers(self.metric_ids, "metric_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            _sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible monitoring policy."""
        return {
            "policy_id": self.policy_id,
            "metric_ids": list(self.metric_ids),
            "construct_ids": list(self.construct_ids),
        }


@dataclass(frozen=True)
class ReportingPolicy:
    """Allowed report artifacts and construct scopes for one assessment."""

    policy_id: str
    format_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]
    include_exact_values: bool = True

    def __post_init__(self) -> None:
        """Normalize report formats, construct references, and disclosure policy."""
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "format_ids",
            _sorted_identifiers(self.format_ids, "format_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            _sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )
        object.__setattr__(
            self,
            "include_exact_values",
            _strict_boolean(self.include_exact_values, "include_exact_values"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible reporting policy."""
        return {
            "policy_id": self.policy_id,
            "format_ids": list(self.format_ids),
            "construct_ids": list(self.construct_ids),
            "include_exact_values": self.include_exact_values,
        }


@dataclass(frozen=True)
class AssessmentSpec:
    """Factory-sealed content-addressed assessment and scoring-policy graph."""

    assessment_id: str
    assessment_version: str
    constructs: tuple[ConstructSpec, ...]
    rubric_fingerprints: tuple[str, ...]
    response_type: AssessmentResponseType
    engine_policy: EnginePolicy
    calibration_policy: CalibrationPolicy
    validation_policy: ValidationPolicy
    adjudication_policy: AdjudicationPolicy
    monitoring_policy: MonitoringPolicy
    reporting_policy: ReportingPolicy
    metadata: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION
    _assessment_token: InitVar[object | None] = None

    def __post_init__(self, _assessment_token: object | None) -> None:
        """Reject direct construction and revalidate the immutable public artifact."""
        if _assessment_token is not _ASSESSMENT_SPEC_TOKEN:
            raise ValueError("AssessmentSpec must be created by build_assessment_spec")
        object.__setattr__(
            self,
            "assessment_id",
            _identifier(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "assessment_version",
            _semantic_version(self.assessment_version, "assessment_version"),
        )
        object.__setattr__(self, "response_type", _response_type(self.response_type))
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _sorted_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    @property
    def construct_ids(self) -> tuple[str, ...]:
        """Return construct identifiers in deterministic assessment order."""
        return tuple(construct.construct_id for construct in self.constructs)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical assessment content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "assessment_version": self.assessment_version,
            "constructs": [construct.to_dict() for construct in self.constructs],
            "rubric_fingerprints": list(self.rubric_fingerprints),
            "response_type": self.response_type.value,
            "engine_policy": self.engine_policy.to_dict(),
            "calibration_policy": self.calibration_policy.to_dict(),
            "validation_policy": self.validation_policy.to_dict(),
            "adjudication_policy": self.adjudication_policy.to_dict(),
            "monitoring_policy": self.monitoring_policy.to_dict(),
            "reporting_policy": self.reporting_policy.to_dict(),
            "metadata": _thaw_json_value(self.metadata),
        }

    @property
    def assessment_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable assessment contract."""
        return artifact_digest(self._content_dict())

    @property
    def assessment_handle(self) -> str:
        """Return a descriptive 128-bit public assessment-contract handle."""
        return f"assessment_spec_{self.assessment_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic assessment identities."""
        return {
            **self._content_dict(),
            "assessment_handle": self.assessment_handle,
            "assessment_fingerprint": self.assessment_fingerprint,
        }


def _materialize_constructs(values: Iterable[Any]) -> tuple[ConstructSpec, ...]:
    """Return bounded typed constructs in deterministic identifier order."""
    raw = _bounded_values(
        values,
        "constructs",
        minimum=1,
        maximum=_MAX_ASSESSMENT_CONSTRUCTS,
    )
    for index, construct in enumerate(raw):
        if not isinstance(construct, ConstructSpec):
            raise TypeError(f"constructs[{index}] must be a ConstructSpec")
    constructs = tuple(sorted(raw, key=lambda entry: entry.construct_id))
    ids = tuple(construct.construct_id for construct in constructs)
    if len(set(ids)) != len(ids):
        raise AssessmentSpecError(
            "duplicate_construct_id",
            "$.constructs",
            "construct identifiers must be unique",
        )
    return constructs


def _materialize_rubrics(values: Iterable[Any]) -> tuple[RubricSpecification, ...]:
    """Return bounded typed rubrics in deterministic fingerprint order."""
    raw = _bounded_values(
        values,
        "rubrics",
        minimum=1,
        maximum=_MAX_ASSESSMENT_RUBRICS,
    )
    for index, rubric in enumerate(raw):
        if not isinstance(rubric, RubricSpecification):
            raise TypeError(f"rubrics[{index}] must be a RubricSpecification")
    return tuple(sorted(raw, key=lambda entry: entry.fingerprint))


def _validate_policy_types(
    engine_policy: Any,
    calibration_policy: Any,
    validation_policy: Any,
    adjudication_policy: Any,
    monitoring_policy: Any,
    reporting_policy: Any,
) -> None:
    """Reject policy objects from parallel or incompatible contract systems."""
    expected = (
        ("engine_policy", engine_policy, EnginePolicy),
        ("calibration_policy", calibration_policy, CalibrationPolicy),
        ("validation_policy", validation_policy, ValidationPolicy),
        ("adjudication_policy", adjudication_policy, AdjudicationPolicy),
        ("monitoring_policy", monitoring_policy, MonitoringPolicy),
        ("reporting_policy", reporting_policy, ReportingPolicy),
    )
    for name, value, expected_type in expected:
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be a {expected_type.__name__}")


def _validate_policy_constructs(
    construct_ids: tuple[str, ...],
    policy_name: str,
    policy_construct_ids: tuple[str, ...],
) -> None:
    """Require every policy construct reference to resolve in the assessment."""
    known = set(construct_ids)
    for index, construct_id in enumerate(policy_construct_ids):
        if construct_id not in known:
            raise AssessmentSpecError(
                "unknown_policy_construct",
                f"$.{policy_name}.construct_ids[{index}]",
                "policy construct reference is not declared by the assessment",
            )


def build_assessment_spec(
    *,
    assessment_id: str,
    assessment_version: str,
    constructs: Iterable[ConstructSpec],
    rubrics: Iterable[RubricSpecification],
    response_type: AssessmentResponseType | str,
    engine_policy: EnginePolicy,
    calibration_policy: CalibrationPolicy,
    validation_policy: ValidationPolicy,
    adjudication_policy: AdjudicationPolicy,
    monitoring_policy: MonitoringPolicy,
    reporting_policy: ReportingPolicy,
    metadata: Mapping[str, Any] | None = None,
) -> AssessmentSpec:
    """Build one cross-reference-validated immutable assessment contract."""
    normalized_response_type = _response_type(response_type)
    _validate_policy_types(
        engine_policy,
        calibration_policy,
        validation_policy,
        adjudication_policy,
        monitoring_policy,
        reporting_policy,
    )
    normalized_constructs = _materialize_constructs(constructs)
    normalized_rubrics = _materialize_rubrics(rubrics)

    rubrics_by_fingerprint: dict[str, RubricSpecification] = {}
    fingerprints_by_id: dict[str, str] = {}
    for rubric in normalized_rubrics:
        fingerprint = rubric.fingerprint
        if fingerprint in rubrics_by_fingerprint:
            raise AssessmentSpecError(
                "duplicate_rubric_fingerprint",
                "$.rubrics",
                "rubric fingerprints must be unique",
            )
        prior = fingerprints_by_id.get(rubric.rubric_id)
        if prior is not None and prior != fingerprint:
            raise AssessmentSpecError(
                "duplicate_rubric_id",
                "$.rubrics",
                "one rubric identifier cannot name multiple fingerprints",
            )
        rubrics_by_fingerprint[fingerprint] = rubric
        fingerprints_by_id[rubric.rubric_id] = fingerprint

    referenced: set[str] = set()
    for construct_index, construct in enumerate(normalized_constructs):
        for fingerprint_index, fingerprint in enumerate(construct.rubric_fingerprints):
            path = (
                f"$.constructs[{construct_index}].rubric_fingerprints"
                f"[{fingerprint_index}]"
            )
            rubric = rubrics_by_fingerprint.get(fingerprint)
            if rubric is None:
                raise AssessmentSpecError(
                    "unknown_rubric_fingerprint",
                    path,
                    "rubric fingerprint is absent from the supplied registry",
                )
            if rubric.construct_id != construct.construct_id:
                raise AssessmentSpecError(
                    "rubric_construct_mismatch",
                    path,
                    "rubric construct does not match the declared construct",
                )
            referenced.add(fingerprint)

    unused = sorted(set(rubrics_by_fingerprint).difference(referenced))
    if unused:
        raise AssessmentSpecError(
            "unused_rubric_fingerprint",
            "$.rubrics",
            "every supplied rubric must be bound to one declared construct",
        )

    construct_ids = tuple(construct.construct_id for construct in normalized_constructs)
    _validate_policy_constructs(
        construct_ids,
        "calibration_policy",
        calibration_policy.construct_ids,
    )
    _validate_policy_constructs(
        construct_ids,
        "validation_policy",
        validation_policy.construct_ids,
    )
    _validate_policy_constructs(
        construct_ids,
        "adjudication_policy",
        adjudication_policy.construct_ids,
    )
    _validate_policy_constructs(
        construct_ids,
        "monitoring_policy",
        monitoring_policy.construct_ids,
    )
    _validate_policy_constructs(
        construct_ids,
        "reporting_policy",
        reporting_policy.construct_ids,
    )

    return AssessmentSpec(
        assessment_id=assessment_id,
        assessment_version=assessment_version,
        constructs=normalized_constructs,
        rubric_fingerprints=tuple(sorted(rubrics_by_fingerprint)),
        response_type=normalized_response_type,
        engine_policy=engine_policy,
        calibration_policy=calibration_policy,
        validation_policy=validation_policy,
        adjudication_policy=adjudication_policy,
        monitoring_policy=monitoring_policy,
        reporting_policy=reporting_policy,
        metadata={} if metadata is None else metadata,
        schema_version=SCHEMA_VERSION,
        _assessment_token=_ASSESSMENT_SPEC_TOKEN,
    )
