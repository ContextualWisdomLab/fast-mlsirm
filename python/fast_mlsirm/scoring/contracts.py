"""Immutable assessment and policy contracts for automated scoring."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import json
import math
import operator
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from fast_mlsirm.rubric import ResponseFormat, RubricSpecification

SCORING_SCHEMA_VERSION = "1.0"
MAX_CONSTRUCTS = 32
MAX_RUBRICS = 64
MAX_POLICY_VALUES = 64
MAX_METADATA_ENTRIES = 64
MAX_METADATA_VALUES = 64
MAX_METADATA_DEPTH = 8
MAX_METADATA_TEXT = 4_096
MAX_ENGINE_ATTEMPTS = 16
MAX_MONITORING_WINDOW = 1_000_000

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_METADATA_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSESSMENT_CONSTRUCTION_TOKEN = object()


class AutomatedScoringError(ValueError):
    """Base exception exposing one stable code and optional public field."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        """Store bounded machine-readable context without rejected values."""
        super().__init__(message)
        self.code = code
        self.field = field


class InvalidAssessmentSpecError(AutomatedScoringError):
    """Raised when an assessment or policy contract fails closed."""


class CalibrationModel(str, Enum):
    """Existing Rust-backed model families available to scoring orchestration."""

    FACETS = "facets"
    MIRT = "mirt"
    BIFACTOR = "bifactor"
    TESTLET = "testlet"


class GateComparison(str, Enum):
    """Supported directions for one declared validation threshold."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"


def _error(code: str, message: str, *, field: str | None = None) -> None:
    """Raise one structured assessment-contract error."""
    raise InvalidAssessmentSpecError(code, message, field=field)


def _text(value: Any, name: str, *, maximum: int = MAX_METADATA_TEXT) -> str:
    """Normalize bounded non-empty text without echoing rejected content."""
    if not isinstance(value, str):
        _error("invalid_text_type", f"{name} must be a string", field=name)
    normalized = value.strip()
    if not normalized:
        _error("empty_text", f"{name} must not be empty", field=name)
    if len(normalized) > maximum:
        _error(
            "text_too_long",
            f"{name} must contain at most {maximum} characters",
            field=name,
        )
    return normalized


def _identifier(value: Any, name: str) -> str:
    """Normalize a descriptive two-or-more-token lower snake-case identifier."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        _error(
            "invalid_identifier",
            f"{name} must use two-or-more-token lower snake_case",
            field=name,
        )
    return normalized


def _semantic_version(value: Any, name: str) -> str:
    """Normalize a canonical numeric semantic version."""
    normalized = _text(value, name, maximum=64)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(normalized) is None:
        _error(
            "invalid_semantic_version",
            f"{name} must be a canonical semantic version (major.minor.patch)",
            field=name,
        )
    return normalized


def _fingerprint(value: Any, name: str) -> str:
    """Normalize a complete lower-hexadecimal SHA-256 digest."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        _error(
            "invalid_fingerprint",
            f"{name} must be a 64-character lower hexadecimal digest",
            field=name,
        )
    return normalized


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Normalize a bounded integer while rejecting booleans and fractions."""
    if isinstance(value, bool):
        _error("invalid_integer", f"{name} must be an integer", field=name)
    try:
        normalized = operator.index(value)
    except TypeError:
        _error("invalid_integer", f"{name} must be an integer", field=name)
    if not minimum <= normalized <= maximum:
        _error(
            "integer_out_of_range",
            f"{name} must be between {minimum} and {maximum}",
            field=name,
        )
    return normalized


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    """Normalize a finite real number and optional lower bound."""
    if isinstance(value, bool):
        _error("invalid_number", f"{name} must be a finite number", field=name)
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        _error("invalid_number", f"{name} must be a finite number", field=name)
    if not math.isfinite(normalized):
        _error("non_finite_number", f"{name} must be finite", field=name)
    if minimum is not None and normalized < minimum:
        _error(
            "number_below_minimum",
            f"{name} must be at least {minimum}",
            field=name,
        )
    return normalized


def _boolean(value: Any, name: str) -> bool:
    """Require an exact Boolean policy value."""
    if not isinstance(value, bool):
        _error("invalid_boolean", f"{name} must be a boolean", field=name)
    return value


def _bounded_tuple(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[Any, ...]:
    """Materialize a caller collection under a strict finite work budget."""
    if isinstance(values, (str, bytes)):
        _error("invalid_collection", f"{name} must be a collection", field=name)
    try:
        iterator = iter(values)
    except TypeError:
        _error("invalid_collection", f"{name} must be a collection", field=name)
    output: list[Any] = []
    for index, value in enumerate(iterator):
        if index >= maximum:
            _error(
                "collection_too_large",
                f"{name} must contain at most {maximum} values",
                field=name,
            )
        output.append(value)
    if len(output) < minimum:
        _error(
            "collection_too_small",
            f"{name} must contain at least {minimum} value"
            f"{'s' if minimum != 1 else ''}",
            field=name,
        )
    return tuple(output)


def _identifier_tuple(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_POLICY_VALUES,
) -> tuple[str, ...]:
    """Normalize one finite unique collection of public identifiers."""
    raw = _bounded_tuple(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _identifier(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        _error(
            "duplicate_identifier",
            f"{name} must not contain duplicates",
            field=name,
        )
    return normalized


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    """Normalize an enum instance or exact serialized value."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        choices = [member.value for member in enum_type]
        _error(
            "invalid_enum_value",
            f"{name} must be one of {choices}",
            field=name,
        )


def _freeze_json(value: Any, name: str, *, depth: int = 0) -> Any:
    """Return deeply immutable bounded JSON content or fail closed."""
    if depth > MAX_METADATA_DEPTH:
        _error(
            "metadata_too_deep",
            f"{name} exceeds the maximum nesting depth of {MAX_METADATA_DEPTH}",
            field=name,
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _error("non_finite_metadata", f"{name} must be finite", field=name)
        return value
    if isinstance(value, str):
        if len(value) > MAX_METADATA_TEXT:
            _error(
                "metadata_text_too_long",
                f"{name} must contain at most {MAX_METADATA_TEXT} characters",
                field=name,
            )
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_METADATA_VALUES:
            _error(
                "collection_too_large",
                f"{name} must contain at most {MAX_METADATA_VALUES} values",
                field=name,
            )
        return tuple(
            _freeze_json(item, f"{name}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_ENTRIES:
            _error(
                "metadata_mapping_too_large",
                f"{name} must contain at most {MAX_METADATA_ENTRIES} entries",
                field=name,
            )
        normalized: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str) or _METADATA_KEY_PATTERN.fullmatch(key) is None:
                _error(
                    "invalid_metadata_key",
                    f"{name} keys must use lower snake_case",
                    field=name,
                )
            normalized[key] = _freeze_json(
                value[key],
                f"{name}.{key}",
                depth=depth + 1,
            )
        return MappingProxyType(normalized)
    _error(
        "unsupported_metadata_type",
        f"{name} must contain only JSON-compatible values",
        field=name,
    )


def _json_ready(value: Any) -> Any:
    """Return a mutable JSON representation of one immutable contract value."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "_canonical_dict"):
        return value._canonical_dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _error("non_finite_metadata", "numeric values must be finite")
        return value
    _error(
        "unsupported_metadata_type",
        "content must contain only JSON-compatible values",
    )


def canonical_json(value: Any) -> str:
    """Serialize a reviewed contract or bounded JSON value deterministically."""
    frozen = _freeze_json(value, "content") if not hasattr(value, "to_dict") else value
    return json.dumps(
        _json_ready(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def artifact_digest(value: Any) -> str:
    """Return a complete SHA-256 identity over canonical UTF-8 JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConstructSpec:
    """One explicitly declared construct and buyer-facing reporting label."""

    construct_id: str
    construct_definition: str
    reporting_label: str

    def __post_init__(self) -> None:
        """Normalize and validate construct identity and bounded text."""
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
            "reporting_label",
            _text(self.reporting_label, "reporting_label", maximum=256),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible construct declaration."""
        return {
            "construct_id": self.construct_id,
            "construct_definition": self.construct_definition,
            "reporting_label": self.reporting_label,
        }


@dataclass(frozen=True)
class EnginePolicy:
    """Declare which scorer engines may contribute observations."""

    allowed_engine_ids: tuple[str, ...] = ()
    require_evidence: bool = True
    maximum_attempts: int = 1

    def __post_init__(self) -> None:
        """Normalize finite engine allowlists and execution controls."""
        object.__setattr__(
            self,
            "allowed_engine_ids",
            _identifier_tuple(
                self.allowed_engine_ids,
                "allowed_engine_ids",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "require_evidence",
            _boolean(self.require_evidence, "require_evidence"),
        )
        object.__setattr__(
            self,
            "maximum_attempts",
            _integer(
                self.maximum_attempts,
                "maximum_attempts",
                minimum=1,
                maximum=MAX_ENGINE_ATTEMPTS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the engine policy as JSON-compatible content."""
        return {
            "allowed_engine_ids": list(self.allowed_engine_ids),
            "require_evidence": self.require_evidence,
            "maximum_attempts": self.maximum_attempts,
        }


@dataclass(frozen=True)
class CalibrationPolicy:
    """Declare the existing Rust-backed calibration family and design guards."""

    model: CalibrationModel = CalibrationModel.FACETS
    minimum_raters: int = 1
    require_connected_design: bool = True
    allow_missing_observations: bool = True

    def __post_init__(self) -> None:
        """Normalize the calibration model and finite design requirements."""
        object.__setattr__(
            self,
            "model",
            _enum_value(self.model, CalibrationModel, "model"),
        )
        object.__setattr__(
            self,
            "minimum_raters",
            _integer(
                self.minimum_raters,
                "minimum_raters",
                minimum=1,
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "require_connected_design",
            _boolean(self.require_connected_design, "require_connected_design"),
        )
        object.__setattr__(
            self,
            "allow_missing_observations",
            _boolean(self.allow_missing_observations, "allow_missing_observations"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible calibration controls."""
        return {
            "model": self.model.value,
            "minimum_raters": self.minimum_raters,
            "require_connected_design": self.require_connected_design,
            "allow_missing_observations": self.allow_missing_observations,
        }


@dataclass(frozen=True)
class ValidationGate:
    """One evidence threshold evaluated by future validation orchestration."""

    metric_id: str
    comparison: GateComparison
    threshold: float
    minimum_observations: int = 2
    group_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize one bounded metric threshold and optional subgroup."""
        object.__setattr__(
            self,
            "metric_id",
            _identifier(self.metric_id, "metric_id"),
        )
        object.__setattr__(
            self,
            "comparison",
            _enum_value(self.comparison, GateComparison, "comparison"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_number(self.threshold, "threshold"),
        )
        object.__setattr__(
            self,
            "minimum_observations",
            _integer(
                self.minimum_observations,
                "minimum_observations",
                minimum=2,
                maximum=MAX_MONITORING_WINDOW,
            ),
        )
        if self.group_id is not None:
            object.__setattr__(
                self,
                "group_id",
                _identifier(self.group_id, "group_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the validation gate as JSON-compatible content."""
        return {
            "metric_id": self.metric_id,
            "comparison": self.comparison.value,
            "threshold": self.threshold,
            "minimum_observations": self.minimum_observations,
            "group_id": self.group_id,
        }


@dataclass(frozen=True)
class ValidationPolicy:
    """Preserve ordered validation gates and declared subgroup requirements."""

    gates: tuple[ValidationGate, ...]
    required_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize bounded gates and reject duplicate metric/group identities."""
        raw = _bounded_tuple(
            self.gates,
            "gates",
            minimum=1,
            maximum=MAX_POLICY_VALUES,
        )
        for index, gate in enumerate(raw):
            if not isinstance(gate, ValidationGate):
                _error(
                    "invalid_validation_gate",
                    f"gates[{index}] must be a ValidationGate",
                    field="gates",
                )
        identities = tuple((gate.metric_id, gate.group_id) for gate in raw)
        if len(set(identities)) != len(identities):
            _error(
                "duplicate_validation_gate",
                "gates must not repeat a metric and group pair",
                field="gates",
            )
        object.__setattr__(self, "gates", tuple(raw))
        object.__setattr__(
            self,
            "required_group_ids",
            _identifier_tuple(
                self.required_group_ids,
                "required_group_ids",
                minimum=0,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return ordered validation policy content."""
        return {
            "gates": [gate.to_dict() for gate in self.gates],
            "required_group_ids": list(self.required_group_ids),
        }


@dataclass(frozen=True)
class AdjudicationPolicy:
    """Declare transparent conditions that route observations to human review."""

    trigger_codes: tuple[str, ...]
    maximum_score_distance: float | None = None
    maximum_uncertainty: float | None = None
    require_evidence: bool = True

    def __post_init__(self) -> None:
        """Normalize trigger identities and optional non-negative thresholds."""
        object.__setattr__(
            self,
            "trigger_codes",
            _identifier_tuple(
                self.trigger_codes,
                "trigger_codes",
                minimum=1,
            ),
        )
        if self.maximum_score_distance is not None:
            object.__setattr__(
                self,
                "maximum_score_distance",
                _finite_number(
                    self.maximum_score_distance,
                    "maximum_score_distance",
                    minimum=0.0,
                ),
            )
        if self.maximum_uncertainty is not None:
            object.__setattr__(
                self,
                "maximum_uncertainty",
                _finite_number(
                    self.maximum_uncertainty,
                    "maximum_uncertainty",
                    minimum=0.0,
                ),
            )
        object.__setattr__(
            self,
            "require_evidence",
            _boolean(self.require_evidence, "require_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible human-review controls."""
        return {
            "trigger_codes": list(self.trigger_codes),
            "maximum_score_distance": self.maximum_score_distance,
            "maximum_uncertainty": self.maximum_uncertainty,
            "require_evidence": self.require_evidence,
        }


@dataclass(frozen=True)
class MonitoringPolicy:
    """Declare evidence windows and version changes monitored after deployment."""

    window_size: int
    minimum_observations: int
    monitored_group_ids: tuple[str, ...] = ()
    alert_on_rubric_change: bool = True
    alert_on_engine_change: bool = True

    def __post_init__(self) -> None:
        """Normalize finite monitoring windows and subgroup identities."""
        window_size = _integer(
            self.window_size,
            "window_size",
            minimum=2,
            maximum=MAX_MONITORING_WINDOW,
        )
        minimum_observations = _integer(
            self.minimum_observations,
            "minimum_observations",
            minimum=2,
            maximum=MAX_MONITORING_WINDOW,
        )
        if minimum_observations > window_size:
            _error(
                "monitoring_sample_exceeds_window",
                "minimum_observations must not exceed window_size",
                field="minimum_observations",
            )
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "minimum_observations", minimum_observations)
        object.__setattr__(
            self,
            "monitored_group_ids",
            _identifier_tuple(
                self.monitored_group_ids,
                "monitored_group_ids",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "alert_on_rubric_change",
            _boolean(self.alert_on_rubric_change, "alert_on_rubric_change"),
        )
        object.__setattr__(
            self,
            "alert_on_engine_change",
            _boolean(self.alert_on_engine_change, "alert_on_engine_change"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible monitoring controls."""
        return {
            "window_size": self.window_size,
            "minimum_observations": self.minimum_observations,
            "monitored_group_ids": list(self.monitored_group_ids),
            "alert_on_rubric_change": self.alert_on_rubric_change,
            "alert_on_engine_change": self.alert_on_engine_change,
        }


@dataclass(frozen=True)
class AssessmentSpec:
    """Factory-sealed immutable contract binding exact rubric and policy versions."""

    assessment_id: str
    assessment_version: str
    constructs: tuple[ConstructSpec, ...]
    rubric_fingerprints: tuple[str, ...]
    response_format: ResponseFormat
    declared_engine_ids: tuple[str, ...]
    declared_group_ids: tuple[str, ...]
    engine_policy: EnginePolicy
    calibration_policy: CalibrationPolicy
    validation_policy: ValidationPolicy
    adjudication_policy: AdjudicationPolicy
    monitoring_policy: MonitoringPolicy
    metadata: Mapping[str, Any]
    schema_version: str = SCORING_SCHEMA_VERSION
    _construction_token: InitVar[object | None] = None

    def __post_init__(self, _construction_token: object | None) -> None:
        """Prevent direct construction and normalize already replayed content."""
        if _construction_token is not _ASSESSMENT_CONSTRUCTION_TOKEN:
            _error(
                "factory_required",
                "AssessmentSpec must be created by build_assessment_spec",
                field="assessment_spec",
            )
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
        if self.schema_version != SCORING_SCHEMA_VERSION:
            _error(
                "unsupported_schema_version",
                f"schema_version must be '{SCORING_SCHEMA_VERSION}'",
                field="schema_version",
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(self.metadata, "metadata"),
        )

    def _canonical_dict(self) -> dict[str, Any]:
        """Return fingerprinted content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "assessment_version": self.assessment_version,
            "constructs": [construct.to_dict() for construct in self.constructs],
            "rubric_fingerprints": list(self.rubric_fingerprints),
            "response_format": self.response_format.value,
            "declared_engine_ids": list(self.declared_engine_ids),
            "declared_group_ids": list(self.declared_group_ids),
            "engine_policy": self.engine_policy.to_dict(),
            "calibration_policy": self.calibration_policy.to_dict(),
            "validation_policy": self.validation_policy.to_dict(),
            "adjudication_policy": self.adjudication_policy.to_dict(),
            "monitoring_policy": self.monitoring_policy.to_dict(),
            "metadata": _json_ready(self.metadata),
        }

    @property
    def assessment_fingerprint(self) -> str:
        """Return the complete SHA-256 identity of this assessment contract."""
        return artifact_digest(self)

    @property
    def assessment_handle(self) -> str:
        """Return a descriptive public handle retaining 128 fingerprint bits."""
        return f"assessment_spec_{self.assessment_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content plus derived audit identities."""
        payload = self._canonical_dict()
        payload["assessment_fingerprint"] = self.assessment_fingerprint
        payload["assessment_handle"] = self.assessment_handle
        return payload


def _validate_policy_types(
    engine_policy: Any,
    calibration_policy: Any,
    validation_policy: Any,
    adjudication_policy: Any,
    monitoring_policy: Any,
) -> None:
    """Require exact reviewed policy classes before reference validation."""
    pairs = (
        ("engine_policy", engine_policy, EnginePolicy),
        ("calibration_policy", calibration_policy, CalibrationPolicy),
        ("validation_policy", validation_policy, ValidationPolicy),
        ("adjudication_policy", adjudication_policy, AdjudicationPolicy),
        ("monitoring_policy", monitoring_policy, MonitoringPolicy),
    )
    for name, value, expected in pairs:
        if not isinstance(value, expected):
            _error(
                "invalid_policy",
                f"{name} must be a {expected.__name__}",
                field=name,
            )


def build_assessment_spec(
    *,
    assessment_id: str,
    assessment_version: str,
    constructs: Iterable[ConstructSpec],
    rubric_fingerprints: Iterable[str],
    response_format: ResponseFormat | str,
    declared_engine_ids: Iterable[str],
    declared_group_ids: Iterable[str],
    engine_policy: EnginePolicy,
    calibration_policy: CalibrationPolicy,
    validation_policy: ValidationPolicy,
    adjudication_policy: AdjudicationPolicy,
    monitoring_policy: MonitoringPolicy,
    rubrics: Iterable[RubricSpecification],
    schema_version: str = SCORING_SCHEMA_VERSION,
    metadata: Mapping[str, Any] | None = None,
) -> AssessmentSpec:
    """Build an assessment only after exact rubric and policy replay succeeds."""
    normalized_assessment_id = _identifier(assessment_id, "assessment_id")
    normalized_version = _semantic_version(
        assessment_version,
        "assessment_version",
    )
    if schema_version != SCORING_SCHEMA_VERSION:
        _error(
            "unsupported_schema_version",
            f"schema_version must be '{SCORING_SCHEMA_VERSION}'",
            field="schema_version",
        )
    normalized_constructs = _bounded_tuple(
        constructs,
        "constructs",
        minimum=1,
        maximum=MAX_CONSTRUCTS,
    )
    for index, construct in enumerate(normalized_constructs):
        if not isinstance(construct, ConstructSpec):
            _error(
                "invalid_construct",
                f"constructs[{index}] must be a ConstructSpec",
                field="constructs",
            )
    construct_ids = tuple(construct.construct_id for construct in normalized_constructs)
    if len(set(construct_ids)) != len(construct_ids):
        _error(
            "duplicate_construct",
            "constructs must not repeat construct identifiers",
            field="constructs",
        )
    raw_fingerprints = _bounded_tuple(
        rubric_fingerprints,
        "rubric_fingerprints",
        minimum=1,
        maximum=MAX_RUBRICS,
    )
    normalized_fingerprints = tuple(
        _fingerprint(value, f"rubric_fingerprints[{index}]")
        for index, value in enumerate(raw_fingerprints)
    )
    if len(set(normalized_fingerprints)) != len(normalized_fingerprints):
        _error(
            "duplicate_rubric_fingerprint",
            "rubric_fingerprints must not contain duplicates",
            field="rubric_fingerprints",
        )
    normalized_response_format = _enum_value(
        response_format,
        ResponseFormat,
        "response_format",
    )
    engine_ids = _identifier_tuple(
        declared_engine_ids,
        "declared_engine_ids",
        minimum=0,
    )
    group_ids = _identifier_tuple(
        declared_group_ids,
        "declared_group_ids",
        minimum=0,
    )
    _validate_policy_types(
        engine_policy,
        calibration_policy,
        validation_policy,
        adjudication_policy,
        monitoring_policy,
    )
    raw_rubrics = _bounded_tuple(
        rubrics,
        "rubrics",
        minimum=1,
        maximum=MAX_RUBRICS,
    )
    for index, rubric in enumerate(raw_rubrics):
        if not isinstance(rubric, RubricSpecification):
            _error(
                "invalid_rubric",
                f"rubrics[{index}] must be a RubricSpecification",
                field="rubrics",
            )
    registry = {rubric.fingerprint: rubric for rubric in raw_rubrics}
    if len(registry) != len(raw_rubrics):
        _error(
            "duplicate_rubric",
            "rubrics must not contain duplicate fingerprints",
            field="rubrics",
        )
    for fingerprint in normalized_fingerprints:
        rubric = registry.get(fingerprint)
        if rubric is None:
            _error(
                "unknown_rubric_fingerprint",
                "rubric_fingerprints contains a digest absent from rubrics",
                field="rubric_fingerprints",
            )
        if rubric.construct_id not in construct_ids:
            _error(
                "unknown_rubric_construct",
                "a selected rubric references an undeclared construct",
                field="rubric_fingerprints",
            )
        if rubric.response_format is not normalized_response_format:
            _error(
                "rubric_response_format_mismatch",
                "every selected rubric must use response_format",
                field="response_format",
            )
    unknown_engines = set(engine_policy.allowed_engine_ids) - set(engine_ids)
    if unknown_engines:
        _error(
            "unknown_engine_reference",
            "engine_policy references an undeclared engine",
            field="engine_policy",
        )
    referenced_groups = set(validation_policy.required_group_ids)
    referenced_groups.update(
        gate.group_id for gate in validation_policy.gates if gate.group_id is not None
    )
    referenced_groups.update(monitoring_policy.monitored_group_ids)
    if referenced_groups - set(group_ids):
        _error(
            "unknown_group_reference",
            "a validation or monitoring policy references an undeclared group",
            field="declared_group_ids",
        )
    if metadata is not None and not isinstance(metadata, Mapping):
        _error(
            "invalid_metadata",
            "metadata must be a mapping",
            field="metadata",
        )
    normalized_metadata = _freeze_json({} if metadata is None else metadata, "metadata")
    return AssessmentSpec(
        assessment_id=normalized_assessment_id,
        assessment_version=normalized_version,
        constructs=tuple(normalized_constructs),
        rubric_fingerprints=normalized_fingerprints,
        response_format=normalized_response_format,
        declared_engine_ids=engine_ids,
        declared_group_ids=group_ids,
        engine_policy=engine_policy,
        calibration_policy=calibration_policy,
        validation_policy=validation_policy,
        adjudication_policy=adjudication_policy,
        monitoring_policy=monitoring_policy,
        metadata=normalized_metadata,
        schema_version=SCORING_SCHEMA_VERSION,
        _construction_token=_ASSESSMENT_CONSTRUCTION_TOKEN,
    )
