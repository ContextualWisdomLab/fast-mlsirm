"""Immutable provider-neutral assessment and scoring-policy contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Real
import operator
import re
from types import MappingProxyType
from typing import Any, NoReturn

from fast_mlsirm.rubric.models import (
    SCHEMA_VERSION,
    RubricSpecification,
    _bounded_values,
    _canonical_json,
    _identifier,
    _schema_version,
    _semantic_version,
    _sha256_hex,
    _text,
)

MAX_ASSESSMENT_RUBRICS = 64
MAX_ASSESSMENT_CONSTRUCTS = 64
MAX_POLICY_VALUES = 64
MAX_JSON_COLLECTION_VALUES = 64
MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 1_024
MAX_JSON_INTEGER = (1 << 63) - 1
MAX_POLICY_INTEGER = 1_000_000

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSESSMENT_SPEC_TOKEN = object()


class AutomatedScoringError(ValueError):
    """Base exception carrying a stable code and bounded field path."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Initialize one safe machine-readable scoring-contract error."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


class InvalidAssessmentSpecError(AutomatedScoringError):
    """Raised when an assessment or policy contract violates an invariant."""


class ResponseType(str, Enum):
    """Provider-neutral response-media families accepted by an assessment."""

    TEXT_RESPONSE = "text_response"
    AUDIO_RESPONSE = "audio_response"
    IMAGE_RESPONSE = "image_response"
    MULTIMODAL_RESPONSE = "multimodal_response"
    STRUCTURED_RESPONSE = "structured_response"


class MetricDirection(str, Enum):
    """Direction in which a declared metric threshold is satisfied."""

    AT_LEAST = "at_least"
    AT_MOST = "at_most"


def _invalid(code: str, path: str, message: str) -> NoReturn:
    """Raise one bounded assessment-domain validation error."""
    raise InvalidAssessmentSpecError(code, path, message)


def _identifier_value(value: Any, path: str) -> str:
    """Normalize one descriptive identifier as a domain error."""
    try:
        return _identifier(value, path)
    except ValueError as exc:
        raise InvalidAssessmentSpecError(
            "invalid_identifier",
            path,
            "must use two-or-more-token lower snake_case",
        ) from exc


def _semantic_version_value(value: Any, path: str) -> str:
    """Normalize one canonical semantic version as a domain error."""
    try:
        return _semantic_version(value, path)
    except ValueError as exc:
        raise InvalidAssessmentSpecError(
            "invalid_semantic_version",
            path,
            "must be a canonical major.minor.patch semantic version",
        ) from exc


def _schema_version_value(value: Any) -> str:
    """Normalize the implemented wire-schema version as a domain error."""
    try:
        return _schema_version(value)
    except ValueError as exc:
        raise InvalidAssessmentSpecError(
            "unsupported_schema_version",
            "schema_version",
            "must match the schema version implemented by this package",
        ) from exc


def _text_value(value: Any, path: str, *, maximum: int) -> str:
    """Normalize bounded non-empty text without reflecting caller content."""
    try:
        return _text(value, path, maximum=maximum)
    except ValueError as exc:
        raise InvalidAssessmentSpecError(
            "invalid_text_value",
            path,
            f"must be non-empty bounded text of at most {maximum} characters",
        ) from exc


def _boolean_value(value: Any, path: str) -> bool:
    """Accept only a literal boolean value."""
    if not isinstance(value, bool):
        _invalid("invalid_boolean", path, "must be a boolean")
    return value


def _integer_value(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int = MAX_POLICY_INTEGER,
) -> int:
    """Normalize one bounded integer while rejecting booleans and fractions."""
    if isinstance(value, bool):
        _invalid("invalid_integer", path, "must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise InvalidAssessmentSpecError(
            "invalid_integer",
            path,
            "must be an integer",
        ) from exc
    if not minimum <= normalized <= maximum:
        _invalid(
            "integer_out_of_range",
            path,
            f"must be between {minimum} and {maximum}",
        )
    return int(normalized)


def _finite_number(value: Any, path: str) -> float:
    """Normalize one finite real threshold while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        _invalid("invalid_finite_number", path, "must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        _invalid("invalid_finite_number", path, "must be a finite real number")
    return normalized


def _enum_value(value: Any, enum_type: type[Enum], path: str) -> Enum:
    """Normalize an enum instance or exact wire value."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise InvalidAssessmentSpecError(
            "invalid_enum_value",
            path,
            "must be one of the declared wire values",
        ) from exc


def _bounded_tuple(
    values: Iterable[Any],
    path: str,
    *,
    minimum: int,
    maximum: int = MAX_POLICY_VALUES,
) -> tuple[Any, ...]:
    """Materialize a bounded iterable and translate validation failures."""
    try:
        return _bounded_values(values, path, minimum=minimum, maximum=maximum)
    except ValueError as exc:
        raise InvalidAssessmentSpecError(
            "invalid_collection",
            path,
            f"must contain between {minimum} and {maximum} values",
        ) from exc


def _unique_identifiers(
    values: Iterable[Any],
    path: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    """Normalize a bounded, sorted set of descriptive identifiers."""
    materialized = _bounded_tuple(values, path, minimum=minimum)
    normalized = tuple(
        _identifier_value(value, f"{path}[{index}]")
        for index, value in enumerate(materialized)
    )
    if len(set(normalized)) != len(normalized):
        _invalid("duplicate_identifier", path, "must not contain duplicates")
    return tuple(sorted(normalized))


def _fingerprint_value(value: Any, path: str) -> str:
    """Normalize one exact lower-hex SHA-256 fingerprint."""
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        _invalid(
            "invalid_fingerprint",
            path,
            "must be a 64-character lowercase hexadecimal digest",
        )
    return value


def _unique_fingerprints(
    values: Iterable[Any],
    path: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    """Normalize a bounded, sorted set of exact SHA-256 fingerprints."""
    materialized = _bounded_tuple(values, path, minimum=minimum)
    normalized = tuple(
        _fingerprint_value(value, f"{path}[{index}]")
        for index, value in enumerate(materialized)
    )
    if len(set(normalized)) != len(normalized):
        _invalid("duplicate_fingerprint", path, "must not contain duplicates")
    return tuple(sorted(normalized))


def _typed_values(
    values: Iterable[Any],
    expected_type: type[Any],
    path: str,
    *,
    minimum: int,
    invalid_code: str,
) -> tuple[Any, ...]:
    """Materialize bounded values and enforce one exact component type."""
    materialized = _bounded_tuple(values, path, minimum=minimum)
    for index, value in enumerate(materialized):
        if not isinstance(value, expected_type):
            _invalid(
                invalid_code,
                f"{path}[{index}]",
                f"must be a {expected_type.__name__}",
            )
    return materialized


def _freeze_json(
    value: Any,
    path: str,
    *,
    depth: int,
    node_budget: list[int],
) -> Any:
    """Validate and recursively freeze one bounded canonical JSON value."""
    node_budget[0] += 1
    if node_budget[0] > MAX_JSON_NODES:
        _invalid(
            "json_node_budget_exceeded",
            path,
            f"must contain at most {MAX_JSON_NODES} JSON nodes",
        )

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -MAX_JSON_INTEGER <= value <= MAX_JSON_INTEGER:
            _invalid(
                "json_integer_out_of_range",
                path,
                "integer must fit the signed 64-bit metadata range",
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _invalid("non_finite_json_number", path, "number must be finite")
        return value
    if isinstance(value, str):
        if len(value) > 8_192:
            _invalid(
                "json_string_too_long",
                path,
                "string must contain at most 8192 characters",
            )
        return value

    if depth >= MAX_JSON_DEPTH:
        _invalid(
            "json_depth_exceeded",
            path,
            f"must not exceed {MAX_JSON_DEPTH} nested collections",
        )

    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for index, (key, item_value) in enumerate(value.items()):
            if index >= MAX_JSON_COLLECTION_VALUES:
                _invalid(
                    "json_collection_too_large",
                    path,
                    f"mapping must contain at most {MAX_JSON_COLLECTION_VALUES} values",
                )
            if not isinstance(key, str):
                _invalid("invalid_json_key", path, "mapping keys must be strings")
            normalized_key = _text_value(key, f"{path}.<key>", maximum=128)
            if normalized_key != key:
                _invalid(
                    "invalid_json_key",
                    path,
                    "mapping keys must not contain surrounding whitespace",
                )
            items.append(
                (
                    normalized_key,
                    _freeze_json(
                        item_value,
                        f"{path}.{normalized_key}",
                        depth=depth + 1,
                        node_budget=node_budget,
                    ),
                )
            )
        return MappingProxyType(dict(sorted(items)))

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        materialized: list[Any] = []
        for index, item_value in enumerate(value):
            if index >= MAX_JSON_COLLECTION_VALUES:
                _invalid(
                    "json_collection_too_large",
                    path,
                    f"array must contain at most {MAX_JSON_COLLECTION_VALUES} values",
                )
            materialized.append(
                _freeze_json(
                    item_value,
                    f"{path}[{index}]",
                    depth=depth + 1,
                    node_budget=node_budget,
                )
            )
        return tuple(materialized)

    _invalid(
        "unsupported_json_value",
        path,
        "must contain only canonical JSON-compatible values",
    )


def _thaw_json(value: Any) -> Any:
    """Return a fresh JSON-compatible value from recursively frozen metadata."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ConstructSpec:
    """One declared construct and the exact rubrics that operationalize it."""

    construct_id: str
    rubric_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize the construct identifier and exact rubric identities."""
        object.__setattr__(
            self,
            "construct_id",
            _identifier_value(self.construct_id, "construct_id"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical construct representation."""
        return {
            "construct_id": self.construct_id,
            "rubric_fingerprints": list(self.rubric_fingerprints),
        }


@dataclass(frozen=True)
class EnginePolicy:
    """Declare scoring engines and evidence/abstention requirements."""

    engine_ids: tuple[str, ...] = ()
    required_engine_ids: tuple[str, ...] = ()
    require_evidence: bool = True
    allow_abstention: bool = True

    def __post_init__(self) -> None:
        """Normalize engines and reject undeclared required engines."""
        engine_ids = _unique_identifiers(self.engine_ids, "engine_ids")
        required_engine_ids = _unique_identifiers(
            self.required_engine_ids,
            "required_engine_ids",
        )
        if not set(required_engine_ids).issubset(engine_ids):
            _invalid(
                "unknown_required_engine",
                "required_engine_ids",
                "every required engine must be declared in engine_ids",
            )
        object.__setattr__(self, "engine_ids", engine_ids)
        object.__setattr__(self, "required_engine_ids", required_engine_ids)
        object.__setattr__(
            self,
            "require_evidence",
            _boolean_value(self.require_evidence, "require_evidence"),
        )
        object.__setattr__(
            self,
            "allow_abstention",
            _boolean_value(self.allow_abstention, "allow_abstention"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical engine-policy representation."""
        return {
            "engine_ids": list(self.engine_ids),
            "required_engine_ids": list(self.required_engine_ids),
            "require_evidence": self.require_evidence,
            "allow_abstention": self.allow_abstention,
        }


@dataclass(frozen=True)
class CalibrationPolicy:
    """Declare the estimator family and minimum connected evidence contract."""

    model_id: str
    rubric_fingerprints: tuple[str, ...]
    minimum_observations_per_item: int = 2
    minimum_observations_per_rater: int = 2
    require_connected_design: bool = True

    def __post_init__(self) -> None:
        """Normalize the declared calibration contract."""
        object.__setattr__(self, "model_id", _identifier_value(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_observations_per_item",
            _integer_value(
                self.minimum_observations_per_item,
                "minimum_observations_per_item",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_observations_per_rater",
            _integer_value(
                self.minimum_observations_per_rater,
                "minimum_observations_per_rater",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "require_connected_design",
            _boolean_value(self.require_connected_design, "require_connected_design"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical calibration-policy representation."""
        return {
            "model_id": self.model_id,
            "rubric_fingerprints": list(self.rubric_fingerprints),
            "minimum_observations_per_item": self.minimum_observations_per_item,
            "minimum_observations_per_rater": self.minimum_observations_per_rater,
            "require_connected_design": self.require_connected_design,
        }


@dataclass(frozen=True)
class MetricGate:
    """One evidence-aware validation threshold and its declared scope."""

    metric_id: str
    direction: MetricDirection
    threshold: float
    minimum_evidence_count: int = 1
    group_ids: tuple[str, ...] = ()
    rubric_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize one validation gate without inventing policy defaults."""
        object.__setattr__(
            self,
            "metric_id",
            _identifier_value(self.metric_id, "metric_id"),
        )
        object.__setattr__(
            self,
            "direction",
            _enum_value(self.direction, MetricDirection, "direction"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_number(self.threshold, "threshold"),
        )
        object.__setattr__(
            self,
            "minimum_evidence_count",
            _integer_value(
                self.minimum_evidence_count,
                "minimum_evidence_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "group_ids",
            _unique_identifiers(self.group_ids, "group_ids"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical validation-gate representation."""
        return {
            "metric_id": self.metric_id,
            "direction": self.direction.value,
            "threshold": self.threshold,
            "minimum_evidence_count": self.minimum_evidence_count,
            "group_ids": list(self.group_ids),
            "rubric_fingerprints": list(self.rubric_fingerprints),
        }


@dataclass(frozen=True)
class ValidationPolicy:
    """Declare subgroup identities and ordered validation gates."""

    metric_gates: tuple[MetricGate, ...]
    declared_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize validation groups and reject duplicate metric identities."""
        groups = _unique_identifiers(self.declared_group_ids, "declared_group_ids")
        gates = _typed_values(
            self.metric_gates,
            MetricGate,
            "metric_gates",
            minimum=1,
            invalid_code="invalid_metric_gate_value",
        )
        metric_ids = tuple(gate.metric_id for gate in gates)
        if len(set(metric_ids)) != len(metric_ids):
            _invalid(
                "duplicate_metric_id",
                "metric_gates",
                "metric identifiers must be unique",
            )
        object.__setattr__(self, "declared_group_ids", groups)
        object.__setattr__(self, "metric_gates", gates)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical validation-policy representation."""
        return {
            "declared_group_ids": list(self.declared_group_ids),
            "metric_gates": [gate.to_dict() for gate in self.metric_gates],
        }


@dataclass(frozen=True)
class AdjudicationRule:
    """One transparent human-review routing rule and its declared scope."""

    rule_id: str
    threshold: float | None = None
    engine_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    rubric_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize one adjudication rule without interpreting its metric."""
        object.__setattr__(self, "rule_id", _identifier_value(self.rule_id, "rule_id"))
        if self.threshold is not None:
            object.__setattr__(
                self,
                "threshold",
                _finite_number(self.threshold, "threshold"),
            )
        object.__setattr__(
            self,
            "engine_ids",
            _unique_identifiers(self.engine_ids, "engine_ids"),
        )
        object.__setattr__(
            self,
            "group_ids",
            _unique_identifiers(self.group_ids, "group_ids"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical adjudication-rule representation."""
        return {
            "rule_id": self.rule_id,
            "threshold": self.threshold,
            "engine_ids": list(self.engine_ids),
            "group_ids": list(self.group_ids),
            "rubric_fingerprints": list(self.rubric_fingerprints),
        }


@dataclass(frozen=True)
class AdjudicationPolicy:
    """Declare ordered, transparent human-review routing rules."""

    rules: tuple[AdjudicationRule, ...]

    def __post_init__(self) -> None:
        """Reject empty, malformed, or duplicate adjudication rules."""
        rules = _typed_values(
            self.rules,
            AdjudicationRule,
            "rules",
            minimum=1,
            invalid_code="invalid_adjudication_rule_value",
        )
        rule_ids = tuple(rule.rule_id for rule in rules)
        if len(set(rule_ids)) != len(rule_ids):
            _invalid("duplicate_rule_id", "rules", "rule identifiers must be unique")
        object.__setattr__(self, "rules", rules)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical adjudication-policy representation."""
        return {"rules": [rule.to_dict() for rule in self.rules]}


@dataclass(frozen=True)
class MonitoringRule:
    """One versioned drift-monitoring rule and its bounded window."""

    rule_id: str
    direction: MetricDirection
    threshold: float
    window_size: int
    engine_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    rubric_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize one bounded monitoring rule."""
        object.__setattr__(self, "rule_id", _identifier_value(self.rule_id, "rule_id"))
        object.__setattr__(
            self,
            "direction",
            _enum_value(self.direction, MetricDirection, "direction"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_number(self.threshold, "threshold"),
        )
        object.__setattr__(
            self,
            "window_size",
            _integer_value(self.window_size, "window_size", minimum=1),
        )
        object.__setattr__(
            self,
            "engine_ids",
            _unique_identifiers(self.engine_ids, "engine_ids"),
        )
        object.__setattr__(
            self,
            "group_ids",
            _unique_identifiers(self.group_ids, "group_ids"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical monitoring-rule representation."""
        return {
            "rule_id": self.rule_id,
            "direction": self.direction.value,
            "threshold": self.threshold,
            "window_size": self.window_size,
            "engine_ids": list(self.engine_ids),
            "group_ids": list(self.group_ids),
            "rubric_fingerprints": list(self.rubric_fingerprints),
        }


@dataclass(frozen=True)
class MonitoringPolicy:
    """Declare ordered monitoring rules without silently inventing defaults."""

    rules: tuple[MonitoringRule, ...]

    def __post_init__(self) -> None:
        """Reject empty, malformed, or duplicate monitoring rules."""
        rules = _typed_values(
            self.rules,
            MonitoringRule,
            "rules",
            minimum=1,
            invalid_code="invalid_monitoring_rule_value",
        )
        rule_ids = tuple(rule.rule_id for rule in rules)
        if len(set(rule_ids)) != len(rule_ids):
            _invalid("duplicate_rule_id", "rules", "rule identifiers must be unique")
        object.__setattr__(self, "rules", rules)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical monitoring-policy representation."""
        return {"rules": [rule.to_dict() for rule in self.rules]}


def _validate_reference_set(
    references: Iterable[str],
    declared: set[str],
    *,
    code: str,
    path: str,
    noun: str,
) -> None:
    """Reject references outside one centrally declared identity set."""
    if not set(references).issubset(declared):
        _invalid(code, path, f"every referenced {noun} must be declared")


def _validate_policy_references(spec: "AssessmentSpec") -> None:
    """Validate all cross-policy engine, group, and rubric references."""
    rubrics = set(spec.rubric_fingerprints)
    engines = set(spec.engine_policy.engine_ids)
    groups = set(spec.validation_policy.declared_group_ids)

    _validate_reference_set(
        spec.calibration_policy.rubric_fingerprints,
        rubrics,
        code="unknown_policy_rubric",
        path="calibration_policy.rubric_fingerprints",
        noun="rubric fingerprint",
    )
    if set(spec.calibration_policy.rubric_fingerprints) != rubrics:
        _invalid(
            "calibration_rubric_coverage",
            "calibration_policy.rubric_fingerprints",
            "calibration policy must cover every assessment rubric",
        )

    scoped_values: tuple[tuple[str, Iterable[str], set[str], str, str], ...] = ()
    validation_scopes = tuple(
        (
            f"validation_policy.metric_gates[{index}]",
            gate,
        )
        for index, gate in enumerate(spec.validation_policy.metric_gates)
    )
    adjudication_scopes = tuple(
        (
            f"adjudication_policy.rules[{index}]",
            rule,
        )
        for index, rule in enumerate(spec.adjudication_policy.rules)
    )
    monitoring_scopes = tuple(
        (
            f"monitoring_policy.rules[{index}]",
            rule,
        )
        for index, rule in enumerate(spec.monitoring_policy.rules)
    )

    for path, scoped in validation_scopes + adjudication_scopes + monitoring_scopes:
        scoped_values = (
            (
                f"{path}.group_ids",
                scoped.group_ids,
                groups,
                "unknown_policy_group",
                "group identifier",
            ),
            (
                f"{path}.rubric_fingerprints",
                scoped.rubric_fingerprints,
                rubrics,
                "unknown_policy_rubric",
                "rubric fingerprint",
            ),
        )
        for scope_path, references, declared, code, noun in scoped_values:
            _validate_reference_set(
                references,
                declared,
                code=code,
                path=scope_path,
                noun=noun,
            )

    for path, scoped in adjudication_scopes + monitoring_scopes:
        _validate_reference_set(
            scoped.engine_ids,
            engines,
            code="unknown_policy_engine",
            path=f"{path}.engine_ids",
            noun="engine identifier",
        )


@dataclass(frozen=True)
class AssessmentSpec:
    """Versioned assessment contract bound to an exact rubric registry."""

    assessment_id: str
    assessment_version: str
    constructs: tuple[ConstructSpec, ...]
    rubric_fingerprints: tuple[str, ...]
    response_type: ResponseType
    engine_policy: EnginePolicy
    calibration_policy: CalibrationPolicy
    validation_policy: ValidationPolicy
    adjudication_policy: AdjudicationPolicy
    monitoring_policy: MonitoringPolicy
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    _rubric_registry: InitVar[tuple[RubricSpecification, ...] | None] = None
    _factory_token: InitVar[object | None] = None

    def __post_init__(
        self,
        _rubric_registry: tuple[RubricSpecification, ...] | None,
        _factory_token: object | None,
    ) -> None:
        """Seal construction and validate registry and policy composition."""
        if _factory_token is not _ASSESSMENT_SPEC_TOKEN:
            _invalid(
                "factory_required",
                "assessment_spec",
                "must be created by build_assessment_spec",
            )
        object.__setattr__(
            self,
            "assessment_id",
            _identifier_value(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "assessment_version",
            _semantic_version_value(self.assessment_version, "assessment_version"),
        )
        constructs = _typed_values(
            self.constructs,
            ConstructSpec,
            "constructs",
            minimum=1,
            invalid_code="invalid_construct_value",
        )
        construct_ids = tuple(construct.construct_id for construct in constructs)
        if len(set(construct_ids)) != len(construct_ids):
            _invalid(
                "duplicate_construct_id",
                "constructs",
                "construct identifiers must be unique",
            )
        object.__setattr__(
            self,
            "constructs",
            tuple(sorted(constructs, key=lambda value: value.construct_id)),
        )
        object.__setattr__(
            self,
            "rubric_fingerprints",
            _unique_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "response_type",
            _enum_value(self.response_type, ResponseType, "response_type"),
        )
        policy_fields = (
            ("engine_policy", self.engine_policy, EnginePolicy),
            ("calibration_policy", self.calibration_policy, CalibrationPolicy),
            ("validation_policy", self.validation_policy, ValidationPolicy),
            ("adjudication_policy", self.adjudication_policy, AdjudicationPolicy),
            ("monitoring_policy", self.monitoring_policy, MonitoringPolicy),
        )
        for path, value, expected_type in policy_fields:
            if not isinstance(value, expected_type):
                _invalid(
                    "invalid_policy_value",
                    path,
                    f"must be a {expected_type.__name__}",
                )
        if not isinstance(self.metadata, Mapping):
            _invalid("invalid_metadata_mapping", "metadata", "must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(
                self.metadata,
                "metadata",
                depth=0,
                node_budget=[0],
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_version_value(self.schema_version),
        )

        if _rubric_registry is None:
            _invalid(
                "factory_required",
                "rubric_registry",
                "must be supplied by build_assessment_spec",
            )
        registry = _typed_values(
            _rubric_registry,
            RubricSpecification,
            "rubric_registry",
            minimum=1,
            invalid_code="invalid_rubric_registry_value",
        )
        registry_fingerprints = tuple(rubric.fingerprint for rubric in registry)
        if len(set(registry_fingerprints)) != len(registry_fingerprints):
            _invalid(
                "duplicate_rubric_fingerprint",
                "rubric_registry",
                "rubric fingerprints must be unique",
            )
        registry_ids = tuple(rubric.rubric_id for rubric in registry)
        if len(set(registry_ids)) != len(registry_ids):
            _invalid(
                "duplicate_rubric_id",
                "rubric_registry",
                "rubric identifiers must be unique",
            )
        if set(registry_fingerprints) != set(self.rubric_fingerprints):
            _invalid(
                "rubric_registry_mismatch",
                "rubric_fingerprints",
                "must exactly match the supplied rubric registry",
            )

        registry_by_fingerprint = {
            rubric.fingerprint: rubric for rubric in registry
        }
        assigned: dict[str, str] = {}
        for construct in self.constructs:
            for fingerprint in construct.rubric_fingerprints:
                if fingerprint in assigned:
                    _invalid(
                        "rubric_assigned_multiple_times",
                        "constructs",
                        "each rubric fingerprint must belong to one construct",
                    )
                if fingerprint not in registry_by_fingerprint:
                    _invalid(
                        "unknown_construct_rubric",
                        "constructs",
                        "construct references must exist in the rubric registry",
                    )
                assigned[fingerprint] = construct.construct_id
        if set(assigned) != set(registry_fingerprints):
            _invalid(
                "rubric_registry_not_covered",
                "constructs",
                "construct declarations must cover every registry rubric",
            )
        for fingerprint, rubric in registry_by_fingerprint.items():
            if assigned[fingerprint] != rubric.construct_id:
                _invalid(
                    "rubric_construct_mismatch",
                    "constructs",
                    "rubric construct identities must match their declarations",
                )

        _validate_policy_references(self)

    @property
    def assessment_handle(self) -> str:
        """Return a descriptive 128-bit public handle for this exact contract."""
        return f"assessment_spec_{self.artifact_digest()[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh canonical JSON-compatible assessment representation."""
        return {
            "schema_version": self.schema_version,
            "assessment_version": self.assessment_version,
            "assessment_id": self.assessment_id,
            "constructs": [construct.to_dict() for construct in self.constructs],
            "rubric_fingerprints": list(self.rubric_fingerprints),
            "response_type": self.response_type.value,
            "engine_policy": self.engine_policy.to_dict(),
            "calibration_policy": self.calibration_policy.to_dict(),
            "validation_policy": self.validation_policy.to_dict(),
            "adjudication_policy": self.adjudication_policy.to_dict(),
            "monitoring_policy": self.monitoring_policy.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }

    def canonical_json(self) -> str:
        """Return the deterministic canonical JSON representation."""
        return _canonical_json(self.to_dict())

    def artifact_digest(self) -> str:
        """Return the SHA-256 identity of this complete assessment contract."""
        return _sha256_hex(self.to_dict())


def build_assessment_spec(
    *,
    assessment_id: str,
    assessment_version: str,
    constructs: Iterable[ConstructSpec],
    rubrics: Iterable[RubricSpecification],
    response_type: ResponseType,
    engine_policy: EnginePolicy,
    calibration_policy: CalibrationPolicy,
    validation_policy: ValidationPolicy,
    adjudication_policy: AdjudicationPolicy,
    monitoring_policy: MonitoringPolicy,
    metadata: Mapping[str, Any] | None = None,
) -> AssessmentSpec:
    """Build one assessment contract against an explicit exact-rubric registry."""
    registry = _typed_values(
        rubrics,
        RubricSpecification,
        "rubric_registry",
        minimum=1,
        invalid_code="invalid_rubric_registry_value",
    )
    registry = tuple(
        sorted(
            registry,
            key=lambda rubric: (
                rubric.construct_id,
                rubric.rubric_id,
                rubric.fingerprint,
            ),
        )
    )
    construct_values = _typed_values(
        constructs,
        ConstructSpec,
        "constructs",
        minimum=1,
        invalid_code="invalid_construct_value",
    )
    return AssessmentSpec(
        assessment_id=assessment_id,
        assessment_version=assessment_version,
        constructs=tuple(construct_values),
        rubric_fingerprints=tuple(rubric.fingerprint for rubric in registry),
        response_type=response_type,
        engine_policy=engine_policy,
        calibration_policy=calibration_policy,
        validation_policy=validation_policy,
        adjudication_policy=adjudication_policy,
        monitoring_policy=monitoring_policy,
        metadata={} if metadata is None else metadata,
        schema_version=SCHEMA_VERSION,
        _rubric_registry=registry,
        _factory_token=_ASSESSMENT_SPEC_TOKEN,
    )
