"""Cross-reference-validated assessment specification graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
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

from ._validation import (
    MAX_ASSESSMENT_CONSTRUCTS,
    MAX_ASSESSMENT_RUBRICS,
    CanonicalContract,
    artifact_digest,
    freeze_metadata,
    sorted_fingerprints,
    thaw_json_value,
)
from .policies import (
    AdjudicationPolicy,
    CalibrationPolicy,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
)

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


@dataclass(frozen=True)
class ConstructSpec(CanonicalContract):
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
            sorted_fingerprints(
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

    _content_dict = to_dict


@dataclass(frozen=True)
class AssessmentSpec(CanonicalContract):
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
        """Reject direct construction and normalize the immutable public artifact."""
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
            sorted_fingerprints(
                self.rubric_fingerprints,
                "rubric_fingerprints",
                minimum=1,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
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
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def assessment_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable assessment contract."""
        return artifact_digest(self)

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
        maximum=MAX_ASSESSMENT_CONSTRUCTS,
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


def _materialize_rubrics(values: Iterable[Any]) -> tuple[tuple[str, RubricSpecification], ...]:
    """Return bounded typed rubrics with each fingerprint computed exactly once."""
    raw = _bounded_values(
        values,
        "rubrics",
        minimum=1,
        maximum=MAX_ASSESSMENT_RUBRICS,
    )
    keyed: list[tuple[str, RubricSpecification]] = []
    for index, rubric in enumerate(raw):
        if not isinstance(rubric, RubricSpecification):
            raise TypeError(f"rubrics[{index}] must be a RubricSpecification")
        keyed.append((rubric.fingerprint, rubric))
    return tuple(sorted(keyed, key=lambda entry: entry[0]))


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
    for fingerprint, rubric in normalized_rubrics:
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
        for fingerprint_index, rubric_fingerprint in enumerate(
            construct.rubric_fingerprints
        ):
            path = (
                f"$.constructs[{construct_index}].rubric_fingerprints"
                f"[{fingerprint_index}]"
            )
            rubric = rubrics_by_fingerprint.get(rubric_fingerprint)
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
            referenced.add(rubric_fingerprint)

    if set(rubrics_by_fingerprint).difference(referenced):
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
