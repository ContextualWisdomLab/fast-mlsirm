"""Provider-neutral enterprise issue adapters for governed scoring contracts.

The adapters retain descriptive identifiers, bounded counts, fingerprints, and
source-text-free span references. They do not store report, lead, complaint, or
customer text; infer semantic issues; estimate psychometric models; calculate
utility; or claim causal intervention effects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import operator
from typing import Any

from fast_mlsirm.rubric import RubricSpecification

from .._contract_safety import artifact_digest, bounded_values, freeze_metadata
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    enum_value,
    fingerprint,
    thaw_json_value,
)
from ..assessment import AssessmentSpec
from ..authorization import build_scoring_request
from ..execution import (
    EngineDescriptor,
    EvidenceReference,
    EvidenceRole,
    ObservationGranularity,
    ScoringEngine,
    ScoringRequest,
    ScoringResult,
)

MAX_ENTERPRISE_SOURCE_CHARACTERS = 100_000_000
MAX_ENTERPRISE_SOURCE_UNITS = 10_000_000
MAX_ENTERPRISE_EVIDENCE_SPANS = 64
MAX_ENTERPRISE_COUNTEREVIDENCE = 64
MAX_ENTERPRISE_PERSPECTIVES = 64
MAX_ENTERPRISE_INTERVENTIONS = 64

_SOURCE_TOKEN = object()
_EVIDENCE_TOKEN = object()
_COUNTEREVIDENCE_TOKEN = object()
_ISSUE_TOKEN = object()
_PERSPECTIVE_TOKEN = object()
_INTERVENTION_TOKEN = object()
_REQUEST_TOKEN = object()


class EnterpriseAssertionKind(str, Enum):
    """Epistemic role of one enterprise assertion without collapsing evidence."""

    DIRECT_FACT = "direct_fact"
    SUPPORTED_INFERENCE = "supported_inference"
    COUNTEREVIDENCE = "counterevidence"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    STAKEHOLDER_VALUE_JUDGMENT = "stakeholder_value_judgment"


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return one bounded nonnegative integer without Boolean coercion."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer between 0 and {maximum}",
        )
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer between 0 and {maximum}",
        ) from None
    if isinstance(normalized, bool) or not 0 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be between 0 and {maximum}",
        )
    return int(normalized)


def _typed_unique_values(
    values: Iterable[Any],
    *,
    name: str,
    expected_type: type[Any],
    fingerprint_name: str,
    maximum: int,
    minimum: int = 0,
) -> tuple[Any, ...]:
    """Return bounded, type-safe, fingerprint-unique values in stable order."""
    raw = bounded_values(values, name, minimum=minimum, maximum=maximum)
    for index, value in enumerate(raw):
        if not isinstance(value, expected_type):
            raise assessment_error(
                f"invalid_{name}",
                f"$.{name}[{index}]",
                f"{name} entries must be {expected_type.__name__} values",
            )
    identities = tuple(getattr(value, fingerprint_name) for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} entries must be unique",
        )
    return tuple(sorted(raw, key=lambda value: getattr(value, fingerprint_name)))


def _evidence_role(kind: EnterpriseAssertionKind) -> EvidenceRole:
    """Map an epistemic assertion kind to the shared evidence-role contract."""
    if kind is EnterpriseAssertionKind.COUNTEREVIDENCE:
        return EvidenceRole.COUNTER
    if kind in {
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
        EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
    }:
        return EvidenceRole.CONTEXT
    return EvidenceRole.SUPPORTING


@dataclass(frozen=True)
class EnterpriseSourceRecord(CanonicalContract):
    """Factory-sealed enterprise source provenance without source text."""

    source_id: str
    source_type_id: str
    source_content_fingerprint: str
    source_character_count: int
    source_unit_count: int
    subject_identifier_fingerprint: str | None
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _source_token: InitVar[object | None] = None

    def __post_init__(self, _source_token: object | None) -> None:
        """Reject direct construction and normalize source provenance."""
        if _source_token is not _SOURCE_TOKEN:
            raise assessment_error(
                "unverified_enterprise_source",
                "$",
                "use build_enterprise_source_record",
            )
        object.__setattr__(self, "source_id", descriptive_identifier(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_type_id",
            descriptive_identifier(self.source_type_id, "source_type_id"),
        )
        object.__setattr__(
            self,
            "source_content_fingerprint",
            fingerprint(self.source_content_fingerprint, "source_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_character_count",
            _nonnegative_integer(
                self.source_character_count,
                "source_character_count",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "source_unit_count",
            _nonnegative_integer(
                self.source_unit_count,
                "source_unit_count",
                MAX_ENTERPRISE_SOURCE_UNITS,
            ),
        )
        if self.subject_identifier_fingerprint is not None:
            object.__setattr__(
                self,
                "subject_identifier_fingerprint",
                fingerprint(
                    self.subject_identifier_fingerprint,
                    "subject_identifier_fingerprint",
                ),
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical source content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_type_id": self.source_type_id,
            "source_content_fingerprint": self.source_content_fingerprint,
            "source_character_count": self.source_character_count,
            "source_unit_count": self.source_unit_count,
            "subject_identifier_fingerprint": self.subject_identifier_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def source_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized source record."""
        return artifact_digest(self)

    @property
    def source_handle(self) -> str:
        """Return a descriptive 128-bit public source handle."""
        return f"enterprise_source_{self.source_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical source content and deterministic identities."""
        return {
            **self._content_dict(),
            "source_handle": self.source_handle,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceSpanRecord(CanonicalContract):
    """Factory-sealed source span with an explicit epistemic assertion kind."""

    evidence_reference: EvidenceReference
    source_fingerprint: str
    assertion_kind: EnterpriseAssertionKind
    start_offset: int
    end_offset: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _evidence_token: InitVar[object | None] = None

    def __post_init__(self, _evidence_token: object | None) -> None:
        """Reject direct construction and normalize evidence provenance."""
        if _evidence_token is not _EVIDENCE_TOKEN:
            raise assessment_error(
                "unverified_evidence_span",
                "$",
                "use build_evidence_span_record",
            )
        if not isinstance(self.evidence_reference, EvidenceReference):
            raise assessment_error(
                "invalid_evidence_reference",
                "$.evidence_reference",
                "evidence_reference must be an EvidenceReference",
            )
        object.__setattr__(
            self,
            "source_fingerprint",
            fingerprint(self.source_fingerprint, "source_fingerprint"),
        )
        object.__setattr__(
            self,
            "assertion_kind",
            enum_value(self.assertion_kind, EnterpriseAssertionKind, "assertion_kind"),
        )
        object.__setattr__(
            self,
            "start_offset",
            _nonnegative_integer(
                self.start_offset,
                "start_offset",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "end_offset",
            _nonnegative_integer(
                self.end_offset,
                "end_offset",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        if self.end_offset <= self.start_offset:
            raise assessment_error(
                "invalid_evidence_offsets",
                "$.end_offset",
                "end_offset must be greater than start_offset",
            )
        expected_role = _evidence_role(self.assertion_kind)
        if self.evidence_reference.evidence_role is not expected_role:
            raise assessment_error(
                "evidence_role_mismatch",
                "$.evidence_reference.evidence_role",
                "shared evidence role does not match the assertion kind",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical evidence content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "evidence_reference": self.evidence_reference.to_dict(),
            "source_fingerprint": self.source_fingerprint,
            "assertion_kind": self.assertion_kind.value,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized evidence span."""
        return artifact_digest(self)

    @property
    def evidence_handle(self) -> str:
        """Return a descriptive 128-bit public evidence-span handle."""
        return f"enterprise_evidence_{self.evidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical evidence content and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_handle": self.evidence_handle,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True)
class CounterevidenceRecord(CanonicalContract):
    """Factory-sealed counterevidence bound to one explicit claim identity."""

    counterevidence_id: str
    target_claim_id: str
    evidence_span: EvidenceSpanRecord
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _counterevidence_token: InitVar[object | None] = None

    def __post_init__(self, _counterevidence_token: object | None) -> None:
        """Reject direct construction and normalize counterevidence provenance."""
        if _counterevidence_token is not _COUNTEREVIDENCE_TOKEN:
            raise assessment_error(
                "unverified_counterevidence_record",
                "$",
                "use build_counterevidence_record",
            )
        object.__setattr__(
            self,
            "counterevidence_id",
            descriptive_identifier(self.counterevidence_id, "counterevidence_id"),
        )
        object.__setattr__(
            self,
            "target_claim_id",
            descriptive_identifier(self.target_claim_id, "target_claim_id"),
        )
        if not isinstance(self.evidence_span, EvidenceSpanRecord):
            raise assessment_error(
                "invalid_evidence_span",
                "$.evidence_span",
                "evidence_span must be an EvidenceSpanRecord",
            )
        if self.evidence_span.assertion_kind is not EnterpriseAssertionKind.COUNTEREVIDENCE:
            raise assessment_error(
                "invalid_counterevidence_kind",
                "$.evidence_span.assertion_kind",
                "counterevidence records require a counterevidence span",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "counterevidence_id": self.counterevidence_id,
            "target_claim_id": self.target_claim_id,
            "evidence_span": self.evidence_span.to_dict(),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def counterevidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact counterevidence record."""
        return artifact_digest(self)

    @property
    def counterevidence_handle(self) -> str:
        """Return a descriptive 128-bit public counterevidence handle."""
        return f"counterevidence_record_{self.counterevidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content and identities."""
        return {
            **self._content_dict(),
            "counterevidence_handle": self.counterevidence_handle,
            "counterevidence_fingerprint": self.counterevidence_fingerprint,
        }


@dataclass(frozen=True)
class AtomicIssueRecord(CanonicalContract):
    """Factory-sealed atomic issue with separated evidence and counterevidence."""

    issue_id: str
    issue_family_id: str
    domain_id: str
    issue_content_fingerprint: str
    issue_character_count: int
    issue_unit_count: int
    evidence_spans: tuple[EvidenceSpanRecord, ...]
    counterevidence_records: tuple[CounterevidenceRecord, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _issue_token: InitVar[object | None] = None

    def __post_init__(self, _issue_token: object | None) -> None:
        """Reject direct construction and normalize issue provenance."""
        if _issue_token is not _ISSUE_TOKEN:
            raise assessment_error(
                "unverified_atomic_issue",
                "$",
                "use build_atomic_issue_record",
            )
        for field_name in ("issue_id", "issue_family_id", "domain_id"):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "issue_content_fingerprint",
            fingerprint(self.issue_content_fingerprint, "issue_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "issue_character_count",
            _nonnegative_integer(
                self.issue_character_count,
                "issue_character_count",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "issue_unit_count",
            _nonnegative_integer(
                self.issue_unit_count,
                "issue_unit_count",
                MAX_ENTERPRISE_SOURCE_UNITS,
            ),
        )
        evidence = _typed_unique_values(
            self.evidence_spans,
            name="evidence_spans",
            expected_type=EvidenceSpanRecord,
            fingerprint_name="evidence_fingerprint",
            maximum=MAX_ENTERPRISE_EVIDENCE_SPANS,
        )
        counters = _typed_unique_values(
            self.counterevidence_records,
            name="counterevidence_records",
            expected_type=CounterevidenceRecord,
            fingerprint_name="counterevidence_fingerprint",
            maximum=MAX_ENTERPRISE_COUNTEREVIDENCE,
        )
        if not evidence and not counters:
            raise assessment_error(
                "missing_issue_evidence",
                "$.evidence_spans",
                "atomic issues require evidence or counterevidence",
            )
        if any(
            value.assertion_kind is EnterpriseAssertionKind.COUNTEREVIDENCE
            for value in evidence
        ):
            raise assessment_error(
                "misplaced_counterevidence_span",
                "$.evidence_spans",
                "counterevidence spans must use counterevidence_records",
            )
        object.__setattr__(self, "evidence_spans", evidence)
        object.__setattr__(self, "counterevidence_records", counters)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical issue content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "issue_id": self.issue_id,
            "issue_family_id": self.issue_family_id,
            "domain_id": self.domain_id,
            "issue_content_fingerprint": self.issue_content_fingerprint,
            "issue_character_count": self.issue_character_count,
            "issue_unit_count": self.issue_unit_count,
            "evidence_spans": [value.to_dict() for value in self.evidence_spans],
            "counterevidence_records": [
                value.to_dict() for value in self.counterevidence_records
            ],
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def issue_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized atomic issue."""
        return artifact_digest(self)

    @property
    def issue_handle(self) -> str:
        """Return a descriptive 128-bit public issue handle."""
        return f"atomic_issue_{self.issue_fingerprint[:32]}"

    @property
    def evidence_references(self) -> tuple[EvidenceReference, ...]:
        """Return all shared evidence references in deterministic order."""
        references = [value.evidence_reference for value in self.evidence_spans]
        references.extend(
            value.evidence_span.evidence_reference
            for value in self.counterevidence_records
        )
        return tuple(sorted(references, key=lambda value: value.evidence_fingerprint))

    def to_dict(self) -> dict[str, Any]:
        """Return canonical issue content and deterministic identities."""
        return {
            **self._content_dict(),
            "issue_handle": self.issue_handle,
            "issue_fingerprint": self.issue_fingerprint,
        }


@dataclass(frozen=True)
class StakeholderPerspective(CanonicalContract):
    """Factory-sealed stakeholder value judgment bound to one exact issue."""

    perspective_id: str
    stakeholder_group_id: str
    issue_fingerprint: str
    value_judgment_fingerprint: str
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _perspective_token: InitVar[object | None] = None

    def __post_init__(self, _perspective_token: object | None) -> None:
        """Reject direct construction and normalize perspective provenance."""
        if _perspective_token is not _PERSPECTIVE_TOKEN:
            raise assessment_error(
                "unverified_stakeholder_perspective",
                "$",
                "use build_stakeholder_perspective",
            )
        object.__setattr__(
            self,
            "perspective_id",
            descriptive_identifier(self.perspective_id, "perspective_id"),
        )
        object.__setattr__(
            self,
            "stakeholder_group_id",
            descriptive_identifier(self.stakeholder_group_id, "stakeholder_group_id"),
        )
        object.__setattr__(
            self,
            "issue_fingerprint",
            fingerprint(self.issue_fingerprint, "issue_fingerprint"),
        )
        object.__setattr__(
            self,
            "value_judgment_fingerprint",
            fingerprint(
                self.value_judgment_fingerprint,
                "value_judgment_fingerprint",
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical perspective content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "perspective_id": self.perspective_id,
            "stakeholder_group_id": self.stakeholder_group_id,
            "issue_fingerprint": self.issue_fingerprint,
            "value_judgment_fingerprint": self.value_judgment_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def perspective_fingerprint(self) -> str:
        """Return SHA-256 over the exact stakeholder perspective."""
        return artifact_digest(self)

    @property
    def perspective_handle(self) -> str:
        """Return a descriptive 128-bit public perspective handle."""
        return f"stakeholder_perspective_{self.perspective_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical perspective content and deterministic identities."""
        return {
            **self._content_dict(),
            "perspective_handle": self.perspective_handle,
            "perspective_fingerprint": self.perspective_fingerprint,
        }


@dataclass(frozen=True)
class CandidateIntervention(CanonicalContract):
    """Factory-sealed intervention option without inferred causal effect."""

    intervention_id: str
    intervention_family_id: str
    issue_fingerprint: str
    action_content_fingerprint: str
    decision_policy_id: str | None
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _intervention_token: InitVar[object | None] = None

    def __post_init__(self, _intervention_token: object | None) -> None:
        """Reject direct construction and normalize intervention provenance."""
        if _intervention_token is not _INTERVENTION_TOKEN:
            raise assessment_error(
                "unverified_candidate_intervention",
                "$",
                "use build_candidate_intervention",
            )
        object.__setattr__(
            self,
            "intervention_id",
            descriptive_identifier(self.intervention_id, "intervention_id"),
        )
        object.__setattr__(
            self,
            "intervention_family_id",
            descriptive_identifier(self.intervention_family_id, "intervention_family_id"),
        )
        object.__setattr__(
            self,
            "issue_fingerprint",
            fingerprint(self.issue_fingerprint, "issue_fingerprint"),
        )
        object.__setattr__(
            self,
            "action_content_fingerprint",
            fingerprint(self.action_content_fingerprint, "action_content_fingerprint"),
        )
        if self.decision_policy_id is not None:
            object.__setattr__(
                self,
                "decision_policy_id",
                descriptive_identifier(self.decision_policy_id, "decision_policy_id"),
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical intervention content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "intervention_id": self.intervention_id,
            "intervention_family_id": self.intervention_family_id,
            "issue_fingerprint": self.issue_fingerprint,
            "action_content_fingerprint": self.action_content_fingerprint,
            "decision_policy_id": self.decision_policy_id,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def intervention_fingerprint(self) -> str:
        """Return SHA-256 over the exact intervention option."""
        return artifact_digest(self)

    @property
    def intervention_handle(self) -> str:
        """Return a descriptive 128-bit public intervention handle."""
        return f"candidate_intervention_{self.intervention_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical intervention content and deterministic identities."""
        return {
            **self._content_dict(),
            "intervention_handle": self.intervention_handle,
            "intervention_fingerprint": self.intervention_fingerprint,
        }


@dataclass(frozen=True)
class EnterpriseIssueScoringRequest(CanonicalContract):
    """Enterprise adapter containing one authoritative shared scoring request."""

    scoring_request: ScoringRequest
    issue_fingerprint: str
    evidence_references: tuple[EvidenceReference, ...]
    perspective_fingerprints: tuple[str, ...]
    intervention_fingerprints: tuple[str, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _request_token: InitVar[object | None] = None

    def __post_init__(self, _request_token: object | None) -> None:
        """Reject direct construction and normalize wrapper provenance."""
        if _request_token is not _REQUEST_TOKEN:
            raise assessment_error(
                "unverified_enterprise_issue_request",
                "$",
                "use build_enterprise_issue_scoring_request",
            )
        if not isinstance(self.scoring_request, ScoringRequest):
            raise assessment_error(
                "invalid_scoring_request",
                "$.scoring_request",
                "scoring_request must be a ScoringRequest",
            )
        object.__setattr__(
            self,
            "issue_fingerprint",
            fingerprint(self.issue_fingerprint, "issue_fingerprint"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            tuple(sorted(self.evidence_references, key=lambda value: value.evidence_fingerprint)),
        )
        object.__setattr__(
            self,
            "perspective_fingerprints",
            tuple(sorted(fingerprint(value, "perspective_fingerprint") for value in self.perspective_fingerprints)),
        )
        object.__setattr__(
            self,
            "intervention_fingerprints",
            tuple(sorted(fingerprint(value, "intervention_fingerprint") for value in self.intervention_fingerprints)),
        )
        object.__setattr__(self, "schema_version", assessment_schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical request-wrapper content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "scoring_request": self.scoring_request.to_dict(),
            "issue_fingerprint": self.issue_fingerprint,
            "evidence_references": [value.to_dict() for value in self.evidence_references],
            "perspective_fingerprints": list(self.perspective_fingerprints),
            "intervention_fingerprints": list(self.intervention_fingerprints),
        }

    @property
    def request_fingerprint(self) -> str:
        """Return SHA-256 over the complete enterprise request adapter."""
        return artifact_digest(self)

    @property
    def request_handle(self) -> str:
        """Return a descriptive 128-bit public request handle."""
        return f"enterprise_issue_request_{self.request_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical request content and deterministic identities."""
        return {
            **self._content_dict(),
            "request_handle": self.request_handle,
            "request_fingerprint": self.request_fingerprint,
        }


def build_enterprise_source_record(
    *,
    source_id: str,
    source_type_id: str,
    source_content_fingerprint: str,
    source_character_count: int,
    source_unit_count: int,
    subject_identifier_fingerprint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EnterpriseSourceRecord:
    """Build one immutable source record without retaining source text."""
    return EnterpriseSourceRecord(
        source_id=source_id,
        source_type_id=source_type_id,
        source_content_fingerprint=source_content_fingerprint,
        source_character_count=source_character_count,
        source_unit_count=source_unit_count,
        subject_identifier_fingerprint=subject_identifier_fingerprint,
        metadata={} if metadata is None else metadata,
        _source_token=_SOURCE_TOKEN,
    )


def build_evidence_span_record(
    *,
    source: EnterpriseSourceRecord,
    span_id: str,
    content_fingerprint: str,
    assertion_kind: EnterpriseAssertionKind | str,
    start_offset: int,
    end_offset: int,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceSpanRecord:
    """Build one exact source span and its shared evidence reference."""
    if not isinstance(source, EnterpriseSourceRecord):
        raise assessment_error(
            "invalid_enterprise_source",
            "$.source",
            "source must be an EnterpriseSourceRecord",
        )
    normalized_kind = enum_value(
        assertion_kind,
        EnterpriseAssertionKind,
        "assertion_kind",
    )
    normalized_start = _nonnegative_integer(
        start_offset,
        "start_offset",
        source.source_character_count,
    )
    normalized_end = _nonnegative_integer(
        end_offset,
        "end_offset",
        source.source_character_count,
    )
    if normalized_end <= normalized_start:
        raise assessment_error(
            "invalid_evidence_offsets",
            "$.end_offset",
            "end_offset must be greater than start_offset",
        )
    reference = EvidenceReference(
        source_id=source.source_id,
        span_id=span_id,
        content_fingerprint=content_fingerprint,
        evidence_role=_evidence_role(normalized_kind),
    )
    return EvidenceSpanRecord(
        evidence_reference=reference,
        source_fingerprint=source.source_fingerprint,
        assertion_kind=normalized_kind,
        start_offset=normalized_start,
        end_offset=normalized_end,
        metadata={} if metadata is None else metadata,
        _evidence_token=_EVIDENCE_TOKEN,
    )


def build_counterevidence_record(
    *,
    counterevidence_id: str,
    target_claim_id: str,
    evidence_span: EvidenceSpanRecord,
    metadata: Mapping[str, Any] | None = None,
) -> CounterevidenceRecord:
    """Build one counterevidence record bound to a declared claim identity."""
    return CounterevidenceRecord(
        counterevidence_id=counterevidence_id,
        target_claim_id=target_claim_id,
        evidence_span=evidence_span,
        metadata={} if metadata is None else metadata,
        _counterevidence_token=_COUNTEREVIDENCE_TOKEN,
    )


def build_atomic_issue_record(
    *,
    issue_id: str,
    issue_family_id: str,
    domain_id: str,
    issue_content_fingerprint: str,
    issue_character_count: int,
    issue_unit_count: int,
    evidence_spans: Iterable[EvidenceSpanRecord] = (),
    counterevidence_records: Iterable[CounterevidenceRecord] = (),
    metadata: Mapping[str, Any] | None = None,
) -> AtomicIssueRecord:
    """Build one atomic issue while preserving evidence-role separation."""
    return AtomicIssueRecord(
        issue_id=issue_id,
        issue_family_id=issue_family_id,
        domain_id=domain_id,
        issue_content_fingerprint=issue_content_fingerprint,
        issue_character_count=issue_character_count,
        issue_unit_count=issue_unit_count,
        evidence_spans=tuple(evidence_spans),
        counterevidence_records=tuple(counterevidence_records),
        metadata={} if metadata is None else metadata,
        _issue_token=_ISSUE_TOKEN,
    )


def build_stakeholder_perspective(
    *,
    perspective_id: str,
    stakeholder_group_id: str,
    issue: AtomicIssueRecord,
    value_judgment_fingerprint: str,
    metadata: Mapping[str, Any] | None = None,
) -> StakeholderPerspective:
    """Build one stakeholder-specific value judgment for an exact issue."""
    if not isinstance(issue, AtomicIssueRecord):
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an AtomicIssueRecord",
        )
    return StakeholderPerspective(
        perspective_id=perspective_id,
        stakeholder_group_id=stakeholder_group_id,
        issue_fingerprint=issue.issue_fingerprint,
        value_judgment_fingerprint=value_judgment_fingerprint,
        metadata={} if metadata is None else metadata,
        _perspective_token=_PERSPECTIVE_TOKEN,
    )


def build_candidate_intervention(
    *,
    intervention_id: str,
    intervention_family_id: str,
    issue: AtomicIssueRecord,
    action_content_fingerprint: str,
    decision_policy_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateIntervention:
    """Build one intervention option without estimating its causal effect."""
    if not isinstance(issue, AtomicIssueRecord):
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an AtomicIssueRecord",
        )
    return CandidateIntervention(
        intervention_id=intervention_id,
        intervention_family_id=intervention_family_id,
        issue_fingerprint=issue.issue_fingerprint,
        action_content_fingerprint=action_content_fingerprint,
        decision_policy_id=decision_policy_id,
        metadata={} if metadata is None else metadata,
        _intervention_token=_INTERVENTION_TOKEN,
    )


def build_enterprise_issue_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    issue: AtomicIssueRecord,
    occasion_id: str,
    task_id: str,
    task_revision_fingerprint: str,
    task_family_id: str,
    criterion_ids: Iterable[str],
    stakeholder_perspectives: Iterable[StakeholderPerspective] = (),
    candidate_interventions: Iterable[CandidateIntervention] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EnterpriseIssueScoringRequest:
    """Compile one enterprise issue into the authoritative scoring request."""
    if not isinstance(issue, AtomicIssueRecord):
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an AtomicIssueRecord",
        )
    perspectives = _typed_unique_values(
        stakeholder_perspectives,
        name="stakeholder_perspectives",
        expected_type=StakeholderPerspective,
        fingerprint_name="perspective_fingerprint",
        maximum=MAX_ENTERPRISE_PERSPECTIVES,
    )
    interventions = _typed_unique_values(
        candidate_interventions,
        name="candidate_interventions",
        expected_type=CandidateIntervention,
        fingerprint_name="intervention_fingerprint",
        maximum=MAX_ENTERPRISE_INTERVENTIONS,
    )
    for index, value in enumerate(perspectives):
        if value.issue_fingerprint != issue.issue_fingerprint:
            raise assessment_error(
                "perspective_issue_mismatch",
                f"$.stakeholder_perspectives[{index}].issue_fingerprint",
                "stakeholder perspective is not bound to the supplied issue",
            )
    for index, value in enumerate(interventions):
        if value.issue_fingerprint != issue.issue_fingerprint:
            raise assessment_error(
                "intervention_issue_mismatch",
                f"$.candidate_interventions[{index}].issue_fingerprint",
                "candidate intervention is not bound to the supplied issue",
            )
    request_metadata = {
        "enterprise_issue_fingerprint": issue.issue_fingerprint,
        "enterprise_issue_family_id": issue.issue_family_id,
        "enterprise_domain_id": issue.domain_id,
        "enterprise_evidence_fingerprints": [
            value.evidence_fingerprint for value in issue.evidence_spans
        ],
        "enterprise_counterevidence_fingerprints": [
            value.counterevidence_fingerprint
            for value in issue.counterevidence_records
        ],
        "enterprise_perspective_fingerprints": [
            value.perspective_fingerprint for value in perspectives
        ],
        "enterprise_intervention_fingerprints": [
            value.intervention_fingerprint for value in interventions
        ],
        "enterprise_adapter_metadata": {} if metadata is None else metadata,
    }
    shared_request = build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id=issue.issue_id,
        response_id=f"{issue.issue_id}_record",
        task_id=task_id,
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=task_family_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=issue.issue_content_fingerprint,
        response_character_count=issue.issue_character_count,
        response_unit_count=issue.issue_unit_count,
        metadata=request_metadata,
    )
    return EnterpriseIssueScoringRequest(
        scoring_request=shared_request,
        issue_fingerprint=issue.issue_fingerprint,
        evidence_references=issue.evidence_references,
        perspective_fingerprints=tuple(
            value.perspective_fingerprint for value in perspectives
        ),
        intervention_fingerprints=tuple(
            value.intervention_fingerprint for value in interventions
        ),
        _request_token=_REQUEST_TOKEN,
    )


def score_enterprise_issue_request(
    engine: ScoringEngine,
    request: EnterpriseIssueScoringRequest,
) -> ScoringResult:
    """Execute an enterprise adapter through the shared scoring-engine protocol."""
    if not isinstance(request, EnterpriseIssueScoringRequest):
        raise assessment_error(
            "invalid_enterprise_issue_request",
            "$.request",
            "request must be an EnterpriseIssueScoringRequest",
        )
    if not isinstance(engine, ScoringEngine):
        raise assessment_error(
            "invalid_scoring_engine",
            "$.engine",
            "engine must satisfy the ScoringEngine protocol",
        )
    result = engine.score(request.scoring_request)
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "engine must return a ScoringResult",
        )
    descriptor = engine.descriptor
    if not isinstance(descriptor, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine.descriptor",
            "engine descriptor must be an EngineDescriptor",
        )
    if result.request_fingerprint != request.scoring_request.request_fingerprint:
        raise assessment_error(
            "enterprise_result_request_mismatch",
            "$.result.request_fingerprint",
            "engine result does not match the enterprise scoring request",
        )
    if result.engine_fingerprint != descriptor.engine_fingerprint:
        raise assessment_error(
            "enterprise_result_engine_mismatch",
            "$.result.engine_fingerprint",
            "engine result does not match the engine descriptor",
        )
    return result


__all__ = [
    "AtomicIssueRecord",
    "CandidateIntervention",
    "CounterevidenceRecord",
    "EnterpriseAssertionKind",
    "EnterpriseIssueScoringRequest",
    "EnterpriseSourceRecord",
    "EvidenceSpanRecord",
    "MAX_ENTERPRISE_COUNTEREVIDENCE",
    "MAX_ENTERPRISE_EVIDENCE_SPANS",
    "MAX_ENTERPRISE_INTERVENTIONS",
    "MAX_ENTERPRISE_PERSPECTIVES",
    "MAX_ENTERPRISE_SOURCE_CHARACTERS",
    "MAX_ENTERPRISE_SOURCE_UNITS",
    "StakeholderPerspective",
    "build_atomic_issue_record",
    "build_candidate_intervention",
    "build_counterevidence_record",
    "build_enterprise_issue_scoring_request",
    "build_enterprise_source_record",
    "build_evidence_span_record",
    "build_stakeholder_perspective",
    "score_enterprise_issue_request",
]
