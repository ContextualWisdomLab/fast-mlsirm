"""Provider-neutral enterprise-issue source and evidence adapters.

The contracts retain only descriptive identifiers, bounded source-span offsets,
content fingerprints, and immutable metadata. They do not store source text,
extract semantic issues, estimate psychometric parameters, rank issues, infer
causal effects, or authorize automated decisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import operator
from typing import Any

from .._contract_safety import (
    artifact_digest,
    enum_value,
    freeze_metadata,
)
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    fingerprint,
    thaw_json_value,
)
from ..execution import EvidenceReference, EvidenceRole

MAX_ENTERPRISE_SOURCE_CHARACTERS = 100_000_000

_ENTERPRISE_SOURCE_TOKEN = object()
_EVIDENCE_SPAN_TOKEN = object()
_COUNTEREVIDENCE_TOKEN = object()


class EnterpriseSourceKind(str, Enum):
    """Supported provider-neutral enterprise source families."""

    REPORT_RECORD = "report_record"
    SALES_LEAD_RECORD = "sales_lead_record"
    CUSTOMER_COMMENT_RECORD = "customer_comment_record"
    CUSTOMER_COMPLAINT_RECORD = "customer_complaint_record"
    OTHER_SOURCE_RECORD = "other_source_record"


class EvidenceAssertionKind(str, Enum):
    """Epistemic role of one evidence span without collapsing it into sentiment."""

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
    if not 0 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be between 0 and {maximum}",
        )
    return int(normalized)


def _evidence_role(assertion_kind: EvidenceAssertionKind) -> EvidenceRole:
    """Map one epistemic assertion kind onto the shared evidence-role contract."""
    if assertion_kind is EvidenceAssertionKind.COUNTEREVIDENCE:
        return EvidenceRole.COUNTER
    if assertion_kind in {
        EvidenceAssertionKind.UNRESOLVED_AMBIGUITY,
        EvidenceAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
    }:
        return EvidenceRole.CONTEXT
    return EvidenceRole.SUPPORTING


@dataclass(frozen=True)
class EnterpriseSourceRecord(CanonicalContract):
    """Factory-sealed content-addressed enterprise source without source text."""

    source_id: str
    source_kind: EnterpriseSourceKind
    source_content_fingerprint: str
    source_revision_fingerprint: str
    source_character_count: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _source_token: InitVar[object | None] = None

    def __post_init__(self, _source_token: object | None) -> None:
        """Reject direct construction and normalize source provenance."""
        if _source_token is not _ENTERPRISE_SOURCE_TOKEN:
            raise assessment_error(
                "unverified_enterprise_source",
                "$",
                "use build_enterprise_source_record",
            )
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "source_kind",
            enum_value(self.source_kind, EnterpriseSourceKind, "source_kind"),
        )
        object.__setattr__(
            self,
            "source_content_fingerprint",
            fingerprint(self.source_content_fingerprint, "source_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_revision_fingerprint",
            fingerprint(
                self.source_revision_fingerprint,
                "source_revision_fingerprint",
            ),
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
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical source content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "source_content_fingerprint": self.source_content_fingerprint,
            "source_revision_fingerprint": self.source_revision_fingerprint,
            "source_character_count": self.source_character_count,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def source_record_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized source record."""
        return artifact_digest(self)

    @property
    def source_record_handle(self) -> str:
        """Return a descriptive 128-bit public source-record handle."""
        return f"enterprise_source_{self.source_record_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical source content and deterministic identities."""
        return {
            **self._content_dict(),
            "source_record_handle": self.source_record_handle,
            "source_record_fingerprint": self.source_record_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceSpanRecord(CanonicalContract):
    """Factory-sealed exact source span with an explicit epistemic role."""

    span_id: str
    source_id: str
    source_record_fingerprint: str
    assertion_kind: EvidenceAssertionKind
    content_fingerprint: str
    start_offset: int
    end_offset: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _span_token: InitVar[object | None] = None

    def __post_init__(self, _span_token: object | None) -> None:
        """Reject direct construction and normalize source-span provenance."""
        if _span_token is not _EVIDENCE_SPAN_TOKEN:
            raise assessment_error(
                "unverified_evidence_span",
                "$",
                "use build_evidence_span_record",
            )
        object.__setattr__(
            self,
            "span_id",
            descriptive_identifier(self.span_id, "span_id"),
        )
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "source_record_fingerprint",
            fingerprint(self.source_record_fingerprint, "source_record_fingerprint"),
        )
        object.__setattr__(
            self,
            "assertion_kind",
            enum_value(
                self.assertion_kind,
                EvidenceAssertionKind,
                "assertion_kind",
            ),
        )
        object.__setattr__(
            self,
            "content_fingerprint",
            fingerprint(self.content_fingerprint, "content_fingerprint"),
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
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical span content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "source_record_fingerprint": self.source_record_fingerprint,
            "assertion_kind": self.assertion_kind.value,
            "content_fingerprint": self.content_fingerprint,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def evidence_span_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized evidence span."""
        return artifact_digest(self)

    @property
    def evidence_span_handle(self) -> str:
        """Return a descriptive 128-bit public evidence-span handle."""
        return f"enterprise_evidence_{self.evidence_span_fingerprint[:32]}"

    @property
    def shared_evidence_reference(self) -> EvidenceReference:
        """Project the exact span into the existing shared evidence contract."""
        return EvidenceReference(
            source_id=self.source_id,
            span_id=self.span_id,
            content_fingerprint=self.content_fingerprint,
            evidence_role=_evidence_role(self.assertion_kind),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical span content and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_span_handle": self.evidence_span_handle,
            "evidence_span_fingerprint": self.evidence_span_fingerprint,
            "shared_evidence_reference": self.shared_evidence_reference.to_dict(),
        }


@dataclass(frozen=True)
class CounterevidenceRecord(CanonicalContract):
    """Factory-sealed counterevidence binding for one issue statement."""

    counterevidence_id: str
    issue_statement_fingerprint: str
    evidence_span_id: str
    evidence_span_fingerprint: str
    source_record_fingerprint: str
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
            "issue_statement_fingerprint",
            fingerprint(
                self.issue_statement_fingerprint,
                "issue_statement_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "evidence_span_id",
            descriptive_identifier(self.evidence_span_id, "evidence_span_id"),
        )
        object.__setattr__(
            self,
            "evidence_span_fingerprint",
            fingerprint(self.evidence_span_fingerprint, "evidence_span_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_record_fingerprint",
            fingerprint(self.source_record_fingerprint, "source_record_fingerprint"),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "counterevidence_id": self.counterevidence_id,
            "issue_statement_fingerprint": self.issue_statement_fingerprint,
            "evidence_span_id": self.evidence_span_id,
            "evidence_span_fingerprint": self.evidence_span_fingerprint,
            "source_record_fingerprint": self.source_record_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def counterevidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact counterevidence binding."""
        return artifact_digest(self)

    @property
    def counterevidence_handle(self) -> str:
        """Return a descriptive 128-bit public counterevidence handle."""
        return f"counterevidence_record_{self.counterevidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content and deterministic identities."""
        return {
            **self._content_dict(),
            "counterevidence_handle": self.counterevidence_handle,
            "counterevidence_fingerprint": self.counterevidence_fingerprint,
        }


def build_enterprise_source_record(
    *,
    source_id: str,
    source_kind: EnterpriseSourceKind | str,
    source_content_fingerprint: str,
    source_revision_fingerprint: str,
    source_character_count: int,
    metadata: Mapping[str, Any],
) -> EnterpriseSourceRecord:
    """Build one immutable source record without retaining source text."""
    return EnterpriseSourceRecord(
        source_id=source_id,
        source_kind=source_kind,
        source_content_fingerprint=source_content_fingerprint,
        source_revision_fingerprint=source_revision_fingerprint,
        source_character_count=source_character_count,
        metadata=metadata,
        _source_token=_ENTERPRISE_SOURCE_TOKEN,
    )


def build_evidence_span_record(
    *,
    span_id: str,
    source_record: EnterpriseSourceRecord,
    assertion_kind: EvidenceAssertionKind | str,
    content_fingerprint: str,
    start_offset: int,
    end_offset: int,
    metadata: Mapping[str, Any],
) -> EvidenceSpanRecord:
    """Build one exact source span and bind it to one verified source record."""
    if not isinstance(source_record, EnterpriseSourceRecord):
        raise assessment_error(
            "invalid_source_record",
            "$.source_record",
            "source_record must be an EnterpriseSourceRecord",
        )
    return EvidenceSpanRecord(
        span_id=span_id,
        source_id=source_record.source_id,
        source_record_fingerprint=source_record.source_record_fingerprint,
        assertion_kind=assertion_kind,
        content_fingerprint=content_fingerprint,
        start_offset=start_offset,
        end_offset=end_offset,
        metadata=metadata,
        _span_token=_EVIDENCE_SPAN_TOKEN,
    )


def build_counterevidence_record(
    *,
    counterevidence_id: str,
    issue_statement_fingerprint: str,
    evidence_span: EvidenceSpanRecord,
    metadata: Mapping[str, Any],
) -> CounterevidenceRecord:
    """Build a counterevidence binding from an explicit counterevidence span."""
    if not isinstance(evidence_span, EvidenceSpanRecord):
        raise assessment_error(
            "invalid_evidence_span",
            "$.evidence_span",
            "evidence_span must be an EvidenceSpanRecord",
        )
    if evidence_span.assertion_kind is not EvidenceAssertionKind.COUNTEREVIDENCE:
        raise assessment_error(
            "invalid_counterevidence_span",
            "$.evidence_span.assertion_kind",
            "counterevidence records require a counterevidence span",
        )
    return CounterevidenceRecord(
        counterevidence_id=counterevidence_id,
        issue_statement_fingerprint=issue_statement_fingerprint,
        evidence_span_id=evidence_span.span_id,
        evidence_span_fingerprint=evidence_span.evidence_span_fingerprint,
        source_record_fingerprint=evidence_span.source_record_fingerprint,
        metadata=metadata,
        _counterevidence_token=_COUNTEREVIDENCE_TOKEN,
    )
