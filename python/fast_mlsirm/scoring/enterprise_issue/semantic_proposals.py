"""Validate untrusted semantic issue proposals against exact enterprise sources.

The module is a provider-neutral trust boundary. Providers propose primitive
issue and span mappings over transient source text; package code replays the
source identity and reconstructs only the existing enterprise evidence records.
It stores no raw source or issue text and performs no scoring, calibration,
ranking, utility, causal, sentiment, or queue arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import operator
from typing import Any, Protocol, runtime_checkable

from .._contract_safety import (
    descriptive_identifier,
    enum_value,
    freeze_metadata,
)
from .._validation import (
    AssessmentSpecError,
    assessment_error,
    fingerprint,
    thaw_json_value,
)
from .semantic import (
    StaticEnterpriseIssueExtractor,
    extract_enterprise_atomic_issues,
)
from .contracts import (
    MAX_ENTERPRISE_ISSUE_EVIDENCE,
    MAX_ENTERPRISE_ISSUE_SOURCES,
    MAX_ENTERPRISE_SOURCE_CHARACTERS,
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    StakeholderPerspective,
)

MAX_SEMANTIC_ISSUE_PROPOSALS = 64
MAX_SEMANTIC_ASSERTIONS_PER_ISSUE = MAX_ENTERPRISE_ISSUE_EVIDENCE
MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS = 4_096

_PROPOSAL_KEYS = frozenset(
    {
        "issue_id",
        "issue_family_id",
        "issue_statement",
        "assertions",
        "metadata",
    }
)
_ASSERTION_KEYS = frozenset(
    {
        "source_id",
        "start_offset",
        "end_offset",
        "assertion_kind",
        "stakeholder_id",
        "metadata",
    }
)
_MANAGED_METADATA_KEYS = frozenset(
    {
        "semantic_provider_revision_fingerprint",
        "semantic_assertion_fingerprints",
        "semantic_perspective_fingerprints",
    }
)
_RESERVED_METADATA_TOKENS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "prompt",
    "secret",
    "source_text",
    "token",
)


@runtime_checkable
class EnterpriseSemanticIssueProvider(Protocol):
    """Provider-neutral semantic issue proposal interface.

    Implementations may call local or remote models outside this package. The
    stable boundary treats every returned value as untrusted primitive data.
    """

    provider_revision_fingerprint: str

    def propose(
        self,
        *,
        sources: tuple[tuple[EnterpriseSourceRecord, str], ...],
    ) -> Iterable[Mapping[str, Any]]:
        """Return primitive semantic issue proposals for exact source revisions."""
        ...


@dataclass(frozen=True)
class OfflineSemanticIssueFixtureProvider:
    """Deterministic disconnected provider backed by primitive fixture proposals."""

    provider_revision_fingerprint: str
    proposals: Any

    def __post_init__(self) -> None:
        """Normalize provider identity and materialize one repeatable fixture."""
        object.__setattr__(
            self,
            "provider_revision_fingerprint",
            fingerprint(
                self.provider_revision_fingerprint,
                "provider_revision_fingerprint",
            ),
        )
        if isinstance(self.proposals, (str, bytes, bytearray)):
            raise assessment_error(
                "invalid_semantic_issue_proposals",
                "$.proposals",
                "proposals must be a collection",
            )
        try:
            materialized = tuple(self.proposals)
        except AssessmentSpecError:
            raise
        except Exception:
            raise assessment_error(
                "invalid_semantic_issue_proposals",
                "$.proposals",
                "proposals could not be materialized safely",
            ) from None
        object.__setattr__(self, "proposals", materialized)

    def propose(
        self,
        *,
        sources: tuple[tuple[EnterpriseSourceRecord, str], ...],
    ) -> tuple[Any, ...]:
        """Return a fresh primitive fixture without inspecting source content."""
        del sources
        return deepcopy(self.proposals)


def _bounded_collection(
    values: Any,
    *,
    path: str,
    maximum: int,
    invalid_code: str,
    too_many_code: str,
    missing_code: str | None = None,
) -> tuple[Any, ...]:
    """Materialize one iterable with caller-specific stable resource errors."""
    if isinstance(values, (str, bytes, bytearray)):
        raise assessment_error(
            invalid_code,
            path,
            "value must be a bounded collection",
        )
    try:
        iterator = iter(values)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            invalid_code,
            path,
            "value must be a bounded collection",
        ) from None
    output: list[Any] = []
    try:
        for index, value in enumerate(iterator):
            if index >= maximum:
                raise assessment_error(
                    too_many_code,
                    path,
                    f"collection must contain at most {maximum} values",
                )
            output.append(value)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            invalid_code,
            path,
            "collection could not be materialized safely",
        ) from None
    if not output and missing_code is not None:
        raise assessment_error(
            missing_code,
            path,
            "collection must contain at least one value",
        )
    return tuple(output)


def _offset(value: Any, name: str, path: str) -> int:
    """Return one bounded nonnegative Unicode-code-point offset."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a nonnegative integer",
        )
    try:
        normalized = operator.index(value)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a nonnegative integer",
        ) from None
    if isinstance(normalized, bool) or not 0 <= normalized <= MAX_ENTERPRISE_SOURCE_CHARACTERS:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be between 0 and {MAX_ENTERPRISE_SOURCE_CHARACTERS}",
        )
    return int(normalized)


def _reserved_metadata_key(value: Any) -> bool:
    """Return whether one normalized metadata key names sensitive content."""
    if type(value) is not str:
        return False
    normalized = value.casefold()
    return any(token in normalized for token in _RESERVED_METADATA_TOKENS)


def _contains_reserved_metadata(value: Any) -> bool:
    """Return whether a normalized primitive tree contains a reserved key."""
    if isinstance(value, dict):
        return any(
            _reserved_metadata_key(key) or _contains_reserved_metadata(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_reserved_metadata(child) for child in value)
    return False


def _semantic_metadata(
    value: Any,
    *,
    path: str,
    allow_managed: bool = False,
) -> dict[str, Any]:
    """Return bounded primitive metadata without source or credential fields."""
    if not isinstance(value, Mapping):
        raise assessment_error(
            "invalid_semantic_metadata",
            path,
            "semantic metadata must be a mapping",
        )
    try:
        frozen = freeze_metadata(value)
    except AssessmentSpecError as error:
        if error.code == "sensitive_metadata_field":
            raise assessment_error(
                "reserved_semantic_metadata",
                path,
                "semantic metadata cannot contain source or credential fields",
            ) from None
        raise
    primitive = thaw_json_value(frozen)
    if type(primitive) is not dict:
        raise assessment_error(
            "invalid_semantic_metadata",
            path,
            "semantic metadata must be a mapping",
        )
    if _contains_reserved_metadata(primitive):
        raise assessment_error(
            "reserved_semantic_metadata",
            path,
            "semantic metadata cannot contain source or credential fields",
        )
    if not allow_managed and any(key in primitive for key in _MANAGED_METADATA_KEYS):
        raise assessment_error(
            "reserved_semantic_metadata",
            path,
            "semantic provenance metadata is package-managed",
        )
    return primitive


def _source_values(values: Any) -> tuple[EnterpriseSourceRecord, ...]:
    """Return exact unique source records in deterministic fingerprint order."""
    raw = _bounded_collection(
        values,
        path="$.enterprise_sources",
        maximum=MAX_ENTERPRISE_ISSUE_SOURCES,
        invalid_code="invalid_enterprise_sources",
        too_many_code="too_many_enterprise_sources",
        missing_code="missing_enterprise_sources",
    )
    for index, value in enumerate(raw):
        if type(value) is not EnterpriseSourceRecord:
            raise assessment_error(
                "invalid_enterprise_sources",
                f"$.enterprise_sources[{index}]",
                "enterprise sources must be exact EnterpriseSourceRecord values",
            )
    source_ids = tuple(value.source_id for value in raw)
    if len(set(source_ids)) != len(source_ids):
        raise assessment_error(
            "duplicate_enterprise_source_ids",
            "$.enterprise_sources",
            "enterprise source identifiers must be unique",
        )
    fingerprints = tuple(value.source_record_fingerprint for value in raw)
    if len(set(fingerprints)) != len(fingerprints):
        raise assessment_error(
            "duplicate_enterprise_source_records",
            "$.enterprise_sources",
            "enterprise source revisions must be unique",
        )
    return tuple(sorted(raw, key=lambda value: value.source_record_fingerprint))


def _verified_sources(
    values: Any,
    source_text_by_id: Any,
) -> tuple[tuple[EnterpriseSourceRecord, str], ...]:
    """Replay exact transient source text before any provider execution."""
    records = _source_values(values)
    if not isinstance(source_text_by_id, Mapping):
        raise assessment_error(
            "invalid_enterprise_source_text_mapping",
            "$.source_text_by_id",
            "source_text_by_id must be a mapping",
        )
    expected_keys = {record.source_id for record in records}
    try:
        actual_keys = set(source_text_by_id)
    except Exception:
        raise assessment_error(
            "invalid_enterprise_source_text_mapping",
            "$.source_text_by_id",
            "source text keys could not be inspected safely",
        ) from None
    if actual_keys != expected_keys:
        raise assessment_error(
            "enterprise_source_text_keys_mismatch",
            "$.source_text_by_id",
            "source text keys must exactly match enterprise source identifiers",
        )
    output: list[tuple[EnterpriseSourceRecord, str]] = []
    for record in records:
        try:
            source_text = source_text_by_id[record.source_id]
        except Exception:
            raise assessment_error(
                "invalid_enterprise_source_text_mapping",
                "$.source_text_by_id",
                "source text could not be read safely",
            ) from None
        if type(source_text) is not str:
            raise assessment_error(
                "invalid_enterprise_source_text",
                f"$.source_text_by_id.{record.source_id}",
                "enterprise source text must be a string",
            )
        if len(source_text) != record.source_character_count:
            raise assessment_error(
                "enterprise_source_character_count_mismatch",
                f"$.source_text_by_id.{record.source_id}",
                "source text does not match the declared character count",
            )
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if digest != record.source_content_fingerprint:
            raise assessment_error(
                "enterprise_source_content_mismatch",
                f"$.source_text_by_id.{record.source_id}",
                "source text does not match the declared content fingerprint",
            )
        output.append((record, source_text))
    return tuple(output)


def _proposal_mapping(value: Any, index: int) -> dict[str, Any]:
    """Return one exact primitive issue proposal mapping."""
    path = f"$.semantic_issue_proposals[{index}]"
    if type(value) is not dict or set(value) != _PROPOSAL_KEYS:
        raise assessment_error(
            "invalid_semantic_issue_proposal",
            path,
            "semantic issue proposals must contain the exact public field set",
        )
    return value


def _assertion_mapping(value: Any, proposal_index: int, assertion_index: int) -> dict[str, Any]:
    """Return one exact primitive semantic assertion mapping."""
    path = f"$.semantic_issue_proposals[{proposal_index}].assertions[{assertion_index}]"
    if type(value) is not dict or set(value) != _ASSERTION_KEYS:
        raise assessment_error(
            "invalid_semantic_assertion",
            path,
            "semantic assertions must contain the exact public field set",
        )
    return value


def _issue_statement(value: Any, path: str) -> str:
    """Return one nonempty bounded transient issue statement."""
    if type(value) is not str or not value or len(value) > MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS:
        raise assessment_error(
            "invalid_issue_statement",
            path,
            (
                "issue_statement must be nonempty text containing at most "
                f"{MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS} characters"
            ),
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise assessment_error(
            "invalid_issue_statement",
            path,
            "issue_statement must be valid UTF-8 text",
        ) from None
    return value


def _derived_identifier(prefix: str, *values: object) -> str:
    """Return one descriptive content-addressed identifier."""
    digest = hashlib.sha256(
        "\u001f".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest[:32]}"


def _build_span(
    *,
    assertion: dict[str, Any],
    proposal_index: int,
    assertion_index: int,
    sources_by_id: dict[str, tuple[EnterpriseSourceRecord, str]],
    provider_revision_fingerprint: str,
) -> tuple[EvidenceSpanRecord, str | None]:
    """Reconstruct one exact evidence span and optional stakeholder identity."""
    base_path = f"$.semantic_issue_proposals[{proposal_index}].assertions[{assertion_index}]"
    source_id = descriptive_identifier(
        assertion["source_id"],
        "source_id",
        f"{base_path}.source_id",
    )
    if source_id not in sources_by_id:
        raise assessment_error(
            "unknown_semantic_assertion_source",
            f"{base_path}.source_id",
            "semantic assertion source is not in the verified source packet",
        )
    record, source_text = sources_by_id[source_id]
    start = _offset(assertion["start_offset"], "start_offset", f"{base_path}.start_offset")
    end = _offset(assertion["end_offset"], "end_offset", f"{base_path}.end_offset")
    if end <= start or start >= len(source_text) or end > len(source_text):
        raise assessment_error(
            "invalid_semantic_assertion_offsets",
            f"{base_path}.end_offset",
            "semantic assertion offsets must define one nonempty in-range span",
        )
    kind = enum_value(
        assertion["assertion_kind"],
        EnterpriseAssertionKind,
        "assertion_kind",
        f"{base_path}.assertion_kind",
    )
    stakeholder = assertion["stakeholder_id"]
    if kind is EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT:
        if stakeholder is None:
            raise assessment_error(
                "missing_semantic_stakeholder",
                f"{base_path}.stakeholder_id",
                "stakeholder value judgments require a stakeholder identifier",
            )
        stakeholder_id = descriptive_identifier(
            stakeholder,
            "stakeholder_id",
            f"{base_path}.stakeholder_id",
        )
    else:
        if stakeholder is not None:
            raise assessment_error(
                "unexpected_semantic_stakeholder",
                f"{base_path}.stakeholder_id",
                "only stakeholder value judgments may name a stakeholder",
            )
        stakeholder_id = None
    metadata = _semantic_metadata(assertion["metadata"], path=f"{base_path}.metadata")
    exact_span = source_text[start:end]
    span_digest = hashlib.sha256(exact_span.encode("utf-8")).hexdigest()
    span_id = _derived_identifier(
        "semantic_span",
        record.source_record_fingerprint,
        start,
        end,
        kind.value,
        stakeholder_id or "no_stakeholder",
        provider_revision_fingerprint,
    )
    span = EvidenceSpanRecord(
        source_id=record.source_id,
        source_record_fingerprint=record.source_record_fingerprint,
        span_id=span_id,
        span_content_fingerprint=span_digest,
        assertion_kind=kind,
        start_offset=start,
        end_offset=end,
        metadata={
            "offset_unit": "python_unicode_code_point",
            "semantic_provider_revision_fingerprint": provider_revision_fingerprint,
            "semantic_assertion_metadata": metadata,
        },
    )
    return span, stakeholder_id


def _compile_proposal(
    proposal: dict[str, Any],
    *,
    proposal_index: int,
    sources_by_id: dict[str, tuple[EnterpriseSourceRecord, str]],
    provider_revision_fingerprint: str,
) -> tuple[AtomicIssueRecord, tuple[StakeholderPerspective, ...]]:
    """Compile one primitive proposal through existing package-owned contracts."""
    base_path = f"$.semantic_issue_proposals[{proposal_index}]"
    issue_id = descriptive_identifier(
        proposal["issue_id"],
        "issue_id",
        f"{base_path}.issue_id",
    )
    issue_family_id = descriptive_identifier(
        proposal["issue_family_id"],
        "issue_family_id",
        f"{base_path}.issue_family_id",
    )
    statement = _issue_statement(proposal["issue_statement"], f"{base_path}.issue_statement")
    issue_content_fingerprint = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    caller_metadata = _semantic_metadata(proposal["metadata"], path=f"{base_path}.metadata")
    assertion_values = _bounded_collection(
        proposal["assertions"],
        path=f"{base_path}.assertions",
        maximum=MAX_SEMANTIC_ASSERTIONS_PER_ISSUE,
        invalid_code="invalid_semantic_assertions",
        too_many_code="too_many_semantic_assertions",
        missing_code="missing_semantic_assertions",
    )

    entries: list[tuple[EvidenceSpanRecord, str | None]] = []
    occurrence_keys: set[tuple[str, int, int, EnterpriseAssertionKind, str | None]] = set()
    for assertion_index, value in enumerate(assertion_values):
        assertion = _assertion_mapping(value, proposal_index, assertion_index)
        span, stakeholder_id = _build_span(
            assertion=assertion,
            proposal_index=proposal_index,
            assertion_index=assertion_index,
            sources_by_id=sources_by_id,
            provider_revision_fingerprint=provider_revision_fingerprint,
        )
        occurrence_key = (
            span.source_record_fingerprint,
            span.start_offset,
            span.end_offset,
            span.assertion_kind,
            stakeholder_id,
        )
        if occurrence_key in occurrence_keys:
            raise assessment_error(
                "duplicate_semantic_assertion",
                f"{base_path}.assertions",
                "semantic assertions must not duplicate one source occurrence",
            )
        occurrence_keys.add(occurrence_key)
        entries.append((span, stakeholder_id))

    entries.sort(
        key=lambda value: (
            value[0].source_record_fingerprint,
            value[0].start_offset,
            value[0].end_offset,
            value[0].assertion_kind.value,
            value[0].span_content_fingerprint,
        )
    )
    for previous, current in zip(entries, entries[1:], strict=False):
        if (
            previous[0].source_record_fingerprint
            == current[0].source_record_fingerprint
            and current[0].start_offset < previous[0].end_offset
        ):
            raise assessment_error(
                "overlapping_semantic_assertions",
                f"{base_path}.assertions",
                "semantic assertions must not overlap within one source revision",
            )

    evidence_spans: list[EvidenceSpanRecord] = []
    counterevidence_records: list[CounterevidenceRecord] = []
    perspectives: list[StakeholderPerspective] = []
    source_fingerprints: set[str] = set()
    assertion_fingerprints: list[str] = []
    for span, stakeholder_id in entries:
        source_fingerprints.add(span.source_record_fingerprint)
        assertion_fingerprints.append(span.evidence_span_fingerprint)
        if span.assertion_kind is EnterpriseAssertionKind.COUNTEREVIDENCE:
            counterevidence_records.append(
                CounterevidenceRecord(
                    counterevidence_id=_derived_identifier(
                        "semantic_counterevidence",
                        issue_content_fingerprint,
                        span.evidence_span_fingerprint,
                    ),
                    issue_content_fingerprint=issue_content_fingerprint,
                    evidence_span=span,
                    metadata={
                        "semantic_provider_revision_fingerprint": provider_revision_fingerprint,
                    },
                )
            )
        elif span.assertion_kind is EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT:
            if stakeholder_id is None:  # pragma: no cover - validated above
                raise RuntimeError("stakeholder identity is unavailable")
            perspectives.append(
                StakeholderPerspective(
                    perspective_id=_derived_identifier(
                        "semantic_perspective",
                        issue_content_fingerprint,
                        stakeholder_id,
                        span.evidence_span_fingerprint,
                    ),
                    stakeholder_id=stakeholder_id,
                    issue_content_fingerprint=issue_content_fingerprint,
                    value_judgment_span=span,
                    metadata={
                        "semantic_provider_revision_fingerprint": provider_revision_fingerprint,
                    },
                )
            )
        else:
            evidence_spans.append(span)

    if not evidence_spans and not counterevidence_records:
        raise assessment_error(
            "missing_semantic_issue_evidence",
            f"{base_path}.assertions",
            "semantic issues require evidence, inference, ambiguity, or counterevidence",
        )
    perspectives.sort(key=lambda value: value.perspective_fingerprint)
    issue_metadata = dict(caller_metadata)
    issue_metadata.update(
        {
            "semantic_provider_revision_fingerprint": provider_revision_fingerprint,
            "semantic_assertion_fingerprints": sorted(assertion_fingerprints),
            "semantic_perspective_fingerprints": [
                value.perspective_fingerprint for value in perspectives
            ],
        }
    )
    issue = AtomicIssueRecord(
        issue_id=issue_id,
        issue_family_id=issue_family_id,
        issue_content_fingerprint=issue_content_fingerprint,
        source_record_fingerprints=tuple(sorted(source_fingerprints)),
        evidence_spans=tuple(evidence_spans),
        counterevidence_records=tuple(counterevidence_records),
        metadata=issue_metadata,
    )
    return issue, tuple(perspectives)


def extract_enterprise_semantic_issues(
    enterprise_sources: Iterable[EnterpriseSourceRecord],
    *,
    source_text_by_id: Mapping[str, str],
    provider: EnterpriseSemanticIssueProvider,
) -> tuple[tuple[AtomicIssueRecord, ...], tuple[StakeholderPerspective, ...]]:
    """Validate provider proposals and return canonical issues and perspectives.

    Exact source text is replayed before provider execution and remains transient.
    Provider output is reconstructed into existing package-owned records in
    deterministic fingerprint order.
    """
    sources = _verified_sources(enterprise_sources, source_text_by_id)
    if not isinstance(provider, EnterpriseSemanticIssueProvider):
        raise assessment_error(
            "invalid_semantic_issue_provider",
            "$.provider",
            "provider must implement EnterpriseSemanticIssueProvider",
        )
    try:
        provider_revision = fingerprint(
            provider.provider_revision_fingerprint,
            "provider_revision_fingerprint",
            "$.provider.provider_revision_fingerprint",
        )
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            "invalid_provider_revision_fingerprint",
            "$.provider.provider_revision_fingerprint",
            "provider revision could not be inspected safely",
        ) from None
    try:
        proposals_value = provider.propose(sources=sources)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            "semantic_issue_provider_failure",
            "$.provider",
            "semantic issue provider failed without usable output",
        ) from None
    proposals = _bounded_collection(
        proposals_value,
        path="$.semantic_issue_proposals",
        maximum=MAX_SEMANTIC_ISSUE_PROPOSALS,
        invalid_code="invalid_semantic_issue_proposals",
        too_many_code="too_many_semantic_issue_proposals",
    )
    sources_by_id = {record.source_id: (record, text) for record, text in sources}
    compiled: list[AtomicIssueRecord] = []
    perspectives: list[StakeholderPerspective] = []
    issue_ids: set[str] = set()
    content_fingerprints: set[str] = set()
    for index, value in enumerate(proposals):
        proposal = _proposal_mapping(value, index)
        issue, issue_perspectives = _compile_proposal(
            proposal,
            proposal_index=index,
            sources_by_id=sources_by_id,
            provider_revision_fingerprint=provider_revision,
        )
        if issue.issue_id in issue_ids:
            raise assessment_error(
                "duplicate_semantic_issue_id",
                "$.semantic_issue_proposals",
                "semantic issue identifiers must be unique within one batch",
            )
        if issue.issue_content_fingerprint in content_fingerprints:
            raise assessment_error(
                "duplicate_semantic_issue_content",
                "$.semantic_issue_proposals",
                "semantic issue content revisions must be unique within one batch",
            )
        issue_ids.add(issue.issue_id)
        content_fingerprints.add(issue.issue_content_fingerprint)
        compiled.append(issue)
        perspectives.extend(issue_perspectives)
    validated_issues = extract_enterprise_atomic_issues(
        tuple(record for record, _text in sources),
        {record.source_id: text for record, text in sources},
        extractor=StaticEnterpriseIssueExtractor(tuple(compiled)),
    )
    perspectives.sort(key=lambda value: value.perspective_fingerprint)
    return validated_issues, tuple(perspectives)


__all__ = [
    "MAX_SEMANTIC_ASSERTIONS_PER_ISSUE",
    "MAX_SEMANTIC_ISSUE_PROPOSALS",
    "MAX_SEMANTIC_ISSUE_STATEMENT_CHARACTERS",
    "EnterpriseSemanticIssueProvider",
    "OfflineSemanticIssueFixtureProvider",
    "extract_enterprise_semantic_issues",
]
