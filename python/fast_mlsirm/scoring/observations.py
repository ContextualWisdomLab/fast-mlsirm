"""Lossless provider-neutral scoring-engine and observation provenance contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
import operator
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._contract_safety import artifact_digest, bounded_values, freeze_metadata
from ._validation import (
    MAX_ASSESSMENT_RUBRICS,
    MAX_SIGNED_INTEGER,
    AssessmentSpecError,
    CanonicalContract,
    assessment_error,
    bounded_text,
    descriptive_identifier,
    enum_value,
    fingerprint,
    semantic_version,
    thaw_json_value,
)
from .assessment import AssessmentSpec

OBSERVATION_SCHEMA_VERSION = "1.0"
MAX_EVIDENCE_SPANS = 64
_OBSERVATION_TOKEN = object()


class RaterKind(str, Enum):
    """Origin of one scoring judgment."""

    HUMAN = "human"
    AUTOMATED = "automated"


class ObservationStatus(str, Enum):
    """Outcome state for one requested scoring judgment."""

    SCORED = "scored"
    ABSTAINED = "abstained"
    FAILED = "failed"
    EXCLUDED = "excluded"


class EvidenceSourceKind(str, Enum):
    """Artifact category referenced by one evidence span."""

    RESPONSE = "response"
    PROMPT = "prompt"
    SOURCE = "source"
    EXTERNAL = "external"


def _observation_schema_version(value: Any) -> str:
    """Require the independent observation wire-schema version."""
    if value != OBSERVATION_SCHEMA_VERSION:
        raise assessment_error(
            "invalid_observation_schema_version",
            "$.schema_version",
            f"schema_version must be '{OBSERVATION_SCHEMA_VERSION}'",
        )
    return OBSERVATION_SCHEMA_VERSION


def _bounded_offset(value: Any, name: str) -> int:
    """Return a non-negative signed-64-bit offset with stable callback failures."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a non-negative signed-64-bit integer",
        )
    try:
        normalized = operator.index(value)
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a non-negative signed-64-bit integer",
        ) from None
    if not 0 <= normalized <= MAX_SIGNED_INTEGER:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a non-negative signed-64-bit integer",
        )
    return int(normalized)


def _optional_confidence(value: Any) -> float | None:
    """Return an optional finite confidence probability without Boolean coercion."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise assessment_error(
            "invalid_observation_confidence",
            "$.confidence",
            "confidence must be finite and in [0, 1]",
        )
    try:
        normalized = float(value)
    except Exception:
        raise assessment_error(
            "invalid_observation_confidence",
            "$.confidence",
            "confidence must be finite and in [0, 1]",
        ) from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise assessment_error(
            "invalid_observation_confidence",
            "$.confidence",
            "confidence must be finite and in [0, 1]",
        )
    return 0.0 if normalized == 0.0 else normalized


def _optional_score_category(value: Any) -> int | None:
    """Return an optional exact integer score category while rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be an integer declared by the rubric",
        )
    try:
        normalized = operator.index(value)
    except Exception:
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be an integer declared by the rubric",
        ) from None
    return int(normalized)


@dataclass(frozen=True)
class EngineDescriptor(CanonicalContract):
    """Immutable versioned descriptor for one human or automated scorer path."""

    engine_id: str
    engine_family_id: str
    engine_version: str
    rater_kind: RaterKind
    prompt_template_id: str
    prompt_template_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize engine identity, versioning, kind, prompt, and metadata."""
        object.__setattr__(
            self,
            "engine_id",
            descriptive_identifier(self.engine_id, "engine_id"),
        )
        object.__setattr__(
            self,
            "engine_family_id",
            descriptive_identifier(self.engine_family_id, "engine_family_id"),
        )
        object.__setattr__(
            self,
            "engine_version",
            semantic_version(self.engine_version, "engine_version"),
        )
        object.__setattr__(
            self,
            "rater_kind",
            enum_value(self.rater_kind, RaterKind, "rater_kind"),
        )
        object.__setattr__(
            self,
            "prompt_template_id",
            descriptive_identifier(self.prompt_template_id, "prompt_template_id"),
        )
        object.__setattr__(
            self,
            "prompt_template_version",
            semantic_version(
                self.prompt_template_version,
                "prompt_template_version",
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical engine content without derived identities."""
        return {
            "engine_id": self.engine_id,
            "engine_family_id": self.engine_family_id,
            "engine_version": self.engine_version,
            "rater_kind": self.rater_kind.value,
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_version": self.prompt_template_version,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def engine_fingerprint(self) -> str:
        """Return SHA-256 over the immutable engine descriptor."""
        return artifact_digest(self)

    @property
    def engine_handle(self) -> str:
        """Return a descriptive 128-bit public engine handle."""
        return f"engine_descriptor_{self.engine_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical engine content and deterministic identities."""
        return {
            **self._content_dict(),
            "engine_handle": self.engine_handle,
            "engine_fingerprint": self.engine_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceSpan(CanonicalContract):
    """One raw-content-free source span supporting a scoring judgment."""

    source_kind: EvidenceSourceKind
    source_id: str
    content_digest: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        """Normalize source provenance and validate a positive bounded span."""
        object.__setattr__(
            self,
            "source_kind",
            enum_value(self.source_kind, EvidenceSourceKind, "source_kind"),
        )
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "content_digest",
            fingerprint(self.content_digest, "content_digest"),
        )
        start = _bounded_offset(self.start_offset, "start_offset")
        end = _bounded_offset(self.end_offset, "end_offset")
        if end <= start:
            raise assessment_error(
                "invalid_evidence_span",
                "$.end_offset",
                "end_offset must be greater than start_offset",
            )
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical evidence provenance without raw content."""
        return {
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "content_digest": self.content_digest,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    @property
    def span_fingerprint(self) -> str:
        """Return SHA-256 over the exact source span provenance."""
        return artifact_digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical evidence provenance and its fingerprint."""
        return {
            **self._content_dict(),
            "span_fingerprint": self.span_fingerprint,
        }


def _normalize_evidence_spans(values: Iterable[Any]) -> tuple[EvidenceSpan, ...]:
    """Return bounded typed evidence spans in deterministic fingerprint order."""
    raw = bounded_values(
        values,
        "evidence_spans",
        minimum=0,
        maximum=MAX_EVIDENCE_SPANS,
    )
    spans: list[tuple[str, EvidenceSpan]] = []
    for index, span in enumerate(raw):
        if not isinstance(span, EvidenceSpan):
            raise assessment_error(
                "invalid_evidence_span",
                f"$.evidence_spans[{index}]",
                "evidence entries must be EvidenceSpan values",
            )
        spans.append((span.span_fingerprint, span))
    fingerprints = tuple(item[0] for item in spans)
    if len(set(fingerprints)) != len(fingerprints):
        raise assessment_error(
            "duplicate_evidence_span",
            "$.evidence_spans",
            "evidence spans must be unique",
        )
    return tuple(span for _, span in sorted(spans, key=lambda item: item[0]))


def _normalize_state(
    status: ObservationStatus | str,
    score_category: Any,
    reason_code: Any,
) -> tuple[ObservationStatus, int | None, str | None]:
    """Normalize mutually exclusive score and reason state fields."""
    normalized_status = enum_value(status, ObservationStatus, "status")
    normalized_score = _optional_score_category(score_category)
    normalized_reason = (
        None
        if reason_code is None
        else descriptive_identifier(reason_code, "reason_code")
    )
    if normalized_status is ObservationStatus.SCORED:
        if normalized_score is None:
            raise assessment_error(
                "missing_score_category",
                "$.score_category",
                "scored observations require score_category",
            )
        if normalized_reason is not None:
            raise assessment_error(
                "scored_reason_forbidden",
                "$.reason_code",
                "scored observations cannot contain reason_code",
            )
    else:
        if normalized_score is not None:
            raise assessment_error(
                "non_scored_category_forbidden",
                "$.score_category",
                "non-scored observations cannot contain score_category",
            )
        if normalized_reason is None:
            raise assessment_error(
                "missing_reason_code",
                "$.reason_code",
                "non-scored observations require reason_code",
            )
    return normalized_status, normalized_score, normalized_reason


@dataclass(frozen=True)
class ScoreObservation(CanonicalContract):
    """Factory-sealed lossless scoring judgment bound to exact provenance."""

    assessment_fingerprint: str
    engine: EngineDescriptor
    response_id: str
    task_id: str
    rater_id: str
    occasion_id: str
    construct_id: str
    rubric_fingerprint: str
    status: ObservationStatus
    score_category: int | None
    confidence: float | None
    reason_code: str | None
    evidence_spans: tuple[EvidenceSpan, ...]
    metadata: Mapping[str, Any]
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    _observation_token: InitVar[object | None] = None

    def __post_init__(self, _observation_token: object | None) -> None:
        """Reject direct construction and normalize the immutable observation."""
        if _observation_token is not _OBSERVATION_TOKEN:
            raise assessment_error(
                "unverified_score_observation",
                "$",
                "ScoreObservation must be created by build_score_observation",
            )
        object.__setattr__(
            self,
            "assessment_fingerprint",
            fingerprint(self.assessment_fingerprint, "assessment_fingerprint"),
        )
        if not isinstance(self.engine, EngineDescriptor):
            raise assessment_error(
                "invalid_observation_engine",
                "$.engine",
                "engine must use the package-owned EngineDescriptor",
            )
        for field_name in (
            "response_id",
            "task_id",
            "rater_id",
            "occasion_id",
            "construct_id",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "rubric_fingerprint",
            fingerprint(self.rubric_fingerprint, "rubric_fingerprint"),
        )
        status, score, reason = _normalize_state(
            self.status,
            self.score_category,
            self.reason_code,
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score_category", score)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "confidence", _optional_confidence(self.confidence))
        object.__setattr__(
            self,
            "evidence_spans",
            _normalize_evidence_spans(self.evidence_spans),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            _observation_schema_version(self.schema_version),
        )

    @property
    def engine_fingerprint(self) -> str:
        """Return the exact engine descriptor fingerprint."""
        return self.engine.engine_fingerprint

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical observation content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "engine": self.engine.to_dict(),
            "response_id": self.response_id,
            "task_id": self.task_id,
            "rater_id": self.rater_id,
            "occasion_id": self.occasion_id,
            "construct_id": self.construct_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "status": self.status.value,
            "score_category": self.score_category,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def observation_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable observation."""
        return artifact_digest(self)

    @property
    def observation_handle(self) -> str:
        """Return a descriptive 128-bit public observation handle."""
        return f"score_observation_{self.observation_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical observation content and deterministic identities."""
        return {
            **self._content_dict(),
            "observation_handle": self.observation_handle,
            "observation_fingerprint": self.observation_fingerprint,
        }


def _rubric_registry(
    values: Iterable[Any],
) -> dict[str, RubricSpecification]:
    """Return a unique exact rubric registry with safely computed fingerprints."""
    raw = bounded_values(
        values,
        "rubrics",
        minimum=1,
        maximum=MAX_ASSESSMENT_RUBRICS,
    )
    registry: dict[str, RubricSpecification] = {}
    for index, rubric in enumerate(raw):
        if not isinstance(rubric, RubricSpecification):
            raise assessment_error(
                "invalid_observation_rubric",
                f"$.rubrics[{index}]",
                "rubric entries must be RubricSpecification values",
            )
        try:
            rubric_fingerprint = rubric.fingerprint
        except Exception:
            raise assessment_error(
                "invalid_observation_rubric",
                f"$.rubrics[{index}]",
                "rubric fingerprint could not be computed safely",
            ) from None
        if rubric_fingerprint in registry:
            raise assessment_error(
                "duplicate_observation_rubric",
                "$.rubrics",
                "rubric fingerprints must be unique",
            )
        registry[rubric_fingerprint] = rubric
    return registry


def _validate_engine_policy(assessment: AssessmentSpec, engine: EngineDescriptor) -> None:
    """Require one scorer kind and automated engine to be enabled by policy."""
    policy = assessment.engine_policy
    if engine.rater_kind is RaterKind.AUTOMATED:
        if not policy.allow_automated_raters:
            raise assessment_error(
                "automated_rater_disabled",
                "$.engine.rater_kind",
                "automated raters are disabled by the assessment policy",
            )
        if engine.engine_id not in policy.engine_ids:
            raise assessment_error(
                "unknown_engine_id",
                "$.engine.engine_id",
                "automated engine is not declared by the assessment policy",
            )
    elif not policy.allow_human_raters:
        raise assessment_error(
            "human_rater_disabled",
            "$.engine.rater_kind",
            "human raters are disabled by the assessment policy",
        )


def build_score_observation(
    *,
    assessment: AssessmentSpec,
    rubrics: Iterable[RubricSpecification],
    engine: EngineDescriptor,
    response_id: str,
    task_id: str,
    rater_id: str,
    occasion_id: str,
    construct_id: str,
    rubric_fingerprint: str,
    status: ObservationStatus | str,
    score_category: int | None,
    confidence: float | None,
    reason_code: str | None,
    evidence_spans: Iterable[EvidenceSpan] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ScoreObservation:
    """Build one graph-validated immutable scoring observation."""
    if not isinstance(assessment, AssessmentSpec):
        raise assessment_error(
            "invalid_observation_assessment",
            "$.assessment",
            "assessment must use the package-owned AssessmentSpec",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_observation_engine",
            "$.engine",
            "engine must use the package-owned EngineDescriptor",
        )
    _validate_engine_policy(assessment, engine)

    normalized_construct_id = descriptive_identifier(construct_id, "construct_id")
    construct = next(
        (
            candidate
            for candidate in assessment.constructs
            if candidate.construct_id == normalized_construct_id
        ),
        None,
    )
    if construct is None:
        raise assessment_error(
            "unknown_observation_construct",
            "$.construct_id",
            "observation construct is not declared by the assessment",
        )

    normalized_rubric_fingerprint = fingerprint(
        rubric_fingerprint,
        "rubric_fingerprint",
    )
    if normalized_rubric_fingerprint not in construct.rubric_fingerprints:
        raise assessment_error(
            "unknown_observation_rubric",
            "$.rubric_fingerprint",
            "rubric is not bound to the observation construct",
        )

    registry = _rubric_registry(rubrics)
    if set(registry) != set(assessment.rubric_fingerprints):
        raise assessment_error(
            "unused_observation_rubric",
            "$.rubrics",
            "rubric registry must match the exact assessment rubric set",
        )
    rubric = registry.get(normalized_rubric_fingerprint)
    if rubric is None:
        raise assessment_error(
            "unknown_observation_rubric",
            "$.rubric_fingerprint",
            "rubric is absent from the supplied registry",
        )
    if rubric.construct_id != normalized_construct_id:
        raise assessment_error(
            "observation_rubric_construct_mismatch",
            "$.rubric_fingerprint",
            "rubric construct does not match the observation construct",
        )

    normalized_status, normalized_score, normalized_reason = _normalize_state(
        status,
        score_category,
        reason_code,
    )
    if normalized_status is ObservationStatus.SCORED:
        valid_scores = {level.score for level in rubric.levels}
        if normalized_score not in valid_scores:
            raise assessment_error(
                "invalid_score_category",
                "$.score_category",
                "score_category is not declared by the exact rubric",
            )

    return ScoreObservation(
        assessment_fingerprint=assessment.assessment_fingerprint,
        engine=engine,
        response_id=response_id,
        task_id=task_id,
        rater_id=rater_id,
        occasion_id=occasion_id,
        construct_id=normalized_construct_id,
        rubric_fingerprint=normalized_rubric_fingerprint,
        status=normalized_status,
        score_category=normalized_score,
        confidence=confidence,
        reason_code=normalized_reason,
        evidence_spans=tuple(evidence_spans),
        metadata={} if metadata is None else metadata,
        schema_version=OBSERVATION_SCHEMA_VERSION,
        _observation_token=_OBSERVATION_TOKEN,
    )
