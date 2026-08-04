"""Lossless scoring observations and bounded evidence-reference contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import math
import operator
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._contract_safety import artifact_digest, bounded_values, freeze_metadata
from ._validation import (
    MAX_ASSESSMENT_RUBRICS,
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
from .assessment import AssessmentResponseType, AssessmentSpec

MAX_EVIDENCE_SPANS = 64
MAX_OBSERVATIONS = 1_000_000
MAX_EVIDENCE_OFFSET = (1 << 63) - 1
_OBSERVATION_TOKEN = object()


class ObservationStatus(str, Enum):
    """Mutually exclusive lifecycle state of one scoring judgment."""

    SCORED = "scored"
    MISSING = "missing"
    ABSTAINED = "abstained"
    FAILED = "failed"
    EXCLUDED = "excluded"


class ObservationLevel(str, Enum):
    """Granularity represented by one scoring observation."""

    CRITERION_LEVEL = "criterion_level"
    HOLISTIC = "holistic"


class RaterKind(str, Enum):
    """Whether a scoring observation came from a human or automated rater."""

    HUMAN = "human"
    AUTOMATED = "automated"


def _optional_identifier(value: Any, name: str, path: str) -> str | None:
    """Return an optional descriptive identifier with stable failures."""
    if value is None:
        return None
    return descriptive_identifier(value, name, path)


def _optional_version(value: Any, name: str, path: str) -> str | None:
    """Return an optional canonical semantic version with stable failures."""
    if value is None:
        return None
    return semantic_version(value, name, path)


def _bounded_nonnegative_integer(value: Any, name: str, path: str) -> int:
    """Return a callback-safe non-negative signed-64-bit integer."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a non-negative integer",
        )
    try:
        normalized = operator.index(value)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a non-negative integer",
        ) from None
    if isinstance(normalized, bool) or not 0 <= normalized <= MAX_EVIDENCE_OFFSET:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be between 0 and {MAX_EVIDENCE_OFFSET}",
        )
    return int(normalized)


def _optional_confidence(value: Any) -> float | None:
    """Return optional scorer-reported confidence in the closed unit interval."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise assessment_error(
            "invalid_confidence",
            "$.confidence",
            "confidence must be a finite number between 0 and 1",
        )
    try:
        normalized = float(value)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            "invalid_confidence",
            "$.confidence",
            "confidence must be a finite number between 0 and 1",
        ) from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise assessment_error(
            "invalid_confidence",
            "$.confidence",
            "confidence must be a finite number between 0 and 1",
        )
    return 0.0 if normalized == 0.0 else normalized


def _optional_score(value: Any) -> int | None:
    """Return an optional callback-safe integer score category."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be an integer rubric category",
        )
    try:
        normalized = operator.index(value)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be an integer rubric category",
        ) from None
    if isinstance(normalized, bool):
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be an integer rubric category",
        )
    return int(normalized)


@dataclass(frozen=True)
class EvidenceSpan(CanonicalContract):
    """Bounded source reference and half-open offsets without raw content."""

    source_id: str
    start_offset: int
    end_offset: int
    evidence_label: str | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        """Normalize reference identity, offsets, label, and optional digest."""
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        start = _bounded_nonnegative_integer(
            self.start_offset,
            "start_offset",
            "$.start_offset",
        )
        end = _bounded_nonnegative_integer(
            self.end_offset,
            "end_offset",
            "$.end_offset",
        )
        if end <= start:
            raise assessment_error(
                "invalid_evidence_offsets",
                "$.end_offset",
                "end_offset must be greater than start_offset",
            )
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)
        if self.evidence_label is not None:
            object.__setattr__(
                self,
                "evidence_label",
                descriptive_identifier(
                    self.evidence_label,
                    "evidence_label",
                ),
            )
        if self.content_digest is not None:
            object.__setattr__(
                self,
                "content_digest",
                fingerprint(self.content_digest, "content_digest"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible evidence reference."""
        return {
            "source_id": self.source_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "evidence_label": self.evidence_label,
            "content_digest": self.content_digest,
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class ScoreObservation(CanonicalContract):
    """Factory-sealed, content-addressed human or automated score event."""

    observation_id: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    response_id: str
    rater_id: str
    rater_kind: RaterKind
    engine_id: str | None
    construct_id: str
    observation_level: ObservationLevel
    criterion_id: str | None
    status: ObservationStatus
    score_category: int | None
    reason_code: str | None
    confidence: float | None
    evidence_spans: tuple[EvidenceSpan, ...]
    occasion_id: str
    scorer_family: str
    scorer_version: str
    prompt_template_version: str | None
    metadata: Mapping[str, Any]
    _observation_token: InitVar[object | None] = None

    def __post_init__(self, _observation_token: object | None) -> None:
        """Reject direct construction and preserve factory-validated values."""
        if _observation_token is not _OBSERVATION_TOKEN:
            raise assessment_error(
                "unverified_score_observation",
                "$",
                "ScoreObservation must be created by build_score_observation",
            )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical observation content without derived identities."""
        return {
            "observation_id": self.observation_id,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "response_id": self.response_id,
            "rater_id": self.rater_id,
            "rater_kind": self.rater_kind.value,
            "engine_id": self.engine_id,
            "construct_id": self.construct_id,
            "observation_level": self.observation_level.value,
            "criterion_id": self.criterion_id,
            "status": self.status.value,
            "score_category": self.score_category,
            "reason_code": self.reason_code,
            "confidence": self.confidence,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "occasion_id": self.occasion_id,
            "scorer_family": self.scorer_family,
            "scorer_version": self.scorer_version,
            "prompt_template_version": self.prompt_template_version,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def observation_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable observation content."""
        return artifact_digest(self)

    @property
    def observation_handle(self) -> str:
        """Return a descriptive 128-bit public observation handle."""
        return f"score_observation_{self.observation_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content plus deterministic observation identities."""
        return {
            **self._content_dict(),
            "observation_handle": self.observation_handle,
            "observation_fingerprint": self.observation_fingerprint,
        }


def _materialize_rubrics(
    values: Iterable[Any],
) -> tuple[tuple[str, RubricSpecification], ...]:
    """Return a bounded exact rubric registry with safe fingerprint access."""
    raw = bounded_values(
        values,
        "rubrics",
        minimum=1,
        maximum=MAX_ASSESSMENT_RUBRICS,
    )
    output: list[tuple[str, RubricSpecification]] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for index, rubric in enumerate(raw):
        if not isinstance(rubric, RubricSpecification):
            raise assessment_error(
                "invalid_rubric",
                f"$.rubrics[{index}]",
                "rubric entries must be RubricSpecification values",
            )
        try:
            rubric_fingerprint = rubric.fingerprint
        except Exception:
            raise assessment_error(
                "invalid_rubric_fingerprint",
                f"$.rubrics[{index}]",
                "rubric fingerprint could not be computed safely",
            ) from None
        if rubric.rubric_id in seen_ids:
            raise assessment_error(
                "duplicate_rubric_id",
                "$.rubrics",
                "rubric identifiers must be unique",
            )
        if rubric_fingerprint in seen_fingerprints:
            raise assessment_error(
                "duplicate_rubric_fingerprint",
                "$.rubrics",
                "rubric fingerprints must be unique",
            )
        seen_ids.add(rubric.rubric_id)
        seen_fingerprints.add(rubric_fingerprint)
        output.append((rubric_fingerprint, rubric))
    return tuple(sorted(output, key=lambda entry: entry[0]))


def _assessment_context(
    assessment: Any,
    rubrics: Iterable[Any],
) -> tuple[AssessmentSpec, dict[str, RubricSpecification], dict[str, str]]:
    """Validate assessment type and exact rubric-to-construct bindings."""
    if not isinstance(assessment, AssessmentSpec):
        raise assessment_error(
            "invalid_assessment_spec",
            "$.assessment",
            "assessment must use the package-owned AssessmentSpec contract",
        )
    normalized = _materialize_rubrics(rubrics)
    rubrics_by_fingerprint = {key: rubric for key, rubric in normalized}
    if set(rubrics_by_fingerprint) != set(assessment.rubric_fingerprints):
        raise assessment_error(
            "rubric_registry_mismatch",
            "$.rubrics",
            "rubric registry must exactly match the assessment fingerprints",
        )
    constructs_by_rubric: dict[str, str] = {}
    for construct in assessment.constructs:
        for rubric_fingerprint in construct.rubric_fingerprints:
            constructs_by_rubric[rubric_fingerprint] = construct.construct_id
    return assessment, rubrics_by_fingerprint, constructs_by_rubric


def _normalize_evidence_spans(values: Iterable[Any]) -> tuple[EvidenceSpan, ...]:
    """Return bounded typed spans in deterministic order without merging."""
    raw = bounded_values(
        values,
        "evidence_spans",
        minimum=0,
        maximum=MAX_EVIDENCE_SPANS,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, EvidenceSpan):
            raise assessment_error(
                "invalid_evidence_span",
                f"$.evidence_spans[{index}]",
                "evidence span entries must be EvidenceSpan values",
            )
    return tuple(
        sorted(
            raw,
            key=lambda span: (
                span.source_id,
                span.start_offset,
                span.end_offset,
                span.evidence_label or "",
                span.content_digest or "",
            ),
        )
    )


def _validate_status(
    *,
    status: ObservationStatus,
    score_category: int | None,
    reason_code: str | None,
) -> None:
    """Enforce lossless status-specific score and reason invariants."""
    if status is ObservationStatus.SCORED:
        if score_category is None:
            raise assessment_error(
                "missing_score_category",
                "$.score_category",
                "scored observations require one rubric category",
            )
        if reason_code is not None:
            raise assessment_error(
                "unexpected_reason_code",
                "$.reason_code",
                "scored observations cannot carry a non-scored reason",
            )
        return
    if score_category is not None:
        raise assessment_error(
            "unexpected_score_category",
            "$.score_category",
            "non-scored observations cannot carry a score category",
        )
    if status is ObservationStatus.MISSING:
        if reason_code is not None:
            raise assessment_error(
                "unexpected_reason_code",
                "$.reason_code",
                "missing observations cannot carry a reason code",
            )
        return
    if reason_code is None:
        raise assessment_error(
            "missing_reason_code",
            "$.reason_code",
            "abstained, failed, and excluded observations require a reason code",
        )


def _validate_observation_level(
    assessment: AssessmentSpec,
    level: ObservationLevel,
    criterion_id: str | None,
) -> None:
    """Require criterion/holistic consistency and assessment capability."""
    if level is ObservationLevel.CRITERION_LEVEL and criterion_id is None:
        raise assessment_error(
            "missing_criterion_id",
            "$.criterion_id",
            "criterion-level observations require a criterion identifier",
        )
    if level is ObservationLevel.HOLISTIC and criterion_id is not None:
        raise assessment_error(
            "unexpected_criterion_id",
            "$.criterion_id",
            "holistic observations cannot carry a criterion identifier",
        )
    allowed = {
        AssessmentResponseType.CRITERION_LEVEL: {ObservationLevel.CRITERION_LEVEL},
        AssessmentResponseType.HOLISTIC: {ObservationLevel.HOLISTIC},
        AssessmentResponseType.MIXED: {
            ObservationLevel.CRITERION_LEVEL,
            ObservationLevel.HOLISTIC,
        },
    }[assessment.response_type]
    if level not in allowed:
        raise assessment_error(
            "unsupported_observation_level",
            "$.observation_level",
            "observation level is not permitted by the assessment response type",
        )


def _validate_rater_engine(
    assessment: AssessmentSpec,
    rater_kind: RaterKind,
    engine_id: str | None,
) -> None:
    """Enforce human/automated policy and exact engine identity."""
    policy = assessment.engine_policy
    if rater_kind is RaterKind.HUMAN:
        if not policy.allow_human_raters:
            raise assessment_error(
                "disabled_human_rater",
                "$.rater_kind",
                "human raters are disabled by the assessment engine policy",
            )
        if engine_id is not None:
            raise assessment_error(
                "unexpected_engine_id",
                "$.engine_id",
                "human observations cannot carry an automated engine identifier",
            )
        return
    if not policy.allow_automated_raters:
        raise assessment_error(
            "disabled_automated_rater",
            "$.rater_kind",
            "automated raters are disabled by the assessment engine policy",
        )
    if engine_id is None:
        raise assessment_error(
            "missing_engine_id",
            "$.engine_id",
            "automated observations require an engine identifier",
        )
    if engine_id not in policy.engine_ids:
        raise assessment_error(
            "unknown_engine_id",
            "$.engine_id",
            "engine identifier is absent from the assessment engine policy",
        )


def build_score_observation(
    *,
    assessment: AssessmentSpec,
    rubrics: Iterable[RubricSpecification],
    observation_id: str,
    rubric_fingerprint: str,
    response_id: str,
    rater_id: str,
    rater_kind: RaterKind | str,
    engine_id: str | None,
    construct_id: str,
    observation_level: ObservationLevel | str,
    criterion_id: str | None,
    status: ObservationStatus | str,
    score_category: int | None,
    reason_code: str | None,
    occasion_id: str,
    scorer_family: str,
    scorer_version: str,
    confidence: float | None = None,
    evidence_spans: Iterable[EvidenceSpan] = (),
    prompt_template_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ScoreObservation:
    """Build one cross-reference-validated immutable scoring observation."""
    assessment_value, rubrics_by_fingerprint, constructs_by_rubric = (
        _assessment_context(assessment, rubrics)
    )
    normalized_rubric_fingerprint = fingerprint(
        rubric_fingerprint,
        "rubric_fingerprint",
    )
    rubric = rubrics_by_fingerprint.get(normalized_rubric_fingerprint)
    if rubric is None:
        raise assessment_error(
            "unknown_rubric_fingerprint",
            "$.rubric_fingerprint",
            "rubric fingerprint is absent from the assessment registry",
        )
    normalized_construct_id = descriptive_identifier(construct_id, "construct_id")
    expected_construct = constructs_by_rubric.get(normalized_rubric_fingerprint)
    if expected_construct != normalized_construct_id:
        raise assessment_error(
            "rubric_construct_mismatch",
            "$.construct_id",
            "construct does not match the selected rubric fingerprint",
        )
    normalized_level = enum_value(
        observation_level,
        ObservationLevel,
        "observation_level",
    )
    normalized_criterion_id = _optional_identifier(
        criterion_id,
        "criterion_id",
        "$.criterion_id",
    )
    _validate_observation_level(
        assessment_value,
        normalized_level,
        normalized_criterion_id,
    )
    normalized_rater_kind = enum_value(rater_kind, RaterKind, "rater_kind")
    normalized_engine_id = _optional_identifier(engine_id, "engine_id", "$.engine_id")
    _validate_rater_engine(
        assessment_value,
        normalized_rater_kind,
        normalized_engine_id,
    )
    normalized_status = enum_value(status, ObservationStatus, "status")
    normalized_score = _optional_score(score_category)
    normalized_reason = _optional_identifier(
        reason_code,
        "reason_code",
        "$.reason_code",
    )
    _validate_status(
        status=normalized_status,
        score_category=normalized_score,
        reason_code=normalized_reason,
    )
    if normalized_score is not None:
        allowed_scores = {level.score for level in rubric.levels}
        if normalized_score not in allowed_scores:
            raise assessment_error(
                "unknown_score_category",
                "$.score_category",
                "score category is absent from the selected rubric",
            )

    return ScoreObservation(
        observation_id=descriptive_identifier(observation_id, "observation_id"),
        assessment_fingerprint=assessment_value.assessment_fingerprint,
        rubric_fingerprint=normalized_rubric_fingerprint,
        response_id=descriptive_identifier(response_id, "response_id"),
        rater_id=descriptive_identifier(rater_id, "rater_id"),
        rater_kind=normalized_rater_kind,
        engine_id=normalized_engine_id,
        construct_id=normalized_construct_id,
        observation_level=normalized_level,
        criterion_id=normalized_criterion_id,
        status=normalized_status,
        score_category=normalized_score,
        reason_code=normalized_reason,
        confidence=_optional_confidence(confidence),
        evidence_spans=_normalize_evidence_spans(evidence_spans),
        occasion_id=descriptive_identifier(occasion_id, "occasion_id"),
        scorer_family=descriptive_identifier(scorer_family, "scorer_family"),
        scorer_version=semantic_version(scorer_version, "scorer_version"),
        prompt_template_version=_optional_version(
            prompt_template_version,
            "prompt_template_version",
            "$.prompt_template_version",
        ),
        metadata=freeze_metadata({} if metadata is None else metadata),
        _observation_token=_OBSERVATION_TOKEN,
    )


def _validate_existing_observation(
    observation: ScoreObservation,
    assessment: AssessmentSpec,
    rubrics_by_fingerprint: dict[str, RubricSpecification],
    constructs_by_rubric: dict[str, str],
    index: int,
) -> None:
    """Require one sealed observation to remain valid for the supplied context."""
    base_path = f"$.observations[{index}]"
    if observation.assessment_fingerprint != assessment.assessment_fingerprint:
        raise assessment_error(
            "assessment_fingerprint_mismatch",
            f"{base_path}.assessment_fingerprint",
            "observation assessment fingerprint does not match the assessment",
        )
    rubric = rubrics_by_fingerprint.get(observation.rubric_fingerprint)
    if rubric is None:
        raise assessment_error(
            "unknown_rubric_fingerprint",
            f"{base_path}.rubric_fingerprint",
            "observation rubric fingerprint is absent from the registry",
        )
    if constructs_by_rubric.get(observation.rubric_fingerprint) != observation.construct_id:
        raise assessment_error(
            "rubric_construct_mismatch",
            f"{base_path}.construct_id",
            "observation construct does not match its rubric fingerprint",
        )
    _validate_observation_level(
        assessment,
        observation.observation_level,
        observation.criterion_id,
    )
    _validate_rater_engine(
        assessment,
        observation.rater_kind,
        observation.engine_id,
    )
    _validate_status(
        status=observation.status,
        score_category=observation.score_category,
        reason_code=observation.reason_code,
    )
    if observation.score_category is not None and observation.score_category not in {
        level.score for level in rubric.levels
    }:
        raise assessment_error(
            "unknown_score_category",
            f"{base_path}.score_category",
            "observation score category is absent from its rubric",
        )


def validate_observations(
    observations: Iterable[ScoreObservation],
    *,
    assessment: AssessmentSpec,
    rubrics: Iterable[RubricSpecification],
    minimum: int = 1,
    maximum: int = MAX_OBSERVATIONS,
) -> tuple[ScoreObservation, ...]:
    """Validate a bounded observation batch while preserving caller order."""
    minimum_value = _bounded_nonnegative_integer(minimum, "minimum", "$.minimum")
    maximum_value = _bounded_nonnegative_integer(maximum, "maximum", "$.maximum")
    if maximum_value > MAX_OBSERVATIONS:
        raise assessment_error(
            "invalid_maximum",
            "$.maximum",
            f"maximum must not exceed {MAX_OBSERVATIONS}",
        )
    if minimum_value > maximum_value:
        raise assessment_error(
            "invalid_observation_bounds",
            "$.minimum",
            "minimum must not exceed maximum",
        )
    raw = bounded_values(
        observations,
        "observations",
        minimum=minimum_value,
        maximum=maximum_value,
    )
    assessment_value, rubrics_by_fingerprint, constructs_by_rubric = (
        _assessment_context(assessment, rubrics)
    )
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    output: list[ScoreObservation] = []
    for index, observation in enumerate(raw):
        if not isinstance(observation, ScoreObservation):
            raise assessment_error(
                "invalid_score_observation",
                f"$.observations[{index}]",
                "observation entries must be ScoreObservation values",
            )
        _validate_existing_observation(
            observation,
            assessment_value,
            rubrics_by_fingerprint,
            constructs_by_rubric,
            index,
        )
        if observation.observation_id in seen_ids:
            raise assessment_error(
                "duplicate_observation_id",
                f"$.observations[{index}]",
                "observation identifiers must be unique",
            )
        if observation.observation_fingerprint in seen_fingerprints:
            raise assessment_error(
                "duplicate_observation_fingerprint",
                f"$.observations[{index}]",
                "observation fingerprints must be unique",
            )
        seen_ids.add(observation.observation_id)
        seen_fingerprints.add(observation.observation_fingerprint)
        output.append(observation)
    return tuple(output)


__all__ = [
    "MAX_EVIDENCE_OFFSET",
    "MAX_EVIDENCE_SPANS",
    "MAX_OBSERVATIONS",
    "EvidenceSpan",
    "ObservationLevel",
    "ObservationStatus",
    "RaterKind",
    "ScoreObservation",
    "build_score_observation",
    "validate_observations",
]
