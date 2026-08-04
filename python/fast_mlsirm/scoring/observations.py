"""Provider-neutral score observations and engine-execution provenance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import operator
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._contract_safety import (
    artifact_digest,
    bounded_values,
    freeze_metadata,
    sorted_identifiers,
)
from ._validation import (
    MAX_POLICY_REFERENCES,
    MAX_SIGNED_INTEGER,
    AssessmentSpecError,
    CanonicalContract,
    assessment_error,
    descriptive_identifier,
    enum_value,
    fingerprint,
    semantic_version,
    thaw_json_value,
)
from .assessment import AssessmentResponseType, AssessmentSpec

SCORING_OBSERVATION_SCHEMA_VERSION = "1.0"
MAX_EVIDENCE_REFERENCES = MAX_POLICY_REFERENCES

_ENGINE_TOKEN = object()
_EVIDENCE_TOKEN = object()
_OBSERVATION_TOKEN = object()
_EXECUTION_TOKEN = object()


class RaterKind(str, Enum):
    """Supported human and automated scoring-agent kinds."""

    HUMAN_RATER = "human_rater"
    AUTOMATED_RATER = "automated_rater"


class ObservationState(str, Enum):
    """Explicit scored and non-scored observation outcomes."""

    OBSERVED_SCORE = "observed_score"
    ABSTAINED_SCORE = "abstained_score"
    FAILED_SCORE = "failed_score"
    EXCLUDED_SCORE = "excluded_score"
    NOT_APPLICABLE_SCORE = "not_applicable_score"


def _observation_schema_version(value: Any) -> str:
    """Require the independently versioned observation wire schema."""
    if value != SCORING_OBSERVATION_SCHEMA_VERSION:
        raise assessment_error(
            "invalid_observation_schema_version",
            "$.schema_version",
            (
                "schema_version must match the supported scoring-observation "
                "wire version"
            ),
        )
    return SCORING_OBSERVATION_SCHEMA_VERSION


def _require_assessment(value: Any) -> AssessmentSpec:
    """Return one package-owned assessment specification."""
    if not isinstance(value, AssessmentSpec):
        raise assessment_error(
            "invalid_assessment_spec",
            "$.assessment",
            "assessment must be a verified AssessmentSpec",
        )
    return value


def _require_rubric(value: Any) -> RubricSpecification:
    """Return one package-owned rubric specification."""
    if not isinstance(value, RubricSpecification):
        raise assessment_error(
            "invalid_observation_rubric",
            "$.rubric",
            "rubric must be a RubricSpecification",
        )
    return value


def _nonnegative_integer(value: Any, name: str, path: str) -> int:
    """Return a signed-64 bounded non-negative integer without coercing Booleans."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a non-negative integer",
        )
    try:
        normalized = operator.index(value)
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must be a non-negative integer",
        ) from None
    if isinstance(normalized, bool) or not 0 <= normalized <= MAX_SIGNED_INTEGER:
        raise assessment_error(
            f"invalid_{name}",
            path,
            f"{name} must fit the non-negative signed-64 range",
        )
    return int(normalized)


def _optional_identifier(value: Any, name: str, path: str) -> str | None:
    """Return an optional descriptive identifier."""
    if value is None:
        return None
    return descriptive_identifier(value, name, path)


def _optional_semantic_version(value: Any, name: str, path: str) -> str | None:
    """Return an optional canonical semantic version."""
    if value is None:
        return None
    return semantic_version(value, name, path)


def _exact_score_category(value: Any) -> int:
    """Return an exact integer category while rejecting Boolean and numeric coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise assessment_error(
            "invalid_score_category",
            "$.score_category",
            "score_category must be one exact rubric integer category",
        )
    return value


def _sorted_evidence_references(
    values: Iterable[Any],
    *,
    minimum: int,
) -> tuple[EvidenceReference, ...]:
    """Return bounded unique evidence references in fingerprint order."""
    raw = bounded_values(
        values,
        "evidence_references",
        minimum=minimum,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    normalized: list[EvidenceReference] = []
    for index, value in enumerate(raw):
        if not isinstance(value, EvidenceReference):
            raise assessment_error(
                "invalid_evidence_reference",
                f"$.evidence_references[{index}]",
                "evidence entries must be verified EvidenceReference values",
            )
        normalized.append(value)
    ordered = tuple(sorted(normalized, key=lambda item: item.evidence_fingerprint))
    fingerprints = tuple(item.evidence_fingerprint for item in ordered)
    if len(set(fingerprints)) != len(fingerprints):
        raise assessment_error(
            "duplicate_evidence_reference",
            "$.evidence_references",
            "evidence references must be unique",
        )
    return ordered


@dataclass(frozen=True)
class ScoringEngineDescriptor(CanonicalContract):
    """Factory-sealed scoring-engine identity and bounded configuration contract."""

    engine_id: str
    engine_family: str
    engine_version: str
    rater_kind: RaterKind
    assessment_fingerprint: str
    assessment_handle: str
    prompt_template_version: str | None
    configuration: Mapping[str, Any]
    schema_version: str = SCORING_OBSERVATION_SCHEMA_VERSION
    _engine_token: InitVar[object | None] = None

    def __post_init__(self, _engine_token: object | None) -> None:
        """Reject direct construction and normalize immutable engine content."""
        if _engine_token is not _ENGINE_TOKEN:
            raise assessment_error(
                "unverified_scoring_engine",
                "$",
                "ScoringEngineDescriptor must be created by build_scoring_engine_descriptor",
            )
        object.__setattr__(
            self,
            "engine_id",
            descriptive_identifier(self.engine_id, "engine_id"),
        )
        object.__setattr__(
            self,
            "engine_family",
            descriptive_identifier(self.engine_family, "engine_family"),
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
            "assessment_fingerprint",
            fingerprint(self.assessment_fingerprint, "assessment_fingerprint"),
        )
        object.__setattr__(
            self,
            "assessment_handle",
            descriptive_identifier(self.assessment_handle, "assessment_handle"),
        )
        object.__setattr__(
            self,
            "prompt_template_version",
            _optional_semantic_version(
                self.prompt_template_version,
                "prompt_template_version",
                "$.prompt_template_version",
            ),
        )
        object.__setattr__(self, "configuration", freeze_metadata(self.configuration))
        object.__setattr__(
            self,
            "schema_version",
            _observation_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return authoritative engine content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "engine_id": self.engine_id,
            "engine_family": self.engine_family,
            "engine_version": self.engine_version,
            "rater_kind": self.rater_kind.value,
            "assessment_fingerprint": self.assessment_fingerprint,
            "assessment_handle": self.assessment_handle,
            "prompt_template_version": self.prompt_template_version,
            "configuration": thaw_json_value(self.configuration),
        }

    @property
    def engine_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable engine descriptor."""
        return artifact_digest(self)

    @property
    def engine_handle(self) -> str:
        """Return the descriptive 128-bit public engine handle."""
        return f"scoring_engine_{self.engine_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical engine content and deterministic identities."""
        return {
            **self._content_dict(),
            "engine_handle": self.engine_handle,
            "engine_fingerprint": self.engine_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceReference(CanonicalContract):
    """Factory-sealed reference to an exact evidence span without raw content."""

    reference_id: str
    source_id: str
    start_offset: int
    end_offset: int
    content_digest: str
    metadata: Mapping[str, Any]
    schema_version: str = SCORING_OBSERVATION_SCHEMA_VERSION
    _evidence_token: InitVar[object | None] = None

    def __post_init__(self, _evidence_token: object | None) -> None:
        """Reject direct construction and normalize immutable span metadata."""
        if _evidence_token is not _EVIDENCE_TOKEN:
            raise assessment_error(
                "unverified_evidence_reference",
                "$",
                "EvidenceReference must be created by build_evidence_reference",
            )
        object.__setattr__(
            self,
            "reference_id",
            descriptive_identifier(self.reference_id, "reference_id"),
        )
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        start = _nonnegative_integer(
            self.start_offset,
            "start_offset",
            "$.start_offset",
        )
        end = _nonnegative_integer(self.end_offset, "end_offset", "$.end_offset")
        if end <= start:
            raise assessment_error(
                "invalid_evidence_range",
                "$",
                "end_offset must be greater than start_offset",
            )
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)
        object.__setattr__(
            self,
            "content_digest",
            fingerprint(self.content_digest, "content_digest"),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            _observation_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return authoritative reference content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "content_digest": self.content_digest,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact evidence reference."""
        return artifact_digest(self)

    @property
    def evidence_handle(self) -> str:
        """Return the descriptive 128-bit public evidence handle."""
        return f"evidence_reference_{self.evidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical reference content and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_handle": self.evidence_handle,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True)
class ScoreObservation(CanonicalContract):
    """Factory-sealed criterion or holistic score-state observation."""

    observation_id: str
    respondent_id: str
    item_id: str
    rater_id: str
    construct_id: str
    criterion_id: str | None
    assessment_fingerprint: str
    assessment_handle: str
    rubric_id: str
    rubric_fingerprint: str
    engine_id: str
    engine_fingerprint: str
    engine_handle: str
    rater_kind: RaterKind
    state: ObservationState
    score_category: int | None
    evidence_references: tuple[EvidenceReference, ...]
    reason_ids: tuple[str, ...]
    uncertainty_metadata: Mapping[str, Any]
    schema_version: str = SCORING_OBSERVATION_SCHEMA_VERSION
    _observation_token: InitVar[object | None] = None

    def __post_init__(self, _observation_token: object | None) -> None:
        """Reject direct construction and normalize immutable observation content."""
        if _observation_token is not _OBSERVATION_TOKEN:
            raise assessment_error(
                "unverified_score_observation",
                "$",
                "ScoreObservation must be created by build_score_observation",
            )
        for field_name in (
            "observation_id",
            "respondent_id",
            "item_id",
            "rater_id",
            "construct_id",
            "assessment_handle",
            "rubric_id",
            "engine_id",
            "engine_handle",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "criterion_id",
            _optional_identifier(self.criterion_id, "criterion_id", "$.criterion_id"),
        )
        for field_name in (
            "assessment_fingerprint",
            "rubric_fingerprint",
            "engine_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "rater_kind",
            enum_value(self.rater_kind, RaterKind, "rater_kind"),
        )
        object.__setattr__(
            self,
            "state",
            enum_value(self.state, ObservationState, "state"),
        )
        if self.score_category is not None:
            object.__setattr__(
                self,
                "score_category",
                _exact_score_category(self.score_category),
            )
        object.__setattr__(
            self,
            "evidence_references",
            _sorted_evidence_references(self.evidence_references, minimum=0),
        )
        object.__setattr__(
            self,
            "reason_ids",
            sorted_identifiers(self.reason_ids, "reason_ids", minimum=0),
        )
        object.__setattr__(
            self,
            "uncertainty_metadata",
            freeze_metadata(self.uncertainty_metadata),
        )
        object.__setattr__(
            self,
            "schema_version",
            _observation_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return authoritative observation content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "respondent_id": self.respondent_id,
            "item_id": self.item_id,
            "rater_id": self.rater_id,
            "construct_id": self.construct_id,
            "criterion_id": self.criterion_id,
            "assessment_fingerprint": self.assessment_fingerprint,
            "assessment_handle": self.assessment_handle,
            "rubric_id": self.rubric_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "engine_id": self.engine_id,
            "engine_fingerprint": self.engine_fingerprint,
            "engine_handle": self.engine_handle,
            "rater_kind": self.rater_kind.value,
            "state": self.state.value,
            "score_category": self.score_category,
            "evidence_references": [
                reference.to_dict() for reference in self.evidence_references
            ],
            "reason_ids": list(self.reason_ids),
            "uncertainty_metadata": thaw_json_value(self.uncertainty_metadata),
        }

    @property
    def observation_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable observation."""
        return artifact_digest(self)

    @property
    def observation_handle(self) -> str:
        """Return the descriptive 128-bit public observation handle."""
        return f"score_observation_{self.observation_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical observation content and deterministic identities."""
        return {
            **self._content_dict(),
            "observation_handle": self.observation_handle,
            "observation_fingerprint": self.observation_fingerprint,
        }


@dataclass(frozen=True)
class ScoringExecution(CanonicalContract):
    """Factory-sealed provenance for admitting one exact score observation."""

    execution_id: str
    assessment_fingerprint: str
    assessment_handle: str
    engine_id: str
    engine_fingerprint: str
    engine_handle: str
    observation_id: str
    observation_fingerprint: str
    observation_handle: str
    execution_metadata: Mapping[str, Any]
    schema_version: str = SCORING_OBSERVATION_SCHEMA_VERSION
    _execution_token: InitVar[object | None] = None

    def __post_init__(self, _execution_token: object | None) -> None:
        """Reject direct construction and normalize immutable execution content."""
        if _execution_token is not _EXECUTION_TOKEN:
            raise assessment_error(
                "unverified_scoring_execution",
                "$",
                "ScoringExecution must be created by build_scoring_execution",
            )
        for field_name in (
            "execution_id",
            "assessment_handle",
            "engine_id",
            "engine_handle",
            "observation_id",
            "observation_handle",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        for field_name in (
            "assessment_fingerprint",
            "engine_fingerprint",
            "observation_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "execution_metadata",
            freeze_metadata(self.execution_metadata),
        )
        object.__setattr__(
            self,
            "schema_version",
            _observation_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return authoritative execution content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "assessment_fingerprint": self.assessment_fingerprint,
            "assessment_handle": self.assessment_handle,
            "engine_id": self.engine_id,
            "engine_fingerprint": self.engine_fingerprint,
            "engine_handle": self.engine_handle,
            "observation_id": self.observation_id,
            "observation_fingerprint": self.observation_fingerprint,
            "observation_handle": self.observation_handle,
            "execution_metadata": thaw_json_value(self.execution_metadata),
        }

    @property
    def execution_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable execution provenance."""
        return artifact_digest(self)

    @property
    def execution_handle(self) -> str:
        """Return the descriptive 128-bit public execution handle."""
        return f"scoring_execution_{self.execution_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical execution content and deterministic identities."""
        return {
            **self._content_dict(),
            "execution_handle": self.execution_handle,
            "execution_fingerprint": self.execution_fingerprint,
        }


def build_scoring_engine_descriptor(
    *,
    assessment: AssessmentSpec,
    engine_id: str,
    engine_family: str,
    engine_version: str,
    rater_kind: RaterKind | str,
    prompt_template_version: str | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> ScoringEngineDescriptor:
    """Build one assessment-authorized human or automated engine descriptor."""
    selected_assessment = _require_assessment(assessment)
    selected_kind = enum_value(rater_kind, RaterKind, "rater_kind")
    if selected_kind is RaterKind.HUMAN_RATER:
        if not selected_assessment.engine_policy.allow_human_raters:
            raise assessment_error(
                "human_rater_disabled",
                "$.rater_kind",
                "the assessment does not allow human raters",
            )
    elif not selected_assessment.engine_policy.allow_automated_raters:
        raise assessment_error(
            "automated_rater_disabled",
            "$.rater_kind",
            "the assessment does not allow automated raters",
        )
    selected_engine_id = descriptive_identifier(engine_id, "engine_id")
    if selected_engine_id not in selected_assessment.engine_policy.engine_ids:
        raise assessment_error(
            "unknown_scoring_engine",
            "$.engine_id",
            "engine_id is absent from the assessment engine policy",
        )
    return ScoringEngineDescriptor(
        engine_id=selected_engine_id,
        engine_family=engine_family,
        engine_version=engine_version,
        rater_kind=selected_kind,
        assessment_fingerprint=selected_assessment.assessment_fingerprint,
        assessment_handle=selected_assessment.assessment_handle,
        prompt_template_version=prompt_template_version,
        configuration={} if configuration is None else configuration,
        schema_version=SCORING_OBSERVATION_SCHEMA_VERSION,
        _engine_token=_ENGINE_TOKEN,
    )


def build_evidence_reference(
    *,
    reference_id: str,
    source_id: str,
    start_offset: int,
    end_offset: int,
    content_digest: str,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceReference:
    """Build one bounded content-free evidence span reference."""
    return EvidenceReference(
        reference_id=reference_id,
        source_id=source_id,
        start_offset=start_offset,
        end_offset=end_offset,
        content_digest=content_digest,
        metadata={} if metadata is None else metadata,
        schema_version=SCORING_OBSERVATION_SCHEMA_VERSION,
        _evidence_token=_EVIDENCE_TOKEN,
    )


def build_score_observation(
    *,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    engine: ScoringEngineDescriptor,
    observation_id: str,
    respondent_id: str,
    item_id: str,
    rater_id: str,
    construct_id: str,
    criterion_id: str | None,
    state: ObservationState | str,
    score_category: int | None = None,
    evidence_references: Iterable[EvidenceReference] = (),
    reason_ids: Iterable[str] = (),
    uncertainty_metadata: Mapping[str, Any] | None = None,
) -> ScoreObservation:
    """Build one exact assessment-, rubric-, and engine-bound score observation."""
    selected_assessment = _require_assessment(assessment)
    selected_rubric = _require_rubric(rubric)
    if not isinstance(engine, ScoringEngineDescriptor):
        raise assessment_error(
            "invalid_scoring_engine",
            "$.engine",
            "engine must be a verified ScoringEngineDescriptor",
        )
    if (
        engine.assessment_fingerprint != selected_assessment.assessment_fingerprint
        or engine.assessment_handle != selected_assessment.assessment_handle
    ):
        raise assessment_error(
            "engine_assessment_mismatch",
            "$.engine",
            "engine descriptor belongs to a different assessment",
        )

    try:
        selected_rubric_fingerprint = selected_rubric.fingerprint
    except Exception:
        raise assessment_error(
            "invalid_observation_rubric",
            "$.rubric",
            "rubric fingerprint could not be computed safely",
        ) from None
    if selected_rubric_fingerprint not in selected_assessment.rubric_fingerprints:
        raise assessment_error(
            "unknown_observation_rubric",
            "$.rubric",
            "rubric fingerprint is absent from the assessment",
        )

    selected_construct_id = descriptive_identifier(construct_id, "construct_id")
    if selected_rubric.construct_id != selected_construct_id:
        raise assessment_error(
            "observation_construct_mismatch",
            "$.construct_id",
            "rubric construct does not match the observation construct",
        )
    construct = next(
        (
            value
            for value in selected_assessment.constructs
            if value.construct_id == selected_construct_id
        ),
        None,
    )
    if (
        construct is None
        or selected_rubric_fingerprint not in construct.rubric_fingerprints
    ):
        raise assessment_error(
            "observation_construct_mismatch",
            "$.construct_id",
            "assessment construct does not bind the selected rubric",
        )

    selected_criterion_id = _optional_identifier(
        criterion_id,
        "criterion_id",
        "$.criterion_id",
    )
    if selected_assessment.response_type is AssessmentResponseType.CRITERION_LEVEL:
        if selected_criterion_id is None:
            raise assessment_error(
                "criterion_id_required",
                "$.criterion_id",
                "criterion-level observations require criterion_id",
            )
    elif selected_assessment.response_type is AssessmentResponseType.HOLISTIC:
        if selected_criterion_id is not None:
            raise assessment_error(
                "criterion_id_not_allowed",
                "$.criterion_id",
                "holistic observations cannot declare criterion_id",
            )

    selected_state = enum_value(state, ObservationState, "state")
    raw_evidence = _sorted_evidence_references(
        evidence_references,
        minimum=1 if selected_state is ObservationState.OBSERVED_SCORE else 0,
    )
    raw_reasons = sorted_identifiers(
        reason_ids,
        "reason_ids",
        minimum=0 if selected_state is ObservationState.OBSERVED_SCORE else 1,
    )
    if selected_state is ObservationState.OBSERVED_SCORE:
        if score_category is None:
            raise assessment_error(
                "observed_score_required",
                "$.score_category",
                "observed_score requires one rubric category",
            )
        selected_score = _exact_score_category(score_category)
        allowed_scores = tuple(level.score for level in selected_rubric.levels)
        if selected_score not in allowed_scores:
            raise assessment_error(
                "invalid_score_category",
                "$.score_category",
                "score_category is absent from the exact rubric levels",
            )
        if not raw_evidence:
            raise assessment_error(
                "observed_evidence_required",
                "$.evidence_references",
                "observed_score requires at least one evidence reference",
            )
    else:
        if score_category is not None:
            raise assessment_error(
                "score_not_allowed",
                "$.score_category",
                "non-observed states cannot contain score_category",
            )
        selected_score = None
        if not raw_reasons:
            raise assessment_error(
                "observation_reason_required",
                "$.reason_ids",
                "non-observed states require at least one reason identifier",
            )

    return ScoreObservation(
        observation_id=observation_id,
        respondent_id=respondent_id,
        item_id=item_id,
        rater_id=rater_id,
        construct_id=selected_construct_id,
        criterion_id=selected_criterion_id,
        assessment_fingerprint=selected_assessment.assessment_fingerprint,
        assessment_handle=selected_assessment.assessment_handle,
        rubric_id=selected_rubric.rubric_id,
        rubric_fingerprint=selected_rubric_fingerprint,
        engine_id=engine.engine_id,
        engine_fingerprint=engine.engine_fingerprint,
        engine_handle=engine.engine_handle,
        rater_kind=engine.rater_kind,
        state=selected_state,
        score_category=selected_score,
        evidence_references=raw_evidence,
        reason_ids=raw_reasons,
        uncertainty_metadata=(
            {} if uncertainty_metadata is None else uncertainty_metadata
        ),
        schema_version=SCORING_OBSERVATION_SCHEMA_VERSION,
        _observation_token=_OBSERVATION_TOKEN,
    )


def build_scoring_execution(
    *,
    assessment: AssessmentSpec,
    engine: ScoringEngineDescriptor,
    observation: ScoreObservation,
    execution_id: str,
    execution_metadata: Mapping[str, Any] | None = None,
) -> ScoringExecution:
    """Build provenance binding one exact assessment, engine, and observation."""
    selected_assessment = _require_assessment(assessment)
    if not isinstance(engine, ScoringEngineDescriptor):
        raise assessment_error(
            "invalid_scoring_engine",
            "$.engine",
            "engine must be a verified ScoringEngineDescriptor",
        )
    if not isinstance(observation, ScoreObservation):
        raise assessment_error(
            "invalid_score_observation",
            "$.observation",
            "observation must be a verified ScoreObservation",
        )
    if (
        engine.assessment_fingerprint != selected_assessment.assessment_fingerprint
        or engine.assessment_handle != selected_assessment.assessment_handle
        or observation.assessment_fingerprint
        != selected_assessment.assessment_fingerprint
        or observation.assessment_handle != selected_assessment.assessment_handle
    ):
        raise assessment_error(
            "execution_assessment_mismatch",
            "$.assessment",
            "assessment, engine, and observation provenance must match",
        )
    if (
        observation.engine_id != engine.engine_id
        or observation.engine_fingerprint != engine.engine_fingerprint
        or observation.engine_handle != engine.engine_handle
        or observation.rater_kind is not engine.rater_kind
    ):
        raise assessment_error(
            "execution_engine_mismatch",
            "$.engine",
            "observation belongs to a different scoring engine",
        )
    return ScoringExecution(
        execution_id=execution_id,
        assessment_fingerprint=selected_assessment.assessment_fingerprint,
        assessment_handle=selected_assessment.assessment_handle,
        engine_id=engine.engine_id,
        engine_fingerprint=engine.engine_fingerprint,
        engine_handle=engine.engine_handle,
        observation_id=observation.observation_id,
        observation_fingerprint=observation.observation_fingerprint,
        observation_handle=observation.observation_handle,
        execution_metadata={} if execution_metadata is None else execution_metadata,
        schema_version=SCORING_OBSERVATION_SCHEMA_VERSION,
        _execution_token=_EXECUTION_TOKEN,
    )


__all__ = [
    "MAX_EVIDENCE_REFERENCES",
    "SCORING_OBSERVATION_SCHEMA_VERSION",
    "EvidenceReference",
    "ObservationState",
    "RaterKind",
    "ScoreObservation",
    "ScoringEngineDescriptor",
    "ScoringExecution",
    "build_evidence_reference",
    "build_score_observation",
    "build_scoring_engine_descriptor",
    "build_scoring_execution",
]
