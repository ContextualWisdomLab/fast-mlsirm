"""Replay enterprise issue provenance before shared many-facet calibration.

The adapter verifies that one governed request/result/engine execution remains
bound to the exact :class:`AtomicIssueRecord` and then delegates projection to
the existing shared criterion-level calibration contracts. It performs no
likelihood, gradient, Hessian, optimization, scoring, ranking, utility, fairness,
validity, or causal arithmetic.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .._validation import assessment_error, thaw_json_value
from ..calibration import (
    ScoringFacetsRatingRecord,
    build_scoring_facets_rating_records,
)
from ..execution import (
    EngineDescriptor,
    ObservationStatus,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
)
from .contracts import AtomicIssueRecord
from .observation import _enterprise_request_context, _evidence_counts


def _calibration_error(path: str, message: str) -> NoReturn:
    """Raise one stable enterprise calibration replay error."""
    raise assessment_error(
        "enterprise_calibration_provenance_mismatch",
        path,
        message,
    )


def _observation_metadata(
    *,
    issue: AtomicIssueRecord,
    observation: ScoreObservation,
    evidence_fingerprints: tuple[str, ...],
    evidence_counts: tuple[int, int, int],
    path: str,
) -> None:
    """Replay package-managed enterprise observation provenance."""
    supporting, counter, context = evidence_counts
    expected: dict[str, Any] = {
        "enterprise_atomic_issue_fingerprint": issue.atomic_issue_fingerprint,
        "enterprise_issue_content_fingerprint": issue.issue_content_fingerprint,
        "enterprise_observation_evidence_fingerprints": list(
            evidence_fingerprints
        ),
        "enterprise_supporting_evidence_count": supporting,
        "enterprise_counter_evidence_count": counter,
        "enterprise_context_evidence_count": context,
    }
    metadata = thaw_json_value(observation.confidence_metadata)
    if any(metadata.get(key) != value for key, value in expected.items()):
        _calibration_error(
            f"{path}.confidence_metadata",
            "observation enterprise provenance does not replay exactly",
        )


def build_enterprise_issue_facets_rating_records(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one exact enterprise execution into shared facets ratings.

    The function replays the package-managed issue, request, observation, and
    evidence identities before calling
    :func:`~fast_mlsirm.scoring.build_scoring_facets_rating_records`. The shared
    builder remains authoritative for request, result, and engine binding.
    Returned records are the existing shared calibration records; no
    enterprise-specific rating, design, fit, or result schema is introduced.

    Passing this boundary proves provenance consistency only. It does not prove
    that an issue is true or material, that a rating is valid or fair, that the
    many-facet model is adequate, or that any intervention is causally effective.
    """
    if type(issue) is not AtomicIssueRecord:
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an exact AtomicIssueRecord",
        )
    if type(request) is not ScoringRequest:
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be an exact ScoringRequest",
        )
    if type(result) is not ScoringResult:
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be an exact ScoringResult",
        )
    if type(engine) is not EngineDescriptor:
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an exact EngineDescriptor",
        )

    (
        atomic_issue_fingerprint,
        issue_content_fingerprint,
        available_evidence_fingerprints,
        counterevidence_declared,
    ) = _enterprise_request_context(request)
    if atomic_issue_fingerprint != issue.atomic_issue_fingerprint:
        _calibration_error(
            "$.request.metadata.enterprise_atomic_issue_fingerprint",
            "request atomic issue provenance does not match the supplied issue",
        )
    if issue_content_fingerprint != issue.issue_content_fingerprint:
        _calibration_error(
            "$.request.metadata.enterprise_issue_content_fingerprint",
            "request issue content provenance does not match the supplied issue",
        )
    if request.respondent_id != issue.issue_id:
        _calibration_error(
            "$.request.respondent_id",
            "request respondent identity does not match the supplied issue",
        )
    if request.response_content_fingerprint != issue.issue_content_fingerprint:
        _calibration_error(
            "$.request.response_content_fingerprint",
            "request response revision does not match the supplied issue",
        )

    for index, observation in enumerate(result.observations):
        path = f"$.result.observations[{index}]"
        if type(observation) is not ScoreObservation:
            raise assessment_error(
                "invalid_score_observation",
                path,
                "result observations must contain exact ScoreObservation values",
            )
        evidence_fingerprints = tuple(
            value.evidence_fingerprint for value in observation.evidence_references
        )
        if not set(evidence_fingerprints).issubset(
            available_evidence_fingerprints
        ):
            _calibration_error(
                f"{path}.evidence_references",
                "observation evidence is not declared by the enterprise request",
            )
        evidence_counts = _evidence_counts(observation.evidence_references)
        supporting, counter, _ = evidence_counts
        if observation.status is not ObservationStatus.ABSTAINED:
            if supporting == 0:
                _calibration_error(
                    f"{path}.evidence_references",
                    "non-abstained enterprise observations require supporting evidence",
                )
            if counterevidence_declared and counter == 0:
                _calibration_error(
                    f"{path}.evidence_references",
                    "enterprise observations must represent declared counterevidence",
                )
        _observation_metadata(
            issue=issue,
            observation=observation,
            evidence_fingerprints=evidence_fingerprints,
            evidence_counts=evidence_counts,
            path=path,
        )

    return build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )


__all__ = ["build_enterprise_issue_facets_rating_records"]
