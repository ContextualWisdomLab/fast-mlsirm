"""Governed adapters for reference-free RAG scoring requests.

This module deliberately owns validation and provenance marshalling only. It
reuses the canonical :class:`ScoringRequest` contract and performs no retrieval,
LLM inference, metric arithmetic, thresholding, or truth adjudication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from ._contract_safety import enum_value, freeze_metadata
from ._validation import (
    assessment_error,
    descriptive_identifier,
    fingerprint,
    thaw_json_value,
)
from .assessment import AssessmentSpec
from .authorization import build_scoring_request
from .execution import ObservationGranularity, ScoringRequest
from fast_mlsirm.rubric.models import RubricSpecification


class RAGEvidenceRegime(str, Enum):
    """Evidence available to one governed reference-free RAG evaluation."""

    PROMPT_ONLY = "prompt_only"
    RETRIEVED_CONTEXT = "retrieved_context"
    POOLED_CORPUS = "pooled_corpus"
    AUTHORITATIVE_CORPUS = "authoritative_corpus"
    HUMAN_ANCHOR = "human_anchor"


class RAGCandidateVisibility(str, Enum):
    """Whether scoring evidence can expose candidate-system identity/content."""

    CANDIDATE_BLIND = "candidate_blind"
    CANDIDATE_VISIBLE_CROSSFIT = "candidate_visible_crossfit"


_MANAGED_METADATA_KEYS = frozenset(
    {
        "rag_evidence_regime",
        "rag_candidate_visibility",
        "rag_system_configuration_id",
        "rag_system_configuration_fingerprint",
        "rag_retrieval_run_fingerprint",
        "rag_query_revision_fingerprint",
    }
)
_ALLOWED_CALLER_METADATA_KEYS = frozenset({"evaluation_split"})


def _rag_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    evidence_regime: RAGEvidenceRegime,
    candidate_visibility: RAGCandidateVisibility,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    retrieval_run_fingerprint: str,
    query_revision_fingerprint: str,
) -> dict[str, Any]:
    """Return allowlisted caller metadata plus package-managed RAG provenance."""
    raw_metadata: Mapping[str, Any] = {} if metadata is None else metadata
    if not isinstance(raw_metadata, Mapping):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    if any(key in raw_metadata for key in _MANAGED_METADATA_KEYS):
        raise assessment_error(
            "reserved_rag_metadata",
            "$.metadata",
            "RAG provenance metadata is package-managed",
        )
    if any(key not in _ALLOWED_CALLER_METADATA_KEYS for key in raw_metadata):
        raise assessment_error(
            "unsupported_rag_metadata",
            "$.metadata",
            "metadata key is not allowed for RAG scoring requests",
        )
    frozen = freeze_metadata(raw_metadata)
    if not isinstance(frozen, Mapping):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    output = thaw_json_value(frozen)
    if not isinstance(output, dict):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    if "evaluation_split" in output:
        output["evaluation_split"] = descriptive_identifier(
            output["evaluation_split"],
            "evaluation_split",
            "$.metadata.evaluation_split",
        )
    output.update(
        {
            "rag_evidence_regime": evidence_regime.value,
            "rag_candidate_visibility": candidate_visibility.value,
            "rag_system_configuration_id": system_configuration_id,
            "rag_system_configuration_fingerprint": system_configuration_fingerprint,
            "rag_retrieval_run_fingerprint": retrieval_run_fingerprint,
            "rag_query_revision_fingerprint": query_revision_fingerprint,
        }
    )
    return output


def build_rag_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    query_id: str,
    query_revision_fingerprint: str,
    query_testlet_id: str,
    evidence_regime: RAGEvidenceRegime | str,
    candidate_visibility: RAGCandidateVisibility | str,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    system_run_id: str,
    response_id: str,
    retrieval_run_fingerprint: str,
    response_content_fingerprint: str,
    occasion_id: str,
    criterion_ids: Iterable[str] = (),
    response_character_count: int,
    response_unit_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> ScoringRequest:
    """Build one provenance-bound criterion-level RAG scoring request.

    The RAG-specific identities are projected onto the shared scoring axes:
    stochastic system run -> respondent, generated answer -> response, query ->
    task, exact query revision -> task revision, and query testlet -> task
    family. System configuration identity, evidence regime, candidate visibility,
    retrieval-run identity, and exact configuration/revision fingerprints are
    package-managed metadata and therefore participate in the canonical request
    fingerprint.

    Raw query, context, answer, and source text are intentionally absent from
    this interface. Caller metadata is intentionally allowlisted to prevent raw
    content from being smuggled into the canonical artifact. A reference-free
    request records its evidence regime; it does not imply world correctness,
    absolute retrieval recall, or deployment validity.
    """
    normalized_regime = enum_value(
        evidence_regime,
        RAGEvidenceRegime,
        "rag_evidence_regime",
        "$.evidence_regime",
    )
    normalized_visibility = enum_value(
        candidate_visibility,
        RAGCandidateVisibility,
        "rag_candidate_visibility",
        "$.candidate_visibility",
    )
    normalized_query_revision = fingerprint(
        query_revision_fingerprint,
        "query_revision_fingerprint",
        "$.query_revision_fingerprint",
    )
    normalized_system_fingerprint = fingerprint(
        system_configuration_fingerprint,
        "system_configuration_fingerprint",
        "$.system_configuration_fingerprint",
    )
    normalized_retrieval_fingerprint = fingerprint(
        retrieval_run_fingerprint,
        "retrieval_run_fingerprint",
        "$.retrieval_run_fingerprint",
    )

    rag_metadata = _rag_metadata(
        metadata=metadata,
        evidence_regime=normalized_regime,
        candidate_visibility=normalized_visibility,
        system_configuration_id=system_configuration_id,
        system_configuration_fingerprint=normalized_system_fingerprint,
        retrieval_run_fingerprint=normalized_retrieval_fingerprint,
        query_revision_fingerprint=normalized_query_revision,
    )

    return build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id=system_run_id,
        response_id=response_id,
        task_id=query_id,
        task_revision_fingerprint=normalized_query_revision,
        task_family_id=query_testlet_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=response_content_fingerprint,
        response_character_count=response_character_count,
        response_unit_count=response_unit_count,
        metadata=rag_metadata,
    )


__all__ = [
    "RAGCandidateVisibility",
    "RAGEvidenceRegime",
    "build_rag_scoring_request",
]
