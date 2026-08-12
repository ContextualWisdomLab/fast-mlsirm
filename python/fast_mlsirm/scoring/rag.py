"""Governed adapters for reference-free RAG scoring requests.

This module deliberately owns validation and provenance marshalling only. It
reuses the canonical :class:`ScoringRequest` contract and performs no retrieval,
LLM inference, metric arithmetic, thresholding, or truth adjudication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ._contract_safety import artifact_digest, enum_value, freeze_metadata
from ._validation import (
    CanonicalContract,
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


class RAGPerturbationKind(str, Enum):
    """Finite controlled perturbations with preregistered construct semantics."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    EXPLICIT_CONTRADICTION = "explicit_contradiction"
    IRRELEVANT_CONTEXT = "irrelevant_context"
    REQUIRED_EVIDENCE_REMOVAL = "required_evidence_removal"
    CITATION_TARGET_SWAP = "citation_target_swap"
    SEMANTIC_QUERY_PARAPHRASE = "semantic_query_paraphrase"
    STYLE_ONLY_REWRITE = "style_only_rewrite"
    UNANSWERABLE_QUERY = "unanswerable_query"


class RAGPerturbationDirection(str, Enum):
    """Expected direction for one controlled perturbation construct."""

    DECREASE = "decrease"
    INVARIANT = "invariant"
    INCREASE = "increase"


_PERTURBATION_EXPECTATIONS = MappingProxyType(
    {
        RAGPerturbationKind.UNSUPPORTED_CLAIM: (
            "grounded_generation",
            RAGPerturbationDirection.DECREASE,
        ),
        RAGPerturbationKind.EXPLICIT_CONTRADICTION: (
            "grounded_generation",
            RAGPerturbationDirection.DECREASE,
        ),
        RAGPerturbationKind.IRRELEVANT_CONTEXT: (
            "retrieval_relevance",
            RAGPerturbationDirection.DECREASE,
        ),
        RAGPerturbationKind.REQUIRED_EVIDENCE_REMOVAL: (
            "coverage_or_completeness_proxy",
            RAGPerturbationDirection.DECREASE,
        ),
        RAGPerturbationKind.CITATION_TARGET_SWAP: (
            "citation_attribution",
            RAGPerturbationDirection.DECREASE,
        ),
        RAGPerturbationKind.SEMANTIC_QUERY_PARAPHRASE: (
            "robustness",
            RAGPerturbationDirection.INVARIANT,
        ),
        RAGPerturbationKind.STYLE_ONLY_REWRITE: (
            "robustness",
            RAGPerturbationDirection.INVARIANT,
        ),
        RAGPerturbationKind.UNANSWERABLE_QUERY: (
            "answerability_and_abstention",
            RAGPerturbationDirection.INCREASE,
        ),
    }
)


@dataclass(frozen=True)
class RAGPerturbationAnchor(CanonicalContract):
    """Source-free identity and expected direction for a controlled RAG perturbation.

    The anchor binds two distinct governed scoring-request fingerprints to one
    finite perturbation kind. ``expected_construct`` and ``expected_direction``
    are package-derived preregistration metadata; they do not assert that an
    observed system response actually moved in the expected direction.
    """

    anchor_id: str
    baseline_request_fingerprint: str
    perturbed_request_fingerprint: str
    perturbation_kind: RAGPerturbationKind
    expected_construct: str = field(init=False)
    expected_direction: RAGPerturbationDirection = field(init=False)

    def __post_init__(self) -> None:
        """Normalize identity fields and derive immutable perturbation semantics."""
        normalized_anchor_id = descriptive_identifier(
            self.anchor_id,
            "anchor_id",
            "$.anchor_id",
        )
        normalized_baseline = fingerprint(
            self.baseline_request_fingerprint,
            "baseline_request_fingerprint",
            "$.baseline_request_fingerprint",
        )
        normalized_perturbed = fingerprint(
            self.perturbed_request_fingerprint,
            "perturbed_request_fingerprint",
            "$.perturbed_request_fingerprint",
        )
        normalized_kind = enum_value(
            self.perturbation_kind,
            RAGPerturbationKind,
            "rag_perturbation_kind",
            "$.perturbation_kind",
        )
        if normalized_baseline == normalized_perturbed:
            raise assessment_error(
                "identical_rag_perturbation_requests",
                "$.perturbed_request_fingerprint",
                "baseline and perturbed request fingerprints must differ",
            )
        expected_construct, expected_direction = _PERTURBATION_EXPECTATIONS[
            normalized_kind
        ]
        object.__setattr__(self, "anchor_id", normalized_anchor_id)
        object.__setattr__(
            self,
            "baseline_request_fingerprint",
            normalized_baseline,
        )
        object.__setattr__(
            self,
            "perturbed_request_fingerprint",
            normalized_perturbed,
        )
        object.__setattr__(self, "perturbation_kind", normalized_kind)
        object.__setattr__(self, "expected_construct", expected_construct)
        object.__setattr__(self, "expected_direction", expected_direction)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical anchor content without the derived content digest."""
        return {
            "anchor_id": self.anchor_id,
            "baseline_request_fingerprint": self.baseline_request_fingerprint,
            "perturbed_request_fingerprint": self.perturbed_request_fingerprint,
            "perturbation_kind": self.perturbation_kind.value,
            "expected_construct": self.expected_construct,
            "expected_direction": self.expected_direction.value,
        }

    @property
    def anchor_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable perturbation-anchor content."""
        return artifact_digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical anchor content plus its deterministic fingerprint."""
        return {
            **self._content_dict(),
            "anchor_fingerprint": self.anchor_fingerprint,
        }


def build_rag_perturbation_anchor(
    *,
    anchor_id: str,
    baseline_request_fingerprint: str,
    perturbed_request_fingerprint: str,
    perturbation_kind: RAGPerturbationKind | str,
) -> RAGPerturbationAnchor:
    """Build one content-addressed controlled RAG perturbation anchor.

    The contract intentionally stores only bounded identities and package-owned
    preregistration semantics. It does not accept raw query, context, response,
    or source text and performs no scoring, retrieval, or truth adjudication.
    """
    return RAGPerturbationAnchor(
        anchor_id=anchor_id,
        baseline_request_fingerprint=baseline_request_fingerprint,
        perturbed_request_fingerprint=perturbed_request_fingerprint,
        perturbation_kind=perturbation_kind,
    )


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
    normalized_system_configuration_id = descriptive_identifier(
        system_configuration_id,
        "system_configuration_id",
        "$.system_configuration_id",
    )

    rag_metadata = _rag_metadata(
        metadata=metadata,
        evidence_regime=normalized_regime,
        candidate_visibility=normalized_visibility,
        system_configuration_id=normalized_system_configuration_id,
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
    "RAGPerturbationAnchor",
    "RAGPerturbationDirection",
    "RAGPerturbationKind",
    "build_rag_perturbation_anchor",
    "build_rag_scoring_request",
]
