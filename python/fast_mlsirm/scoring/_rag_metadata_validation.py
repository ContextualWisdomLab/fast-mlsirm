"""Callback-safe preflight for governed RAG request metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import rag as _base
from ._contract_safety import freeze_metadata
from ._validation import AssessmentSpecError, assessment_error

_ORIGINAL_RAG_METADATA = _base._rag_metadata


def _preflight_rag_metadata(value: Any) -> Any:
    """Freeze caller metadata before RAG allowlist membership is evaluated."""
    try:
        return freeze_metadata({} if value is None else value)
    except AssessmentSpecError as exc:
        if exc.code == "sensitive_metadata_field":
            raise assessment_error(
                "unsupported_rag_metadata",
                "$.metadata",
                "metadata key is not allowed for RAG scoring requests",
            ) from None
        if exc.code == "invalid_metadata_mapping":
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata keys could not be inspected safely",
            ) from None
        raise


def _rag_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    evidence_regime: _base.RAGEvidenceRegime,
    candidate_visibility: _base.RAGCandidateVisibility,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    retrieval_run_fingerprint: str,
    query_revision_fingerprint: str,
) -> dict[str, Any]:
    """Build RAG metadata only after callback-safe immutable preflight."""
    return _ORIGINAL_RAG_METADATA(
        metadata=_preflight_rag_metadata(metadata),
        evidence_regime=evidence_regime,
        candidate_visibility=candidate_visibility,
        system_configuration_id=system_configuration_id,
        system_configuration_fingerprint=system_configuration_fingerprint,
        retrieval_run_fingerprint=retrieval_run_fingerprint,
        query_revision_fingerprint=query_revision_fingerprint,
    )


def install(module: Any) -> None:
    """Install the callback-safe metadata builder on the loaded RAG module."""
    module._rag_metadata = _rag_metadata


__all__: list[str] = []
