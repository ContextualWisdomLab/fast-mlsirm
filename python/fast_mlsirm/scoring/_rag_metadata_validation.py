"""Callback-safe preflight for governed RAG request metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import rag as _base
from ._contract_safety import freeze_metadata
from ._validation import AssessmentSpecError, _metadata_key, assessment_error

_ORIGINAL_RAG_METADATA = _base._rag_metadata


def _rag_metadata_keys(value: Mapping[Any, Any]) -> tuple[str, ...]:
    """Validate caller keys before any mapping value is requested."""
    try:
        iterator = iter(value)
    except Exception:
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata keys could not be inspected safely",
        ) from None

    output: list[str] = []
    seen: set[str] = set()
    try:
        for index, raw_key in enumerate(iterator):
            try:
                validated_key = _metadata_key(
                    raw_key,
                    f"$.metadata.keys[{index}]",
                )
            except AssessmentSpecError as exc:
                if exc.code == "sensitive_metadata_field":
                    raise assessment_error(
                        "unsupported_rag_metadata",
                        "$.metadata",
                        "metadata key is not allowed for RAG scoring requests",
                    ) from None
                raise
            key = str.__str__(validated_key)
            if key in seen:
                raise assessment_error(
                    "duplicate_metadata_key",
                    f"$.metadata.keys[{index}]",
                    "metadata keys must be unique",
                )
            seen.add(key)
            if key in _base._MANAGED_METADATA_KEYS:
                raise assessment_error(
                    "reserved_rag_metadata",
                    "$.metadata",
                    "RAG provenance metadata is package-managed",
                )
            if key not in _base._ALLOWED_CALLER_METADATA_KEYS:
                raise assessment_error(
                    "unsupported_rag_metadata",
                    "$.metadata",
                    "metadata key is not allowed for RAG scoring requests",
                )
            output.append(key)
    except AssessmentSpecError:
        raise
    except Exception:
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata keys could not be materialized safely",
        ) from None
    return tuple(output)


def _authorized_metadata_values(
    value: Mapping[Any, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Read each authorized value once without re-enumerating caller keys."""
    output: dict[str, Any] = {}
    for key in keys:
        try:
            output[key] = value[key]
        except AssessmentSpecError:
            raise
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata values could not be inspected safely",
            ) from None
    return output


def _preflight_rag_metadata(value: Any) -> Any:
    """Validate keys once, then freeze only their captured authorized values."""
    raw_metadata = {} if value is None else value
    if not isinstance(raw_metadata, Mapping):
        return raw_metadata
    keys = _rag_metadata_keys(raw_metadata)
    authorized_values = _authorized_metadata_values(raw_metadata, keys)
    try:
        return freeze_metadata(authorized_values)
    except AssessmentSpecError as exc:
        if exc.code == "invalid_metadata_mapping":
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata values could not be inspected safely",
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
