"""Callback-safe preflight for governed RAG request metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import rag as _base
from ._contract_safety import freeze_metadata
from ._validation import (
    MAX_METADATA_COLLECTION_VALUES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_NODES,
    AssessmentSpecError,
    _metadata_key,
    assessment_error,
)

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
    index = 0
    while True:
        try:
            raw_key = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata keys could not be materialized safely",
            ) from None
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
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata key could not be inspected safely",
            ) from None
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
        index += 1
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
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata values could not be inspected safely",
            ) from None
    return output


def _snapshot_rag_value(
    value: Any,
    path: str,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
    active_containers: set[int] | None = None,
) -> Any:
    """Copy bounded nested metadata without trusting container callbacks."""
    counts = [0] if node_count is None else node_count
    active = set() if active_containers is None else active_containers
    if depth > MAX_METADATA_DEPTH:
        raise assessment_error(
            "metadata_depth_exceeded",
            path,
            f"metadata exceeds the maximum depth of {MAX_METADATA_DEPTH}",
        )
    counts[0] += 1
    if counts[0] > MAX_METADATA_NODES:
        raise assessment_error(
            "metadata_node_budget_exceeded",
            path,
            f"metadata exceeds the maximum node count of {MAX_METADATA_NODES}",
        )

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise assessment_error(
                "cyclic_metadata_reference",
                path,
                "metadata cannot contain cyclic container references",
            )
        active.add(marker)
        try:
            try:
                iterator = iter(value.items())
            except Exception:
                raise assessment_error(
                    "invalid_metadata_mapping",
                    path,
                    "metadata mapping entries could not be inspected safely",
                ) from None
            output: dict[str, Any] = {}
            index = 0
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    raise assessment_error(
                        "invalid_metadata_mapping",
                        path,
                        "metadata mapping entries could not be materialized safely",
                    ) from None
                if index >= MAX_METADATA_COLLECTION_VALUES:
                    raise assessment_error(
                        "metadata_collection_too_large",
                        path,
                        (
                            "metadata mappings must contain at most "
                            f"{MAX_METADATA_COLLECTION_VALUES} values"
                        ),
                    )
                try:
                    raw_key, child = entry
                except Exception:
                    raise assessment_error(
                        "invalid_metadata_mapping",
                        f"{path}.entries[{index}]",
                        "metadata mapping entries must contain one key and value",
                    ) from None
                safe_key = str.__str__(raw_key) if isinstance(raw_key, str) else raw_key
                key = _metadata_key(safe_key, f"{path}.keys[{index}]")
                if key in output:
                    raise assessment_error(
                        "duplicate_metadata_key",
                        f"{path}.keys[{index}]",
                        "metadata keys must be unique",
                    )
                output[key] = _snapshot_rag_value(
                    child,
                    f"{path}.values[{index}]",
                    depth=depth + 1,
                    node_count=counts,
                    active_containers=active,
                )
                index += 1
            return output
        finally:
            active.remove(marker)

    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise assessment_error(
                "cyclic_metadata_reference",
                path,
                "metadata cannot contain cyclic container references",
            )
        active.add(marker)
        try:
            try:
                size = len(value)
            except Exception:
                raise assessment_error(
                    "invalid_metadata_collection",
                    path,
                    "metadata collection size could not be inspected safely",
                ) from None
            if size > MAX_METADATA_COLLECTION_VALUES:
                raise assessment_error(
                    "metadata_collection_too_large",
                    path,
                    (
                        "metadata collections must contain at most "
                        f"{MAX_METADATA_COLLECTION_VALUES} values"
                    ),
                )
            try:
                iterator = iter(value)
            except Exception:
                raise assessment_error(
                    "invalid_metadata_collection",
                    path,
                    "metadata collection could not be inspected safely",
                ) from None
            output: list[Any] = []
            index = 0
            while True:
                try:
                    child = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    raise assessment_error(
                        "invalid_metadata_collection",
                        path,
                        "metadata collection could not be materialized safely",
                    ) from None
                if index >= MAX_METADATA_COLLECTION_VALUES:
                    raise assessment_error(
                        "metadata_collection_too_large",
                        path,
                        (
                            "metadata collections must contain at most "
                            f"{MAX_METADATA_COLLECTION_VALUES} values"
                        ),
                    )
                output.append(
                    _snapshot_rag_value(
                        child,
                        f"{path}[{index}]",
                        depth=depth + 1,
                        node_count=counts,
                        active_containers=active,
                    )
                )
                index += 1
            return tuple(output) if isinstance(value, tuple) else output
        finally:
            active.remove(marker)

    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    return value


def _preflight_rag_metadata(value: Any) -> Any:
    """Validate keys once, snapshot nested values, and freeze built-in data."""
    raw_metadata = {} if value is None else value
    if not isinstance(raw_metadata, Mapping):
        return raw_metadata
    keys = _rag_metadata_keys(raw_metadata)
    authorized_values = _authorized_metadata_values(raw_metadata, keys)
    safe_values = _snapshot_rag_value(authorized_values, "$.metadata")
    try:
        return freeze_metadata(safe_values)
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
