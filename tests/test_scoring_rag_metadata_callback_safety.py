"""Fail-first callback-safety contracts for caller-provided RAG metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ScoringRequest
from fast_mlsirm.scoring.rag import (
    RAGCandidateVisibility,
    RAGEvidenceRegime,
    build_rag_scoring_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

QUERY_FP = hashlib.sha256(b"rag-callback-query").hexdigest()
SYSTEM_FP = hashlib.sha256(b"rag-callback-system").hexdigest()
RETRIEVAL_FP = hashlib.sha256(b"rag-callback-retrieval").hexdigest()
RESPONSE_FP = hashlib.sha256(b"rag-response-content").hexdigest()
_SECRET = "raw_sensitive_callback_payload"


class _ContainsTrap(Mapping[str, Any]):
    """Expose valid entries while rejecting alien membership callbacks."""

    def __init__(self) -> None:
        self._values = {"evaluation_split": "offline_holdout"}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        del key
        raise RuntimeError(_SECRET)


class _IterationTrap(Mapping[str, Any]):
    """Raise sensitive caller text when metadata keys are enumerated."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(_SECRET)

    def __len__(self) -> int:
        return 1

    def __contains__(self, key: object) -> bool:
        del key
        return False


class _LateIterationTrap(Mapping[str, Any]):
    """Yield one valid key before key iteration fails."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        yield "evaluation_split"
        raise RuntimeError(_SECRET)

    def __len__(self) -> int:
        return 2


class _UnsupportedValueTrap(Mapping[str, Any]):
    """Expose one rejected key whose value callback must never be invoked."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        return iter(("unapproved_metadata",))

    def __len__(self) -> int:
        return 1


class _AllowedValueTrap(Mapping[str, Any]):
    """Expose one approved key whose value callback fails."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split",))

    def __len__(self) -> int:
        return 1


class _DuplicateKeyTrap(Mapping[str, Any]):
    """Yield one approved key twice before any value materialization."""

    def __getitem__(self, key: str) -> Any:
        del key
        return "offline_holdout"

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split", "evaluation_split"))

    def __len__(self) -> int:
        return 2


class _NonStringKeyTrap(Mapping[Any, Any]):
    """Expose a non-string key whose value callback must remain unreachable."""

    def __getitem__(self, key: Any) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[Any]:
        return iter((7,))

    def __len__(self) -> int:
        return 1


def _request(metadata: Any) -> ScoringRequest:
    """Build one deterministic request around caller-controlled metadata."""
    return build_rag_scoring_request(
        request_id="rag_callback_request",
        assessment=assessment(),
        rubric=rubric(),
        query_id="refund_policy_query",
        query_revision_fingerprint=QUERY_FP,
        query_testlet_id="evidence_review",
        evidence_regime=RAGEvidenceRegime.RETRIEVED_CONTEXT,
        candidate_visibility=RAGCandidateVisibility.CANDIDATE_BLIND,
        system_configuration_id="retrieval_stack_a",
        system_configuration_fingerprint=SYSTEM_FP,
        system_run_id="retrieval_stack_a_run_001",
        response_id="generated_response_001",
        retrieval_run_fingerprint=RETRIEVAL_FP,
        response_content_fingerprint=RESPONSE_FP,
        occasion_id="evaluation_wave_001",
        criterion_ids=("grounded_generation",),
        response_character_count=120,
        response_unit_count=3,
        metadata=metadata,
    )


def test_rag_metadata_does_not_invoke_alien_membership_callbacks() -> None:
    """Valid mappings must be inspected through bounded keys, not ``__contains__``."""
    request = _request(_ContainsTrap())

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"


def test_rag_metadata_iteration_failure_is_package_owned_and_non_reflective() -> None:
    """Hostile key iteration must not leak arbitrary exceptions or caller text."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_IterationTrap())

    assert caught.value.code == "invalid_rag_metadata"
    assert _SECRET not in str(caught.value)


def test_rag_metadata_late_iteration_failure_is_non_reflective() -> None:
    """A key iterator that fails after yielding data remains inside the boundary."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_LateIterationTrap())

    assert caught.value.code == "invalid_rag_metadata"
    assert _SECRET not in str(caught.value)


def test_rag_metadata_rejects_unknown_keys_before_reading_values() -> None:
    """A disallowed key must not authorize its caller-controlled value callback."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_UnsupportedValueTrap())

    assert caught.value.code == "unsupported_rag_metadata"
    assert _SECRET not in str(caught.value)


def test_rag_metadata_allowed_value_failure_is_non_reflective() -> None:
    """A failing approved value callback becomes stable package evidence."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_AllowedValueTrap())

    assert caught.value.code == "invalid_rag_metadata"
    assert _SECRET not in str(caught.value)


def test_rag_metadata_rejects_duplicate_keys_before_reading_values() -> None:
    """Duplicate key iteration cannot trigger repeated value callbacks."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_DuplicateKeyTrap())

    assert caught.value.code == "duplicate_metadata_key"


def test_rag_metadata_rejects_non_string_keys_before_reading_values() -> None:
    """Non-string key validation precedes any caller-controlled value callback."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_NonStringKeyTrap())

    assert caught.value.code == "invalid_metadata_key"
    assert _SECRET not in str(caught.value)


def test_rag_metadata_non_mapping_preserves_public_type_validation() -> None:
    """The preflight leaves non-mappings for the established public type guard."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(object())

    assert caught.value.code == "invalid_rag_metadata"


def test_rag_metadata_preserves_nested_preflight_errors() -> None:
    """Non-callback metadata violations retain their specific package error codes."""
    cyclic_value: list[Any] = []
    cyclic_value.append(cyclic_value)

    with pytest.raises(AssessmentSpecError) as caught:
        _request({"evaluation_split": cyclic_value})

    assert caught.value.code == "cyclic_metadata_reference"


def test_rag_metadata_none_preserves_the_empty_caller_contract() -> None:
    """The callback-safe preflight retains the established optional metadata path."""
    payload = _request(None).to_dict()["metadata"]

    assert "evaluation_split" not in payload
    assert payload["rag_evidence_regime"] == "retrieved_context"
