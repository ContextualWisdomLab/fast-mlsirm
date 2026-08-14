"""Fail-first contracts for single-pass RAG metadata authorization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import runpy
from typing import Any

import pytest


_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_rag_metadata_callback_safety.py"))
)
_request = _FIXTURES["_request"]
_SECRET = _FIXTURES["_SECRET"]
AssessmentSpecError = _FIXTURES["AssessmentSpecError"]


class _KeyIterationRace(Mapping[str, Any]):
    """Expose a valid first key pass and reject any second key enumeration."""

    def __init__(self) -> None:
        self.iteration_count = 0

    def __getitem__(self, key: str) -> Any:
        if key == "evaluation_split":
            return "offline_holdout"
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise RuntimeError(_SECRET)
        return iter(("evaluation_split",))

    def __len__(self) -> int:
        return 1


class _PackageErrorValueTrap(Mapping[str, Any]):
    """Raise a caller-created package error containing sensitive callback text."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise AssessmentSpecError(
            "invalid_callback_value",
            "$.metadata.evaluation_split",
            _SECRET,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split",))

    def __len__(self) -> int:
        return 1


def test_rag_metadata_authorizes_keys_once_before_reading_allowed_values() -> None:
    """A mapping cannot change key authority between preflight and value capture."""
    metadata = _KeyIterationRace()

    request = _request(metadata)

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"
    assert metadata.iteration_count == 1


def test_rag_metadata_does_not_trust_caller_raised_package_errors() -> None:
    """A forged package exception from ``__getitem__`` is normalized and redacted."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(_PackageErrorValueTrap())

    assert caught.value.code == "invalid_rag_metadata"
    assert _SECRET not in str(caught.value)
