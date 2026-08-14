"""Fail-first contract for single-pass RAG metadata key authorization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import runpy
from typing import Any


_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_rag_metadata_callback_safety.py"))
)
_request = _FIXTURES["_request"]
_SECRET = _FIXTURES["_SECRET"]


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


def test_rag_metadata_authorizes_keys_once_before_reading_allowed_values() -> None:
    """A mapping cannot change key authority between preflight and value capture."""
    metadata = _KeyIterationRace()

    request = _request(metadata)

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"
    assert metadata.iteration_count == 1
