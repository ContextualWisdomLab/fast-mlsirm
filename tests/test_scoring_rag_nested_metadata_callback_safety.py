"""Fail-first contracts for nested RAG metadata callback failures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import runpy
from typing import Any

from fast_mlsirm.scoring import AssessmentSpecError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_rag_metadata_callback_safety.py"))
)
_request = _FIXTURES["_request"]
_SECRET = _FIXTURES["_SECRET"]


class _LateNestedMappingTrap(Mapping[str, Any]):
    """Yield one nested entry before forging a package error."""

    def __getitem__(self, key: str) -> Any:
        return "safe" if key == "safe_key" else None

    def __iter__(self) -> Iterator[str]:
        return iter(("safe_key",))

    def __len__(self) -> int:
        return 1

    def items(self) -> Iterator[tuple[str, Any]]:
        yield "safe_key", "safe"
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata.evaluation_split",
            _SECRET,
        )


class _LateNestedListTrap(list[Any]):
    """Yield one nested value before forging a package error."""

    def __iter__(self) -> Iterator[Any]:
        yield "safe"
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata.evaluation_split",
            _SECRET,
        )


def test_nested_mapping_package_error_is_non_reflective() -> None:
    """Nested mapping iterators cannot forge trusted package evidence."""
    try:
        _request({"evaluation_split": _LateNestedMappingTrap()})
    except AssessmentSpecError as caught:
        assert caught.code == "invalid_rag_metadata"
        assert _SECRET not in str(caught)
    else:  # pragma: no cover - fail-first assertion aid
        raise AssertionError("hostile nested mapping unexpectedly succeeded")


def test_nested_collection_package_error_is_non_reflective() -> None:
    """Nested collection iterators cannot forge trusted package evidence."""
    try:
        _request({"evaluation_split": _LateNestedListTrap(["safe", "later"])})
    except AssessmentSpecError as caught:
        assert caught.code == "invalid_metadata_collection"
        assert _SECRET not in str(caught)
    else:  # pragma: no cover - fail-first assertion aid
        raise AssertionError("hostile nested collection unexpectedly succeeded")
