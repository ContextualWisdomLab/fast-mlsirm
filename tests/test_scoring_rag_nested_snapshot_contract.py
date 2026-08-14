"""Branch-complete contracts for the RAG nested metadata snapshot."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring._rag_metadata_validation import _snapshot_rag_value
from fast_mlsirm.scoring._validation import MAX_METADATA_NODES

_SECRET = "private_nested_snapshot_payload"


class _StringValue(str):
    """String subclass whose ordinary conversion hook is hostile."""

    def __str__(self) -> str:
        raise RuntimeError(_SECRET)


class _IntegerValue(int):
    """Integer subclass whose ordinary conversion hook is hostile."""

    def __int__(self) -> int:
        raise RuntimeError(_SECRET)


class _FloatValue(float):
    """Float subclass whose ordinary conversion hook is hostile."""

    def __float__(self) -> float:
        raise RuntimeError(_SECRET)


class _ItemsCreationTrap(Mapping[str, Any]):
    """Fail before a nested mapping iterator is created."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Any:
        raise RuntimeError(_SECRET)


class _OversizedMapping(Mapping[str, Any]):
    """Yield more entries than the governed nested mapping limit."""

    def __getitem__(self, key: str) -> Any:
        return key

    def __iter__(self) -> Iterator[str]:
        return (f"metadata_key_{index}" for index in range(65))

    def __len__(self) -> int:
        return 65

    def items(self) -> Iterator[tuple[str, Any]]:
        for index in range(65):
            yield f"metadata_key_{index}", index


class _MalformedEntryMapping(Mapping[str, Any]):
    """Yield an entry that cannot be unpacked as one key/value pair."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("metadata_key",))

    def __len__(self) -> int:
        return 1

    def items(self) -> Iterator[Any]:
        yield object()


class _DuplicateEntryMapping(Mapping[str, Any]):
    """Yield the same nested key twice."""

    def __getitem__(self, key: str) -> Any:
        return key

    def __iter__(self) -> Iterator[str]:
        return iter(("metadata_key", "metadata_key"))

    def __len__(self) -> int:
        return 2

    def items(self) -> Iterator[tuple[str, Any]]:
        yield "metadata_key", 1
        yield "metadata_key", 2


class _LengthTrap(list[Any]):
    """Fail while a nested collection size is inspected."""

    def __len__(self) -> int:
        raise RuntimeError(_SECRET)


class _IteratorCreationTrap(list[Any]):
    """Fail before a nested collection iterator is created."""

    def __iter__(self) -> Iterator[Any]:
        raise RuntimeError(_SECRET)


class _DishonestLengthList(list[Any]):
    """Under-report size while yielding an oversized collection."""

    def __len__(self) -> int:
        return 1



def _assert_error(value: Any, code: str) -> None:
    """Assert one stable non-reflective snapshot failure."""
    with pytest.raises(AssessmentSpecError) as caught:
        _snapshot_rag_value(value, "$.metadata")

    assert caught.value.code == code
    assert _SECRET not in str(caught.value)


def test_snapshot_normalizes_scalar_subclasses_and_container_types() -> None:
    """Built-in scalar identities are restored without alien conversion hooks."""
    payload = {
        "string_value": _StringValue("safe"),
        "boolean_value": True,
        "none_value": None,
        "integer_value": _IntegerValue(7),
        "float_value": _FloatValue(1.5),
        "list_value": [1],
        "tuple_value": (2,),
    }

    snapshot = _snapshot_rag_value(payload, "$.metadata")

    assert snapshot == {
        "string_value": "safe",
        "boolean_value": True,
        "none_value": None,
        "integer_value": 7,
        "float_value": 1.5,
        "list_value": [1],
        "tuple_value": (2,),
    }
    assert type(snapshot["string_value"]) is str
    assert type(snapshot["integer_value"]) is int
    assert type(snapshot["float_value"]) is float


def test_snapshot_preserves_unsupported_leaf_for_governed_freeze() -> None:
    """Unsupported leaves remain available to the established JSON validator."""
    value = object()

    assert _snapshot_rag_value(value, "$.metadata") is value


def test_snapshot_rejects_mapping_iterator_creation_failure() -> None:
    """Nested mapping iterator creation is a non-reflective package error."""
    _assert_error(_ItemsCreationTrap(), "invalid_metadata_mapping")


def test_snapshot_rejects_oversized_nested_mapping() -> None:
    """Nested mappings cannot exceed the governed entry count."""
    _assert_error(_OversizedMapping(), "metadata_collection_too_large")


def test_snapshot_rejects_malformed_nested_mapping_entry() -> None:
    """Nested entries must materialize as one key/value pair."""
    _assert_error(_MalformedEntryMapping(), "invalid_metadata_mapping")


def test_snapshot_rejects_duplicate_nested_mapping_key() -> None:
    """A custom mapping cannot collapse duplicate keys during snapshotting."""
    _assert_error(_DuplicateEntryMapping(), "duplicate_metadata_key")


def test_snapshot_rejects_non_string_nested_mapping_key() -> None:
    """Nested mapping keys retain the established string-only contract."""
    _assert_error({7: "value"}, "invalid_metadata_key")


def test_snapshot_rejects_collection_length_failure() -> None:
    """Nested collection length callbacks cannot escape the package boundary."""
    _assert_error(_LengthTrap([1]), "invalid_metadata_collection")


def test_snapshot_rejects_oversized_concrete_collection() -> None:
    """A concrete nested collection fails before its children are traversed."""
    _assert_error([None] * 65, "metadata_collection_too_large")


def test_snapshot_rejects_collection_iterator_creation_failure() -> None:
    """Nested iterator creation failures are stable and non-reflective."""
    _assert_error(_IteratorCreationTrap([1]), "invalid_metadata_collection")


def test_snapshot_rejects_dishonest_collection_length() -> None:
    """Iteration-time bounds defeat a collection that under-reports length."""
    _assert_error(
        _DishonestLengthList([None] * 65),
        "metadata_collection_too_large",
    )


def test_snapshot_enforces_depth_before_deeper_recursion() -> None:
    """Nested RAG metadata cannot recurse beyond the package depth budget."""
    value: Any = "leaf"
    for index in range(10):
        value = {f"nested_level_{index}": value}

    _assert_error(value, "metadata_depth_exceeded")


def test_snapshot_enforces_node_budget_before_freeze() -> None:
    """The defensive snapshot cannot allocate beyond the metadata node budget."""
    payload = {
        f"metadata_group_{group_index}": [None] * 16
        for group_index in range(64)
    }
    assert 1 + sum(1 + len(values) for values in payload.values()) > MAX_METADATA_NODES

    _assert_error(payload, "metadata_node_budget_exceeded")
