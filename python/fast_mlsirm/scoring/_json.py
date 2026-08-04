"""Bounded immutable JSON normalization for scoring policy metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import json
import math

from ..rubric.models import _identifier
from .errors import contract_error

MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 512
MAX_JSON_COLLECTION_VALUES = 64
MAX_JSON_STRING_CHARACTERS = 2_048
MAX_CANONICAL_JSON_CHARACTERS = 65_536

_FORBIDDEN_CONTENT_KEYS = frozenset(
    {
        "answer_text",
        "essay_text",
        "prompt_text",
        "raw_response",
        "response_content",
        "response_text",
        "source_content",
        "source_text",
    }
)


def _json_key(value: Any, path: str) -> str:
    """Normalize one descriptive metadata key without echoing its raw value."""
    try:
        return _identifier(value, "metadata_key")
    except ValueError as exc:
        raise contract_error("invalid_metadata_key", path, str(exc)) from None


def _normalize_json(
    value: Any,
    path: str,
    depth: int,
    nodes: list[int],
) -> Any:
    """Return one bounded canonical JSON value."""
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        raise contract_error(
            "json_node_budget_exceeded",
            path,
            f"JSON content must contain at most {MAX_JSON_NODES} values",
        )
    if depth > MAX_JSON_DEPTH:
        raise contract_error(
            "json_depth_exceeded",
            path,
            f"JSON content must be at most {MAX_JSON_DEPTH} levels deep",
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise contract_error(
                "non_finite_json_number",
                path,
                "JSON numeric values must be finite",
            )
        return value
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING_CHARACTERS:
            raise contract_error(
                "json_string_too_long",
                path,
                "JSON strings exceed the scoring-contract character budget",
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_JSON_COLLECTION_VALUES:
            raise contract_error(
                "json_collection_too_large",
                path,
                (
                    "JSON objects must contain at most "
                    f"{MAX_JSON_COLLECTION_VALUES} entries"
                ),
            )
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key_path = f"{path}.{raw_key}" if isinstance(raw_key, str) else path
            key = _json_key(raw_key, key_path)
            if key in _FORBIDDEN_CONTENT_KEYS:
                raise contract_error(
                    "sensitive_content_field_forbidden",
                    key_path,
                    "metadata cannot contain response or source text fields",
                )
            if key in output:
                raise contract_error(
                    "duplicate_json_key",
                    key_path,
                    "JSON keys must remain unique after normalization",
                )
            output[key] = _normalize_json(
                child,
                f"{path}.{key}",
                depth + 1,
                nodes,
            )
        return {key: output[key] for key in sorted(output)}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if len(value) > MAX_JSON_COLLECTION_VALUES:
            raise contract_error(
                "json_collection_too_large",
                path,
                (
                    "JSON arrays must contain at most "
                    f"{MAX_JSON_COLLECTION_VALUES} values"
                ),
            )
        return [
            _normalize_json(
                child,
                f"{path}[{index}]",
                depth + 1,
                nodes,
            )
            for index, child in enumerate(value)
        ]
    raise contract_error(
        "unsupported_json_value",
        path,
        (
            "JSON values must use null, booleans, finite numbers, strings, "
            "arrays, or objects"
        ),
    )


def canonical_object_json(
    value: Mapping[str, Any] | None,
    field: str,
) -> str:
    """Return deterministic bounded JSON for one top-level object."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise contract_error(
            f"invalid_{field}",
            f"$.{field}",
            f"{field} must be a mapping",
        )
    normalized = _normalize_json(value, f"$.{field}", 0, [0])
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded) > MAX_CANONICAL_JSON_CHARACTERS:
        raise contract_error(
            f"invalid_{field}",
            f"$.{field}",
            f"{field} exceeds the canonical JSON character budget",
        )
    return encoded


def decode_object_json(value: str, field: str) -> dict[str, Any]:
    """Decode a canonical object stored by a factory-sealed contract."""
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return decoded
