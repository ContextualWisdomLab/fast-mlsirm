"""Bounded validation and canonicalization helpers for scoring contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import operator
import re
from types import MappingProxyType
from typing import Any

from fast_mlsirm.rubric.models import _bounded_values, _identifier, _text

MAX_METADATA_COLLECTION_VALUES = 64
MAX_METADATA_DEPTH = 8
MAX_METADATA_NODES = 1_024
MAX_ASSESSMENT_CONSTRUCTS = 32
MAX_ASSESSMENT_RUBRICS = 64
MAX_POLICY_REFERENCES = 64
MAX_RATERS_PER_RESPONSE = 64
MAX_METADATA_KEY_LENGTH = 128
MAX_METADATA_TEXT_LENGTH = 8_192
MIN_SIGNED_INTEGER = -(1 << 63)
MAX_SIGNED_INTEGER = (1 << 63) - 1
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CanonicalContract:
    """Marker base for package-owned contracts with authoritative content."""

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical contract content without derived identities."""
        raise NotImplementedError


def strict_boolean(value: Any, name: str) -> bool:
    """Return a real Boolean without accepting integer coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def bounded_positive_integer(value: Any, name: str, maximum: int) -> int:
    """Return a bounded positive integer while rejecting booleans and fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}") from exc
    if not 1 <= normalized <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return int(normalized)


def fingerprint(value: Any, name: str) -> str:
    """Return a validated lowercase SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character lower hexadecimal digest")
    return normalized


def sorted_identifiers(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_POLICY_REFERENCES,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique descriptive identifiers."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _identifier(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def sorted_fingerprints(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_ASSESSMENT_RUBRICS,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique SHA-256 fingerprints."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        fingerprint(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _metadata_key(value: Any, path: str) -> str:
    """Return one bounded printable metadata key without hidden whitespace."""
    if not isinstance(value, str):
        raise ValueError(f"{path} metadata keys must be strings")
    if not value or value != value.strip():
        raise ValueError(f"{path} metadata keys must be non-empty and trimmed")
    if len(value) > MAX_METADATA_KEY_LENGTH:
        raise ValueError(
            f"{path} metadata keys must contain at most "
            f"{MAX_METADATA_KEY_LENGTH} characters"
        )
    if not value.isprintable():
        raise ValueError(f"{path} metadata keys must not contain control characters")
    return value


def freeze_json_value(
    value: Any,
    path: str,
    *,
    depth: int,
    node_count: list[int],
) -> Any:
    """Validate and deeply freeze one bounded JSON-compatible value."""
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(
            f"{path} exceeds the maximum metadata depth of {MAX_METADATA_DEPTH}"
        )
    node_count[0] += 1
    if node_count[0] > MAX_METADATA_NODES:
        raise ValueError(
            f"{path} exceeds the maximum metadata node count of {MAX_METADATA_NODES}"
        )
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not MIN_SIGNED_INTEGER <= value <= MAX_SIGNED_INTEGER:
            raise ValueError(f"{path} integer metadata must fit signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numeric metadata must be finite")
        return value
    if isinstance(value, str):
        if len(value) > MAX_METADATA_TEXT_LENGTH:
            raise ValueError(
                f"{path} string metadata must contain at most "
                f"{MAX_METADATA_TEXT_LENGTH} characters"
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_COLLECTION_VALUES:
            raise ValueError(
                f"{path} mappings must contain at most "
                f"{MAX_METADATA_COLLECTION_VALUES} values"
            )
        keys = tuple(_metadata_key(raw_key, path) for raw_key in value)
        return MappingProxyType(
            {
                key: freeze_json_value(
                    value[key],
                    f"{path}.{key}",
                    depth=depth + 1,
                    node_count=node_count,
                )
                for key in sorted(keys)
            }
        )
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_METADATA_COLLECTION_VALUES:
            raise ValueError(
                f"{path} collections must contain at most "
                f"{MAX_METADATA_COLLECTION_VALUES} values"
            )
        return tuple(
            freeze_json_value(
                entry,
                f"{path}[{index}]",
                depth=depth + 1,
                node_count=node_count,
            )
            for index, entry in enumerate(value)
        )
    raise ValueError(f"{path} contains an unsupported metadata value")


def freeze_metadata(value: Any) -> MappingProxyType:
    """Return one deeply immutable bounded metadata mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return freeze_json_value(value, "$.metadata", depth=0, node_count=[0])


def thaw_json_value(value: Any) -> Any:
    """Return ordinary JSON-compatible containers from immutable domain values."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(entry) for entry in value]
    return value


def _canonical_payload(value: Any) -> Any:
    """Return package-owned content without calling caller-defined converters."""
    if isinstance(value, CanonicalContract):
        return value._content_dict()
    if value is None or isinstance(
        value,
        (bool, int, float, str, Mapping, list, tuple),
    ):
        return value
    raise ValueError("value contains an unsupported canonical artifact")


def canonical_json(value: Any) -> str:
    """Serialize bounded JSON-compatible content deterministically as UTF-8 text."""
    frozen = freeze_json_value(
        _canonical_payload(value),
        "$",
        depth=0,
        node_count=[0],
    )
    return json.dumps(
        thaw_json_value(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def artifact_digest(value: Any) -> str:
    """Return the SHA-256 identity of one canonical bounded artifact."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
