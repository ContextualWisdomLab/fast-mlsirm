"""Contract-specific fail-closed wrappers around generic scoring validators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import hashlib
import json

from . import _validation as base

_SENSITIVE_METADATA_FIELDS = frozenset(
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


def bounded_positive_integer(
    value: Any,
    name: str,
    maximum: int,
    path: str | None = None,
) -> int:
    """Translate arbitrary numeric callback failures into the stable domain error."""
    try:
        return base.bounded_positive_integer(value, name, maximum, path)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} must be an integer between 1 and {maximum}",
        ) from None


def bounded_values(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    path: str | None = None,
) -> tuple[Any, ...]:
    """Translate arbitrary iterable callback failures into a stable collection error."""
    try:
        return base.bounded_values(
            values,
            name,
            minimum=minimum,
            maximum=maximum,
            path=path,
        )
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} could not be materialized safely",
        ) from None


def sorted_identifiers(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int = base.MAX_POLICY_REFERENCES,
) -> tuple[str, ...]:
    """Return safe sorted identifiers using the callback-hardened materializer."""
    raw = bounded_values(
        values,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    normalized = tuple(
        base.descriptive_identifier(
            value,
            name,
            f"$.{name}[{index}]",
        )
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise base.assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must not contain duplicates",
        )
    return tuple(sorted(normalized))


def _preflight_metadata(value: Any, path: str) -> Any:
    """Copy mappings into built-ins while redacting callbacks and reserved keys."""
    if isinstance(value, Mapping):
        try:
            iterator = iter(value.items())
        except Exception:
            raise base.assessment_error(
                "invalid_metadata_mapping",
                path,
                "metadata mapping entries could not be inspected safely",
            ) from None
        output: dict[str, Any] = {}
        try:
            for index, entry in enumerate(iterator):
                if index >= base.MAX_METADATA_COLLECTION_VALUES:
                    raise base.assessment_error(
                        "metadata_collection_too_large",
                        path,
                        (
                            "metadata mappings must contain at most "
                            f"{base.MAX_METADATA_COLLECTION_VALUES} values"
                        ),
                    )
                try:
                    raw_key, child = entry
                except Exception:
                    raise base.assessment_error(
                        "invalid_metadata_mapping",
                        f"{path}.entries[{index}]",
                        "metadata mapping entries must contain one key and value",
                    ) from None
                key_path = f"{path}.keys[{index}]"
                key = base._metadata_key(raw_key, key_path)
                if key.casefold() in _SENSITIVE_METADATA_FIELDS:
                    raise base.assessment_error(
                        "sensitive_metadata_field",
                        key_path,
                        "metadata cannot contain response or source content fields",
                    )
                if key in output:
                    raise base.assessment_error(
                        "duplicate_metadata_key",
                        key_path,
                        "metadata keys must be unique",
                    )
                output[key] = _preflight_metadata(
                    child,
                    f"{path}.values[{index}]",
                )
        except base.AssessmentSpecError:
            raise
        except Exception:
            raise base.assessment_error(
                "invalid_metadata_mapping",
                path,
                "metadata mapping entries could not be materialized safely",
            ) from None
        return output
    if isinstance(value, (list, tuple)):
        return type(value)(
            _preflight_metadata(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    return value


def freeze_metadata(value: Any) -> Any:
    """Return immutable metadata after callback-safe reserved-field preflight."""
    return base.freeze_metadata(_preflight_metadata(value, "$.metadata"))


def canonical_json(value: Any) -> str:
    """Serialize complete contracts without reapplying the metadata-only node cap."""
    if not isinstance(value, base.CanonicalContract):
        return base.canonical_json(value)
    encoded = json.dumps(
        base.thaw_json_value(value._content_dict()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        encoded_bytes = encoded.encode("utf-8")
    except UnicodeEncodeError:
        raise base.assessment_error(
            "invalid_utf8_text",
            "$",
            "text must be valid UTF-8",
        ) from None
    if len(encoded_bytes) > base.MAX_CANONICAL_JSON_CHARACTERS:
        raise base.assessment_error(
            "canonical_json_too_large",
            "$",
            (
                "canonical JSON must contain at most "
                f"{base.MAX_CANONICAL_JSON_CHARACTERS} UTF-8 bytes"
            ),
        )
    return encoded


def artifact_digest(value: Any) -> str:
    """Return SHA-256 over the callback-safe canonical contract representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
