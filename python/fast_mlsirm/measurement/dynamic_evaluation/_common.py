"""Shared vocabulary and validation for dynamic evaluation contracts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Iterable, TypeVar

DYNAMIC_EVALUATION_ITEM_CONTRACT_ID = "fast_mlsirm_dynamic_evaluation_item/v1"
MAX_DYNAMIC_EVALUATION_ITEMS = 10_000
MAX_DYNAMIC_EVALUATION_REFERENCES = 256
MAX_DYNAMIC_EVALUATION_CRITERIA = 128
MAX_DYNAMIC_EVALUATION_CATEGORIES = 64
MAX_DYNAMIC_EVALUATION_REFERENCE_CHARS = 256
_CATEGORY_TOKEN = object()
_CRITERION_TOKEN = object()
_CRITERION_SET_TOKEN = object()
_ITEM_TOKEN = object()
_SET_TOKEN = object()
_ENUM_T = TypeVar("_ENUM_T", bound=Enum)


class DynamicItemOrigin(str, Enum):
    """Provenance class for one concrete evaluation item instance."""

    AUTHORED = "authored"
    GENERATED = "generated"
    PRODUCTION_SAMPLE = "production_sample"
    PERTURBATION = "perturbation"
    SYNTHETIC_ADVERSARIAL = "synthetic_adversarial"


class EvaluationItemRole(str, Enum):
    """Operational role of an item in an evaluation run."""

    CANDIDATE = "candidate"
    ANCHOR = "anchor"
    CHALLENGE = "challenge"
    PRODUCTION_SAMPLE = "production_sample"


class ReferenceSemantics(str, Enum):
    """Meaning of the evidence used to evaluate an item response."""

    EXACT = "exact"
    CONSTRAINT = "constraint"
    ACCEPTABLE_SET = "acceptable_set"
    RUBRIC = "rubric"
    PAIRWISE = "pairwise"
    OPEN_ENDED = "open_ended"


class ReferenceStatus(str, Enum):
    """Governance status of an item's response-reference semantics."""

    UNRESOLVED = "unresolved"
    PROVISIONAL = "provisional"
    ADJUDICATION_REQUIRED = "adjudication_required"
    ADJUDICATED = "adjudicated"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"


class RegenerationStatus(str, Enum):
    """Evidence level for regenerating item content from recorded inputs."""

    UNAVAILABLE = "unavailable"
    INPUTS_RECORDED = "inputs_recorded"
    VERIFIED = "verified"


class LinkingStatus(str, Enum):
    """Comparability claim permitted for one frozen evaluation item set."""

    UNAVAILABLE = "unavailable"
    WITHIN_RUN_ONLY = "within_run_only"
    LINKED = "linked"


class DynamicEvaluationContractError(ValueError):
    """Stable fail-closed error for dynamic-evaluation contract violations."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Retain bounded machine-readable rejection metadata."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def _error(code: str, path: str, message: str) -> DynamicEvaluationContractError:
    """Build one stable contract rejection without echoing caller content."""
    return DynamicEvaluationContractError(code, path, message)


def _enum(value: Any, enum_type: type[_ENUM_T], path: str) -> _ENUM_T:
    """Admit an exact enum or its exact string value without caller protocols."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise TypeError(f"{path} must be a {enum_type.__name__} or exact string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(
            "invalid_enum_value", path, f"unsupported {enum_type.__name__}"
        ) from exc


def _reference(value: Any, path: str) -> str:
    """Validate one exact opaque reference without normalization."""
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if (
        not value
        or len(value) > MAX_DYNAMIC_EVALUATION_REFERENCE_CHARS
        or value != value.strip()
        or value.startswith("\ufeff")
        or value.endswith("\ufeff")
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise _error(
            "invalid_reference",
            path,
            "reference must be 1..256 Unicode scalar values without "
            "boundary whitespace or controls",
        )
    return value


def _optional_reference(value: Any, path: str) -> str | None:
    """Validate an optional opaque reference while preserving explicit absence."""
    if value is None:
        return None
    return _reference(value, path)


def _reference_tuple(
    value: Iterable[str],
    path: str,
    *,
    maximum: int,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Copy, validate, and canonicalize exact reference membership."""
    if type(value) not in (tuple, list):
        raise TypeError(f"{path} must be a tuple or list")
    if (not allow_empty and not value) or len(value) > maximum:
        lower = 0 if allow_empty else 1
        raise _error(
            "invalid_reference_count",
            path,
            f"reference collection must contain {lower}..{maximum} entries",
        )
    normalized = tuple(
        _reference(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise _error(
            "duplicate_reference", path, "reference collection must be unique"
        )
    return tuple(sorted(normalized))


def _sha256(value: Any, path: str) -> str:
    """Validate one complete lowercase hexadecimal SHA-256 digest."""
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise _error(
            "invalid_sha256",
            path,
            "digest must be 64 lowercase hexadecimal characters",
        )
    return value


def _fingerprint(payload: dict[str, Any]) -> str:
    """Hash one canonical JSON-compatible immutable domain payload."""
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()