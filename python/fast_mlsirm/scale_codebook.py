"""Explicit, data-independent category keying before Rust-owned IRT fitting.

An item's ``ordered_codes`` lists raw codes from the lowest to the highest
level of the declared construct. This is a codebook lookup, not an estimator:
no keys, item deletions, or signs are chosen from observed correlations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

__all__ = [
    "OrderedItem", "ScaleCodebook", "PreparedScale", "codebook_sha256",
    "prepare_scale_responses",
]
_MAX_ITEMS = 10_000
_MAX_CATEGORIES = 64
_MAX_CELLS = 20_000_000


@dataclass(frozen=True, slots=True)
class OrderedItem:
    """One item identity and raw codes in increasing target-construct order."""

    item_id: str
    ordered_codes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScaleCodebook:
    """Caller-supplied key authority; ``source_ref`` is recorded, not verified.

    All items must declare the same number of categories. Raw and missing
    codes must be distinct signed 32-bit integers. Empty ``missing_codes`` is
    valid; IEEE NaN is always retained as missing. Descriptive fields are
    nonblank exact strings of at most 2,048 characters.
    """

    scale_id: str
    high_score_meaning: str
    source_ref: str
    items: tuple[OrderedItem, ...]
    missing_codes: tuple[int, ...] = (-1,)


@dataclass(frozen=True, slots=True)
class PreparedScale:
    """Keyed, read-only responses in authoritative item order.

    Feed ``responses`` directly to the existing Rust-backed fit/scoring API;
    do not submit this result to the raw-code preparation boundary again.
    The digest identifies the supplied key specification, not its validity
    or the identity of the response data. No rows are removed or reordered.
    """

    responses: np.ndarray
    item_ids: tuple[str, ...]
    n_categories: int
    scale_id: str
    high_score_meaning: str
    codebook_sha256: str


def _text(value: object, name: str) -> None:
    """Reject non-text, blank, or oversized metadata before using protocols."""
    if type(value) is not str or not value.strip() or len(value) > 2_048:
        raise ValueError(f"{name} must be nonblank text of at most 2,048 characters")


def _codes(values: object, name: str, minimum: int) -> None:
    """Admit a bounded exact tuple of distinct signed 32-bit integer codes."""
    if type(values) is not tuple or not minimum <= len(values) <= _MAX_CATEGORIES:
        raise ValueError(f"{name} must be a tuple of {minimum}..64 distinct codes")
    for value in values:
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError(f"{name} must contain exact signed 32-bit integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate codes")


def _validate_codebook(book: ScaleCodebook) -> int:
    """Replay the complete semantic contract at each public trust boundary."""
    if type(book) is not ScaleCodebook:
        raise ValueError("codebook must be an exact ScaleCodebook")
    for name in ("scale_id", "high_score_meaning", "source_ref"):
        _text(getattr(book, name), name)
    if type(book.items) is not tuple or not 1 <= len(book.items) <= _MAX_ITEMS:
        raise ValueError("items must be an exact tuple of 1..10,000 OrderedItem records")
    _codes(book.missing_codes, "missing_codes", 0)
    identities: set[str] = set()
    category_count = 0
    for item in book.items:
        if type(item) is not OrderedItem:
            raise ValueError("items must contain exact OrderedItem records")
        _text(item.item_id, "item_id")
        if item.item_id in identities:
            raise ValueError("codebook contains duplicate item identities")
        identities.add(item.item_id)
        _codes(item.ordered_codes, "ordered_codes", 2)
        if set(item.ordered_codes).intersection(book.missing_codes):
            raise ValueError("an observed code cannot also be a missing code")
        if category_count and len(item.ordered_codes) != category_count:
            raise ValueError("all items must declare the same category count")
        category_count = len(item.ordered_codes)
    return category_count


def codebook_sha256(codebook: ScaleCodebook) -> str:
    """Return a deterministic SHA-256 identity of the validated key contract."""
    _validate_codebook(codebook)
    payload = {
        "schema": "fast_mlsirm.scale_codebook/1",
        "scale_id": codebook.scale_id,
        "high_score_meaning": codebook.high_score_meaning,
        "source_ref": codebook.source_ref,
        "items": [[item.item_id, list(item.ordered_codes)] for item in codebook.items],
        "missing_codes": list(codebook.missing_codes),
    }
    wire = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def prepare_scale_responses(
    responses: np.ndarray,
    column_ids: tuple[str, ...],
    codebook: ScaleCodebook,
) -> PreparedScale:
    """Map raw response codes to ``0..K-1`` using only the declared codebook.

    The input must be an exact nonempty two-dimensional real numeric NumPy
    array, with at most 20,000,000 logical cells. Binary Boolean arrays are
    also admitted. Extended-precision floats, object/text/complex storage,
    infinity, undeclared codes, and ambiguous column identities fail closed.
    Missing responses remain NaN, including item-specific observed-code
    reversal. Every input column must match one codebook item; output columns
    follow codebook order. Statistical arithmetic remains in the Rust core.
    """
    n_categories = _validate_codebook(codebook)
    if type(column_ids) is not tuple or len(column_ids) != len(codebook.items):
        raise ValueError("column_ids must be an exact tuple matching the codebook")
    for item_id in column_ids:
        _text(item_id, "column_id")
    item_ids = tuple(item.item_id for item in codebook.items)
    if len(set(column_ids)) != len(column_ids) or set(column_ids) != set(item_ids):
        raise ValueError("column identities must match the codebook exactly once")
    if type(responses) is not np.ndarray:
        raise ValueError("responses must be an exact NumPy ndarray of raw codes")
    if responses.ndim != 2 or responses.shape[0] == 0 or responses.shape[1] != len(item_ids):
        raise ValueError("responses must be nonempty persons x declared items")
    if responses.size > _MAX_CELLS:
        raise ValueError("responses exceed the 20,000,000 logical-cell limit")
    if responses.dtype.kind not in "biuf" or responses.dtype.itemsize > 8:
        raise ValueError("responses must use Boolean, integer, or at-most-64-bit real storage")
    raw = responses.astype(np.float64, copy=False)
    if np.any(np.isinf(raw)):
        raise ValueError("infinite response values are not missing codes")
    encoded = np.full(raw.shape, np.nan, dtype=np.float64)
    source_columns = {name: j for j, name in enumerate(column_ids)}
    for j, item in enumerate(codebook.items):
        column = raw[:, source_columns[item.item_id]]
        admitted = np.isnan(column) | np.isin(column, codebook.missing_codes)
        for ordinal_code, raw_code in enumerate(item.ordered_codes):
            selected = column == raw_code
            encoded[selected, j] = ordinal_code
            admitted |= selected
        if not admitted.all():
            raise ValueError(f"item {item.item_id!r} contains undeclared response codes")
    immutable = np.frombuffer(encoded.tobytes(), dtype=np.float64).reshape(encoded.shape)
    return PreparedScale(
        responses=immutable, item_ids=item_ids, n_categories=n_categories,
        scale_id=codebook.scale_id, high_score_meaning=codebook.high_score_meaning,
        codebook_sha256=codebook_sha256(codebook),
    )
