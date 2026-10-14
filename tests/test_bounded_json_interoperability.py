"""Interoperability regressions for repository bounded-JSON readers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
_bounded_json = importlib.import_module("scripts._bounded_json")
parse_json_bounded = _bounded_json.parse_json_bounded
read_json_object = _bounded_json.read_json_object


def _integer_token_beyond_runtime_limit() -> str:
    """Return one JSON integer token beyond CPython's active conversion ceiling."""
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("CPython integer-string conversion limit is disabled")
    return "9" * (limit + 1)


@pytest.mark.parametrize(
    "payload",
    (
        '{"status":"ok","status":"failed"}',
        '{"outer":{"id":1,"id":2}}',
    ),
    ids=("root", "nested"),
)
def test_parse_json_bounded_rejects_duplicate_object_members(payload: str) -> None:
    """Direct bounded parsing must reject ambiguous duplicate member names."""
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        parse_json_bounded(payload)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_parse_json_bounded_rejects_nonfinite_numeric_constants(constant: str) -> None:
    """Direct bounded parsing must reject non-standard JSON numeric constants."""
    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        parse_json_bounded(f'{{"value":{constant}}}')


def test_parse_json_bounded_normalizes_integer_conversion_capacity() -> None:
    """Direct parsing must not leak interpreter-specific oversized-int errors."""
    token = _integer_token_beyond_runtime_limit()
    with pytest.raises(ValueError, match="integer conversion capacity"):
        parse_json_bounded(f'{{"value":{token}}}')

    assert parse_json_bounded('{"positive":123,"negative":-456}') == {
        "positive": 123,
        "negative": -456,
    }


def test_read_json_object_normalizes_integer_conversion_capacity(tmp_path: Path) -> None:
    """File-backed parsing must share the stable oversized-integer diagnostic."""
    token = _integer_token_beyond_runtime_limit()
    source = tmp_path / "oversized-integer.json"
    source.write_text(f'{{"value":{token}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="integer conversion capacity"):
        read_json_object(source)


def test_read_json_object_uses_same_strict_value_semantics(tmp_path: Path) -> None:
    """File-backed automation parsing must share duplicate/non-finite rejection."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"id":1,"id":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        read_json_object(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        read_json_object(nonfinite)
