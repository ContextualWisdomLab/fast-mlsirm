"""Tests ensuring JSON numeric overflow (e.g., 1e999 to inf) is rejected at all boundaries."""

import json
from pathlib import Path

import pytest

from fast_mlsirm.io import _load_json_bounded
from fast_mlsirm.llm_judge import JudgeFormatError, _response_object

def test_cross_engine_conformance_rejects_numeric_overflow() -> None:
    """Ensure exact-head cross-engine JSON bounds reject numeric overflow."""

    # Check the parsing hook directly because valid payloads require deep JSON schemas
    from fast_mlsirm.cross_engine_conformance import (
        _reject_duplicate_json_keys,
        _reject_float_nonfinite,
        _reject_json_constant,
    )

    with pytest.raises(ValueError, match="manifest JSON contains non-finite numbers"):
        json.loads(
            '{"a": 1e999}',
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
            parse_float=_reject_float_nonfinite,
        )

    assert json.loads(
        '{"a": 1e300}',
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
        parse_float=_reject_float_nonfinite,
    ) == {"a": 1e300}


def test_io_rejects_numeric_overflow(tmp_path: Path) -> None:
    """Ensure standard bounded IO correctly rejects numeric overflow."""
    path = tmp_path / "test.json"
    path.write_text('{"key": 1e999}')
    with pytest.raises(
        ValueError, match="test_source contains a non-finite JSON numeric value"
    ):
        _load_json_bounded(path, source="test_source")

    path.write_text('{"key": 1e300}')
    assert _load_json_bounded(path, source="test_source")["key"] == 1e300


def test_llm_judge_rejects_numeric_overflow() -> None:
    """Ensure the orchestrator LLM judge parser rejects numeric overflow."""
    payload = '{"key": 1e999}'
    with pytest.raises(
        JudgeFormatError, match="judge response JSON contains non-finite numbers"
    ):
        _response_object(payload, required_fields={"key"})

    payload_valid = '{"key": 1e300}'
    assert _response_object(payload_valid, required_fields={"key"})["key"] == 1e300


def test_candidates_rejects_numeric_overflow() -> None:
    """Ensure generated item candidate parses reject numeric overflow."""

    from fast_mlsirm.rubric.candidates import (
        _NonFiniteJsonNumber,
        _reject_float_nonfinite,
        _reject_nonfinite,
        _unique_object,
    )

    with pytest.raises(_NonFiniteJsonNumber):
        json.loads(
            '{"a": 1e999}',
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float_nonfinite,
        )

    assert json.loads(
        '{"a": 1e300}',
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
        parse_float=_reject_float_nonfinite,
    ) == {"a": 1e300}
