"""Tests ensuring JSON numeric overflow (e.g., 1e999 to inf) is rejected at all boundaries."""

from pathlib import Path

import pytest

from fast_mlsirm.cross_engine_conformance import ConformanceInventory
from fast_mlsirm.io import _load_json_bounded
from fast_mlsirm.llm_judge import JudgeFormatError, _response_object
from fast_mlsirm.rubric import (
    CandidateValidationError,
    parse_generated_item_candidate,
)
from tests.test_rubric_generation import _raw, _request


def test_cross_engine_conformance_rejects_numeric_overflow() -> None:
    """Ensure exact-head cross-engine JSON bounds reject numeric overflow."""
    payload = (
        '{"schema_version": "1.0", "package_version": "1.0.0", "source_commit": '
        '"0000000000000000000000000000000000000000", "capabilities": [], '
        '"inventory_fingerprint": '
        '"0000000000000000000000000000000000000000000000000000000000000000", '
        '"run_provenance": null, '
        '"theta": 1e999}'
    )
    with pytest.raises(ValueError, match="manifest JSON contains non-finite numbers"):
        ConformanceInventory.from_json(payload)

    # A valid large float reaches schema validation instead of the overflow guard.
    payload_valid = (
        '{"schema_version": "1.0", "package_version": "1.0.0", "source_commit": '
        '"0000000000000000000000000000000000000000", "capabilities": [], '
        '"inventory_fingerprint": '
        '"0000000000000000000000000000000000000000000000000000000000000000", '
        '"run_provenance": null, '
        '"theta": 1e300}'
    )
    with pytest.raises(ValueError, match="manifest keys must be exactly"):
        ConformanceInventory.from_json(payload_valid)


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
    request = _request()
    overflowing = _raw(request).replace('"score": 2', '"score": 1e999', 1)
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(overflowing, request)
    assert error.value.code == "nonfinite_json_number"
    assert "1e999" not in str(error.value)

    finite = _raw(request).replace('"score": 2', '"score": 1e300', 1)
    with pytest.raises(CandidateValidationError) as finite_error:
        parse_generated_item_candidate(finite, request)
    assert finite_error.value.code == "invalid_type"
