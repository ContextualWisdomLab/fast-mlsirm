"""Regression tests for unmatched-close JSON depth preflight boundaries."""

from __future__ import annotations

import pytest

import fast_mlsirm.cross_engine_conformance as conformance
import fast_mlsirm.io as io_mod
import fast_mlsirm.llm_judge as judge
import fast_mlsirm.rubric.candidates as candidates
import fast_mlsirm.rubric.generation as generation


def _prefixed_over_depth(prefix: str, limit: int) -> str:
    """Return malformed-close input followed by nesting one above the limit."""
    return prefix + "[" * (limit + 1) + "0" + "]" * (limit + 1)


def _decoder_must_not_run(*_args: object, **_kwargs: object) -> object:
    """Fail if a raw depth preflight admits the hostile payload."""
    raise AssertionError("json.loads must not run for over-budget nesting")


@pytest.mark.parametrize("prefix", ["]", "}"])
def test_manifest_underflow_payload_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    """Manifest preflight ignores unmatched closes without masking later depth."""
    monkeypatch.setattr(conformance.json, "loads", _decoder_must_not_run)

    with pytest.raises(ValueError, match="manifest JSON nesting is too deep"):
        conformance.ConformanceInventory.from_json(
            _prefixed_over_depth(prefix, conformance.MAX_MANIFEST_NESTING)
        )


@pytest.mark.parametrize("prefix", ["]", "}"])
def test_file_json_underflow_payload_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    prefix: str,
) -> None:
    """Artifact preflight rejects masked nesting before stdlib decoding."""
    path = tmp_path / "underflow.json"
    path.write_text(
        _prefixed_over_depth(prefix, io_mod.MAX_JSON_NESTING_DEPTH),
        encoding="utf-8",
    )
    monkeypatch.setattr(io_mod.json, "loads", _decoder_must_not_run)

    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        io_mod._load_json_bounded(path, source="test JSON")


@pytest.mark.parametrize("prefix", ["]", "}"])
def test_candidate_underflow_payload_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    """Candidate preflight rejects masked nesting before provider decoding."""
    request = object.__new__(generation.GenerationRequest)
    monkeypatch.setattr(candidates.json, "loads", _decoder_must_not_run)

    with pytest.raises(candidates.CandidateValidationError) as error:
        candidates.parse_generated_item_candidate(
            _prefixed_over_depth(prefix, candidates.MAX_JSON_DEPTH),
            request,
        )

    assert error.value.code == "json_too_deep"


@pytest.mark.parametrize("prefix", ["]", "}"])
def test_contract_underflow_payload_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    """Generation-contract preflight rejects masked nesting before decoding."""
    monkeypatch.setattr(generation.json, "loads", _decoder_must_not_run)

    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        generation._contract_object(_prefixed_over_depth(prefix, 128))


@pytest.mark.parametrize("prefix", ["]", "}"])
def test_judge_underflow_payload_fails_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    """Judge preflight rejects masked nesting before response decoding."""
    monkeypatch.setattr(judge.json, "loads", _decoder_must_not_run)

    with pytest.raises(judge.JudgeFormatError, match="maximum depth"):
        judge._response_object(
            _prefixed_over_depth(prefix, judge.MAX_JUDGE_JSON_DEPTH),
            required_fields=set(),
        )
