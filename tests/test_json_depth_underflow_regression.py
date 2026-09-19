"""Regression coverage for JSON depth-budget underflow hardening."""

from __future__ import annotations

import pytest

import fast_mlsirm.cross_engine_conformance as conformance
import fast_mlsirm.io as io_module
import fast_mlsirm.llm_judge as llm_judge
import fast_mlsirm.rubric.candidates as candidates
import fast_mlsirm.rubric.generation as generation


def _underflow_then_nest(depth: int) -> str:
    """Return malformed JSON that used unmatched closers to hide later nesting."""
    return "]" * (depth + 8) + "[" * (depth + 1) + "0" + "]" * (depth + 1)


def test_conformance_depth_budget_cannot_be_offset_by_unmatched_closers() -> None:
    """Manifest preflight must count later nesting from zero, never negative depth."""
    with pytest.raises(ValueError, match="manifest JSON nesting is too deep"):
        conformance._validate_raw_manifest_depth(
            _underflow_then_nest(conformance.MAX_MANIFEST_NESTING)
        )


def test_io_depth_budget_rejects_before_recursive_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File JSON preflight must reject the bypass before ``json.loads`` executes."""
    payload = _underflow_then_nest(io_module.MAX_JSON_NESTING_DEPTH)
    monkeypatch.setattr(io_module, "_read_text_bounded", lambda *args, **kwargs: payload)

    def fail_decoder(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads must not run for over-budget nesting")

    monkeypatch.setattr(io_module.json, "loads", fail_decoder)
    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        io_module._load_json_bounded("unused.json", source="test JSON")


def test_judge_depth_budget_cannot_be_offset_by_unmatched_closers() -> None:
    """Judge-response preflight must fail before malformed syntax can hide depth."""
    with pytest.raises(llm_judge.JudgeFormatError, match="nesting exceeds maximum depth"):
        llm_judge._validate_raw_json_depth(
            _underflow_then_nest(llm_judge.MAX_JUDGE_JSON_DEPTH)
        )


def test_candidate_depth_budget_cannot_be_offset_by_unmatched_closers() -> None:
    """Candidate JSON preflight must retain its package-owned depth diagnostic."""
    with pytest.raises(candidates.CandidateValidationError) as caught:
        candidates._validate_raw_json_depth(
            _underflow_then_nest(candidates.MAX_JSON_DEPTH)
        )
    assert caught.value.code == "json_too_deep"


def test_generation_depth_budget_cannot_be_offset_by_unmatched_closers() -> None:
    """Rubric-contract preflight must reject nesting beyond its fixed budget."""
    with pytest.raises(ValueError, match="maximum JSON nesting depth of 128"):
        generation._validate_contract_depth(_underflow_then_nest(128))
