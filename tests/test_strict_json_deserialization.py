"""Regression coverage for strict package-owned JSON trust boundaries."""

from __future__ import annotations

import pytest

from fast_mlsirm.llm_judge import JudgeFormatError, _response_object
from fast_mlsirm.rubric.generation import _contract_object


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_judge_response_rejects_nonfinite_json_constants(literal: str) -> None:
    with pytest.raises(JudgeFormatError, match="non-finite numeric value"):
        _response_object(f'{{"score": {literal}}}', required_fields={"score"})


def test_judge_response_rejects_duplicate_object_members() -> None:
    with pytest.raises(JudgeFormatError, match="duplicate JSON object keys"):
        _response_object('{"score": 0.25, "score": 0.75}', required_fields={"score"})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_generation_contract_rejects_nonfinite_json_constants(literal: str) -> None:
    with pytest.raises(ValueError, match="contract_json must be valid JSON text"):
        _contract_object(f'{{"contract_id": {literal}}}')


def test_generation_contract_rejects_duplicate_object_members() -> None:
    with pytest.raises(ValueError, match="contract_json must be valid JSON text"):
        _contract_object('{"contract_id": "first", "contract_id": "second"}')
