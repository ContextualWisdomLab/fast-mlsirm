import pytest

from fast_mlsirm.rubric.candidates import _validate_raw_json_depth
from fast_mlsirm.rubric.generation import _validate_contract_depth
from fast_mlsirm.cross_engine_conformance import _validate_raw_manifest_depth
from fast_mlsirm.llm_judge import _validate_raw_json_depth as judge_validate_raw_json_depth, JudgeFormatError

# Create a payload that prefix-underflows and then exceeds the max depth.
# Max depths:
# candidates: 128
# generation: 128
# cross_engine_conformance: 128
# llm_judge: 64
# io: 64

def test_candidates_json_depth_underflow():
    # Attempt to underflow depth by 200, then nest 150 deep (exceeds 128)
    payload = ']' * 200 + '{"a":' * 150 + '1' + '}' * 150
    with pytest.raises(Exception, match="JSON nesting exceeds the maximum depth"):
        _validate_raw_json_depth(payload)

def test_generation_json_depth_underflow():
    payload = ']' * 200 + '{"a":' * 150 + '1' + '}' * 150
    with pytest.raises(ValueError, match="contract_json exceeds the maximum JSON nesting depth"):
        _validate_contract_depth(payload)

def test_cross_engine_conformance_json_depth_underflow():
    payload = ']' * 200 + '{"a":' * 150 + '1' + '}' * 150
    with pytest.raises(ValueError, match="manifest JSON nesting is too deep"):
        _validate_raw_manifest_depth(payload)

def test_llm_judge_json_depth_underflow():
    # judge max depth is 64
    payload = ']' * 200 + '{"a":' * 80 + '1' + '}' * 80
    with pytest.raises(JudgeFormatError, match="judge response JSON nesting exceeds maximum depth"):
        judge_validate_raw_json_depth(payload)
