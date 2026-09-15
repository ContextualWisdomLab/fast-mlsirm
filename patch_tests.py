import re

# Update tests/test_io_nonfinite_json.py
io_test_path = "tests/test_io_nonfinite_json.py"
with open(io_test_path, "r") as f:
    io_test = f.read()
if "test_load_json_bounded_rejects_numeric_overflow" not in io_test:
    io_test += """
def test_load_json_bounded_rejects_numeric_overflow(tmp_path):
    artifact = tmp_path / "overflow.json"
    artifact.write_text('{"value": 1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="contains a non-finite JSON numeric value"):
        _load_json_bounded(artifact, source="artifact JSON")
"""
    with open(io_test_path, "w") as f:
        f.write(io_test)


# Update tests/test_llm_judge.py
llm_judge_test_path = "tests/test_llm_judge.py"
with open(llm_judge_test_path, "r") as f:
    llm_judge_test = f.read()
if "test_judge_rejects_numeric_overflow" not in llm_judge_test:
    llm_judge_test += """
def test_judge_rejects_numeric_overflow():
    from fast_mlsirm.llm_judge import _response_object, JudgeFormatError
    with pytest.raises(JudgeFormatError, match="judge response JSON is invalid"):
        _response_object('{"rationale": "test", "meets_threshold": true, "score": 1e999}', required_fields={"rationale", "meets_threshold", "score"})
"""
    with open(llm_judge_test_path, "w") as f:
        f.write(llm_judge_test)


# Update tests/test_cross_engine_conformance.py
cec_test_path = "tests/test_cross_engine_conformance.py"
with open(cec_test_path, "r") as f:
    cec_test = f.read()
if "test_inventory_rejects_numeric_overflow" not in cec_test:
    cec_test += """
def test_inventory_rejects_numeric_overflow():
    payload = '{"inventory_fingerprint": "a", "package_version": "v1", "source_commit": "abc", "capabilities": [], "schema_version": "v1", "run_provenance": {"val": 1e999}}'
    with pytest.raises(ValueError, match="manifest JSON contains a non-finite float"):
        ConformanceInventory.from_json(payload)
"""
    with open(cec_test_path, "w") as f:
        f.write(cec_test)


# Update tests/test_rubric_generation_edge_cases.py
rubric_gen_edge_test_path = "tests/test_rubric_generation_edge_cases.py"
with open(rubric_gen_edge_test_path, "r") as f:
    rubric_gen_edge_test = f.read()
if "test_parser_rejects_numeric_overflow" not in rubric_gen_edge_test:
    rubric_gen_edge_test += """
def test_parser_rejects_numeric_overflow():
    from fast_mlsirm.rubric.candidates import parse_generated_item_candidate, CandidateValidationError
    from unittest.mock import MagicMock
    from fast_mlsirm.rubric.generation import GenerationRequest
    req = MagicMock(spec=GenerationRequest)
    req.contract = {"schema_version": "v1", "generation_provenance": "abc", "criteria": [], "metadata": {}}
    raw = '{"item_text": "a", "rationales": {}, "metadata": {"val": 1e999}, "generation_provenance": "abc", "answer_key": "c", "criteria_coverage": {}, "blueprint_fingerprint": "abc", "request_fingerprint": "xyz", "raw_response_sha256": "zzz"}'
    with pytest.raises(CandidateValidationError) as exc:
        parse_generated_item_candidate(raw, req)
    assert exc.value.code == "nonfinite_json_number"
"""
    with open(rubric_gen_edge_test_path, "w") as f:
        f.write(rubric_gen_edge_test)


# Update tests/test_rubric_generation.py
rubric_gen_test_path = "tests/test_rubric_generation.py"
with open(rubric_gen_test_path, "r") as f:
    rubric_gen_test = f.read()
if "test_contract_rejects_numeric_overflow" not in rubric_gen_test:
    rubric_gen_test += """
def test_contract_rejects_numeric_overflow():
    from fast_mlsirm.rubric.generation import _contract_object
    with pytest.raises(ValueError, match="contract_json contains non-finite numbers"):
        _contract_object('{"schema_version": "v1", "val": 1e999}')
"""
    with open(rubric_gen_test_path, "w") as f:
        f.write(rubric_gen_test)
