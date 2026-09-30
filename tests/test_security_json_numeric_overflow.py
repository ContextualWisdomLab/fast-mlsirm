import pytest
from pathlib import Path
import json

from fast_mlsirm.cross_engine_conformance import ConformanceInventory
from fast_mlsirm.io import _load_json_bounded
from fast_mlsirm.llm_judge import _response_object, JudgeFormatError

def test_cross_engine_conformance_rejects_numeric_overflow():
    payload = '{"schema_version": "1.0", "engine_id": "ref", "engine_version": "0.1.0", "target_os": "linux", "target_arch": "x86_64", "package_version": "0.1.0", "redistribution_status": "ok", "items": [{"item_id": "a", "theta": 1e999}]}'
    with pytest.raises(ValueError, match="manifest JSON contains non-finite numbers"):
        ConformanceInventory.from_json(payload)

def test_io_rejects_numeric_overflow(tmp_path: Path):
    path = tmp_path / "test.json"
    path.write_text('{"key": 1e999}')
    with pytest.raises(ValueError, match="test_source contains a non-finite JSON numeric value"):
        _load_json_bounded(path, source="test_source")

def test_llm_judge_rejects_numeric_overflow():
    payload = '{"key": 1e999}'
    with pytest.raises(JudgeFormatError, match="judge response JSON contains non-finite numbers"):
        _response_object(payload, required_fields={"key"})

from fast_mlsirm.rubric.candidates import parse_generated_item_candidate, CandidateValidationError
from fast_mlsirm.rubric.generation import GenerationRequest

def test_candidates_rejects_numeric_overflow():
    import json
    from fast_mlsirm.rubric.candidates import _reject_float_nonfinite, _NonFiniteJsonNumber
    with pytest.raises(_NonFiniteJsonNumber):
        _reject_float_nonfinite("1e999")
