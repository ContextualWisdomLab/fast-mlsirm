"""Receipt contract for the S1 two-tier GRM FIPC consumer verification script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    script = ROOT / "scripts" / "s1_fipc_consumer_verify.py"
    spec = importlib.util.spec_from_file_location("s1_fipc_consumer_verify", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_receipt_matches_schema_and_reports_missing_binding_honestly(monkeypatch) -> None:
    """Without the FIPC binding the receipt says so instead of hiding a TypeError."""
    module = _load_script()
    monkeypatch.delattr(module._core, "fit_two_tier_grm_fipc", raising=False)
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

    schema = json.loads(
        (ROOT / "contracts" / "s1-fipc-consumer-verify-v1.schema.json").read_text()
    )
    properties = schema["properties"]
    assert set(schema["required"]) <= receipt.keys()
    assert receipt.keys() <= properties.keys()
    assert receipt["row_order_diagnosis"] in properties["row_order_diagnosis"]["enum"]

    assert receipt["row_order_diagnosis"] == "binding_unavailable"
    assert receipt["row_order_max_abs"] is None
    assert receipt["build_source_sha_present"] is False
    assert receipt["all_pass"] is False
    assert "fipc_fit_error" not in receipt
