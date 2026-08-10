"""Fail-first public contracts for acquisition/commercial readiness governance.

The readiness verifier may validate an explicitly supplied deal scenario, but it
must not fabricate a monetary target or use a legacy transaction-value label as
proof of product quality.  These tests pin the smallest public CLI boundary
before the implementation and downstream manifest migration are changed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sales_readiness():
    """Load the repository sales-readiness script as an importable module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_acquisition", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_readiness_does_not_fabricate_a_contract_value() -> None:
    """No deal value is present unless the caller explicitly supplies one."""
    parser = _load_sales_readiness().build_parser()

    args = parser.parse_args(["--acceptance", "acceptance_summary.json"])

    assert args.contract_value_krw is None


def test_generic_acquisition_readiness_gate_is_public() -> None:
    """Evidence completeness uses a truthful generic acquisition-readiness flag."""
    parser = _load_sales_readiness().build_parser()

    args = parser.parse_args(
        [
            "--acceptance",
            "acceptance_summary.json",
            "--require-acquisition-readiness",
        ]
    )

    assert args.require_acquisition_readiness is True


def test_cli_help_does_not_present_krw_2b_as_readiness_proof() -> None:
    """The public verifier description must not encode KRW 2B as a quality gate."""
    help_text = _load_sales_readiness().build_parser().format_help()

    assert "KRW 2B review" not in help_text
    assert "Target contract value for this gate" not in help_text
