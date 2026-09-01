"""Regression tests for the price-neutral acquisition release orchestrator."""

from pathlib import Path

from scripts import build_acquisition_release


ROOT = Path(__file__).resolve().parents[1]


def test_builder_defaults_to_no_transaction_value() -> None:
    """Generic acquisition evidence must not invent a deal value."""
    args = build_acquisition_release.build_parser().parse_args([])

    assert args.contract_value_krw is None


def test_final_sales_namespace_uses_generic_acquisition_profile() -> None:
    """The final gate must require generic evidence without legacy 20B mode."""
    args = build_acquisition_release._sales_args(
        repo_root=ROOT,
        acceptance=ROOT / "acceptance.json",
        dist=ROOT / "dist",
        out=ROOT / "sales.json",
        benchmark=ROOT / "benchmark.json",
        buyer_packet=ROOT / "buyer.json",
        release_index=ROOT / "release-index.json",
        procurement=ROOT / "procurement.json",
        pr_queue=ROOT / "pr-queue.json",
        figma=ROOT / "figma.json",
        require_rust=True,
        check_import=True,
        contract_value_krw=None,
        acquisition=True,
    )

    assert args.require_acquisition_readiness is True
    assert args.require_20b_product is False
    assert args.contract_value_krw is None
    assert args.require_buyer_packet is True
    assert args.require_release_evidence_index is True
    assert args.require_procurement_due_diligence is True
    assert args.require_pr_queue_governance is True
    assert args.require_figma_evidence_sync is True


def test_generic_builder_never_spells_legacy_gate_cli_flag() -> None:
    """The new orchestrator must not silently regress to the compatibility CLI."""
    source = (ROOT / "scripts" / "build_acquisition_release.py").read_text(
        encoding="utf-8"
    )

    assert "--require-20b-product" not in source
