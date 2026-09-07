"""Contracts for the governance index's current authority links and policies."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_INDEX = ROOT / "docs" / "GOVERNANCE_INDEX.md"


def test_governance_index_points_to_canonical_prd_and_trd() -> None:
    """Historical MVP notes must not be advertised as requirements authority."""
    product_row = next(
        line
        for line in GOVERNANCE_INDEX.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Product / technical requirements |")
    )

    assert "docs/PRD.md" in product_row
    assert "docs/TRD.md" in product_row
    assert "prd_trd_summary.md" not in product_row


def test_governance_index_does_not_authorize_direct_provider_credentials() -> None:
    """The current governance index must preserve contextual-orchestrator ownership."""
    text = GOVERNANCE_INDEX.read_text(encoding="utf-8")
    adr_row = next(line for line in text.splitlines() if line.startswith("| ADR-005 |"))

    assert "NVIDIA_NIM_API_KEY" not in adr_row
    assert "contextual-orchestrator" in adr_row
    assert "orchestrator/free" in adr_row
    assert "provider credentials" in adr_row.lower()
    assert "provider/model/group/paid fallback" in adr_row.lower()
