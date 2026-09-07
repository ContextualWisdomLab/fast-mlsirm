"""Contracts for the governance index's requirements authority."""

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
