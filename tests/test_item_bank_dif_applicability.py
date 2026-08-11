"""Fail-first contract for scientifically inapplicable DIF evidence."""

from fast_mlsirm.rubric.item_bank import ItemBankEvidenceKind


def test_item_bank_evidence_domain_can_represent_dif_not_applicable() -> None:
    """Calibration must not require fabricated DIF when no comparison design exists."""
    evidence_kinds = {kind.value for kind in ItemBankEvidenceKind}

    assert "dif_not_applicable" in evidence_kinds
