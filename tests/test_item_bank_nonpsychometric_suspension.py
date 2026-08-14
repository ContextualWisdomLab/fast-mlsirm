"""Regression coverage for non-psychometric item-bank suspension concerns."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleState,
    transition_item_bank_record,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _evidence(kind: ItemBankEvidenceKind | str, suffix: str, character: str):
    """Return one bounded exact evidence identity."""
    return ItemBankEvidenceReference(
        evidence_kind=kind,
        evidence_id=f"{suffix}_evidence",
        evidence_fingerprint=character * 64,
    )


def _active_record():
    """Return one governed active item from the canonical lifecycle fixture."""
    calibrated = _FIXTURES["_calibrated_record"]()
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "approval", "e"),
        ),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )
    return transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )


def test_security_privacy_concern_can_suspend_and_reactivate_without_fake_drift() -> None:
    """Security/privacy quarantine uses exact concern evidence, not invented DIF/drift."""
    active = _active_record()
    suspended = transition_item_bank_record(
        active,
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "security_quarantine", "f"),
            _evidence("security_privacy", "security_finding", "1"),
        ),
        transition_reason_id="security_privacy_quarantine",
    )

    assert suspended.lifecycle_state is ItemBankLifecycleState.SUSPENDED
    assert {reference.evidence_kind.value for reference in suspended.evidence_references} >= {
        "suspension",
        "security_privacy",
    }

    reactivated = transition_item_bank_record(
        suspended,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "security_reactivation", "2"),
            _evidence("security_privacy", "security_recheck", "3"),
        ),
        transition_reason_id="security_privacy_cleared",
    )

    assert reactivated.lifecycle_state is ItemBankLifecycleState.ACTIVE
    assert reactivated.previous_record_fingerprint == suspended.record_fingerprint
