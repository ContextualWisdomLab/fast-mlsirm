"""Fresh-evidence enforcement for governed item-bank reactivation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import item_bank as _base

_ORIGINAL_TRANSITION_ITEM_BANK_RECORD = _base.transition_item_bank_record


def transition_item_bank_record(
    current_record: _base.ItemBankLifecycleRecord,
    target_state: _base.ItemBankLifecycleState,
    *,
    evidence_references: Iterable[_base.ItemBankEvidenceReference],
    transition_reason_id: str,
    approved_use_ids: Iterable[str] | None = None,
) -> _base.ItemBankLifecycleRecord:
    """Reject historical evidence fingerprints on suspended-item reactivation."""
    additions = _base._normalize_evidence_references(
        evidence_references,
        error_type=_base.ItemBankLifecycleError,
    )
    successor = _ORIGINAL_TRANSITION_ITEM_BANK_RECORD(
        current_record,
        target_state,
        evidence_references=additions,
        transition_reason_id=transition_reason_id,
        approved_use_ids=approved_use_ids,
    )
    if (
        current_record.lifecycle_state is _base.ItemBankLifecycleState.SUSPENDED
        and successor.lifecycle_state is _base.ItemBankLifecycleState.ACTIVE
    ):
        required_fresh_kinds = {
            _base.ItemBankEvidenceKind.APPROVAL,
            *current_record.suspension_concern_kinds,
        }
        historical_fingerprints = {
            reference.evidence_fingerprint
            for reference in current_record.evidence_references
        }
        if any(
            reference.evidence_kind in required_fresh_kinds
            and reference.evidence_fingerprint in historical_fingerprints
            for reference in additions
        ):
            raise _base.ItemBankLifecycleError(
                "reused_transition_evidence",
                "$.evidence_references",
                (
                    "reactivation requires fresh evidence fingerprints for approval "
                    "and every suspension concern"
                ),
            )
    return successor


def install(module: Any) -> None:
    """Install fresh-evidence enforcement on the loaded item-bank module."""
    module.transition_item_bank_record = transition_item_bank_record


__all__: list[str] = []
