"""Regression tests for the governed item-bank lifecycle contract."""

from __future__ import annotations

import pytest

from fast_mlsirm.item_bank import (
    ItemBankEntry,
    ItemEvidenceKind,
    ItemLifecycleState,
    advance_item_bank_entry,
)


def _sha(char: str) -> str:
    return char * 64


def _entry(*, comparable: bool = False) -> ItemBankEntry:
    return ItemBankEntry(
        item_id="item_bank_entry_alpha",
        item_version="item_version_001",
        rubric_fingerprint=_sha("a"),
        blueprint_fingerprint=_sha("b"),
        generation_contract_fingerprint=_sha("c"),
        state=ItemLifecycleState.DRAFT,
        claims_cross_version_comparability=comparable,
    )


def test_operational_item_cannot_skip_governed_lifecycle() -> None:
    """Generated JSON alone must never authorize operational activation."""

    with pytest.raises(ValueError, match="transition draft -> active is not allowed"):
        advance_item_bank_entry(
            _entry(),
            ItemLifecycleState.ACTIVE,
            evidence_fingerprints={},
        )


def test_each_forward_transition_requires_state_specific_evidence() -> None:
    """Every forward lifecycle step must fail closed without its evidence."""

    entry = _entry()
    cases = (
        (ItemLifecycleState.AUDITED, ItemEvidenceKind.AUDIT),
        (ItemLifecycleState.SCREENED, ItemEvidenceKind.SCREENING),
        (ItemLifecycleState.PILOTING, ItemEvidenceKind.PILOT_DESIGN),
        (ItemLifecycleState.CALIBRATED, ItemEvidenceKind.CALIBRATION),
        (ItemLifecycleState.APPROVED, ItemEvidenceKind.APPROVAL),
        (ItemLifecycleState.ACTIVE, ItemEvidenceKind.RELEASE),
    )

    for target, required in cases:
        with pytest.raises(ValueError, match=f"missing required evidence: {required.value}"):
            advance_item_bank_entry(entry, target, evidence_fingerprints={})
        entry = advance_item_bank_entry(
            entry,
            target,
            evidence_fingerprints={required: _sha(str(len(entry.history) + 1))},
        )

    assert entry.state is ItemLifecycleState.ACTIVE
    assert tuple(decision.to_state for decision in entry.history) == tuple(
        target for target, _required in cases
    )


def test_cross_version_comparability_requires_linking_evidence_before_activation() -> None:
    """Nominally equal score ranges must not imply cross-version comparability."""

    entry = _entry(comparable=True)
    for target, evidence in (
        (ItemLifecycleState.AUDITED, {ItemEvidenceKind.AUDIT: _sha("1")}),
        (ItemLifecycleState.SCREENED, {ItemEvidenceKind.SCREENING: _sha("2")}),
        (ItemLifecycleState.PILOTING, {ItemEvidenceKind.PILOT_DESIGN: _sha("3")}),
        (ItemLifecycleState.CALIBRATED, {ItemEvidenceKind.CALIBRATION: _sha("4")}),
        (ItemLifecycleState.APPROVED, {ItemEvidenceKind.APPROVAL: _sha("5")}),
    ):
        entry = advance_item_bank_entry(entry, target, evidence_fingerprints=evidence)

    with pytest.raises(ValueError, match="missing required evidence: linking"):
        advance_item_bank_entry(
            entry,
            ItemLifecycleState.ACTIVE,
            evidence_fingerprints={ItemEvidenceKind.RELEASE: _sha("6")},
        )

    active = advance_item_bank_entry(
        entry,
        ItemLifecycleState.ACTIVE,
        evidence_fingerprints={
            ItemEvidenceKind.RELEASE: _sha("6"),
            ItemEvidenceKind.LINKING: _sha("7"),
        },
    )
    assert active.state is ItemLifecycleState.ACTIVE


def test_suspend_resume_and_retire_preserve_immutable_history() -> None:
    """Suspension and retirement keep prior lifecycle decisions auditable."""

    entry = _entry()
    for target, evidence in (
        (ItemLifecycleState.AUDITED, {ItemEvidenceKind.AUDIT: _sha("1")}),
        (ItemLifecycleState.SCREENED, {ItemEvidenceKind.SCREENING: _sha("2")}),
        (ItemLifecycleState.PILOTING, {ItemEvidenceKind.PILOT_DESIGN: _sha("3")}),
        (ItemLifecycleState.CALIBRATED, {ItemEvidenceKind.CALIBRATION: _sha("4")}),
        (ItemLifecycleState.APPROVED, {ItemEvidenceKind.APPROVAL: _sha("5")}),
        (ItemLifecycleState.ACTIVE, {ItemEvidenceKind.RELEASE: _sha("6")}),
    ):
        entry = advance_item_bank_entry(entry, target, evidence_fingerprints=evidence)

    before_suspend = entry.history
    suspended = advance_item_bank_entry(
        entry,
        ItemLifecycleState.SUSPENDED,
        evidence_fingerprints={ItemEvidenceKind.SUSPENSION: _sha("7")},
    )
    assert entry.history is before_suspend
    assert entry.state is ItemLifecycleState.ACTIVE
    assert suspended.state is ItemLifecycleState.SUSPENDED

    resumed = advance_item_bank_entry(
        suspended,
        ItemLifecycleState.ACTIVE,
        evidence_fingerprints={
            ItemEvidenceKind.APPROVAL: _sha("8"),
            ItemEvidenceKind.RELEASE: _sha("9"),
        },
    )
    retired = advance_item_bank_entry(
        resumed,
        ItemLifecycleState.RETIRED,
        evidence_fingerprints={ItemEvidenceKind.RETIREMENT: _sha("d")},
    )
    assert retired.state is ItemLifecycleState.RETIRED
    with pytest.raises(ValueError, match="retired is terminal"):
        advance_item_bank_entry(
            retired,
            ItemLifecycleState.ACTIVE,
            evidence_fingerprints={ItemEvidenceKind.RELEASE: _sha("e")},
        )


def test_contract_rejects_ambiguous_or_mutable_identity_inputs() -> None:
    """Opaque identities and fingerprints are exact bounded built-in strings."""

    class HostileString(str):
        def strip(self) -> str:
            raise AssertionError("caller callback executed")

    with pytest.raises(TypeError, match="item_id must be a built-in str"):
        ItemBankEntry(
            item_id=HostileString("item_bank_entry_alpha"),
            item_version="item_version_001",
            rubric_fingerprint=_sha("a"),
            blueprint_fingerprint=_sha("b"),
            generation_contract_fingerprint=_sha("c"),
            state=ItemLifecycleState.DRAFT,
        )

    with pytest.raises(ValueError, match="rubric_fingerprint must be 64 lowercase hexadecimal characters"):
        ItemBankEntry(
            item_id="item_bank_entry_alpha",
            item_version="item_version_001",
            rubric_fingerprint="not-a-full-digest",
            blueprint_fingerprint=_sha("b"),
            generation_contract_fingerprint=_sha("c"),
            state=ItemLifecycleState.DRAFT,
        )


def test_evidence_fingerprints_are_normalized_into_immutable_decision_tuple() -> None:
    """Caller mappings are copied into deterministic immutable provenance."""

    evidence = {ItemEvidenceKind.AUDIT: _sha("a")}
    advanced = advance_item_bank_entry(
        _entry(),
        ItemLifecycleState.AUDITED,
        evidence_fingerprints=evidence,
    )
    evidence[ItemEvidenceKind.AUDIT] = _sha("b")

    assert advanced.history[-1].evidence_fingerprints == (
        (ItemEvidenceKind.AUDIT, _sha("a")),
    )
