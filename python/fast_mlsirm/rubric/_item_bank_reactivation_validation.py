"""Fresh-evidence and creation-seal enforcement for governed item-bank records."""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any

from . import item_bank as _base

_ORIGINAL_CREATE_RECORD = _base._create_record
_ORIGINAL_VERIFY_CURRENT_RECORD = _base._verify_current_record
_ORIGINAL_TRANSITION_ITEM_BANK_RECORD = _base.transition_item_bank_record
_CREATION_SEALS: dict[
    int,
    tuple[weakref.ReferenceType[_base.ItemBankLifecycleRecord], str],
] = {}


def _forget_creation_seal(
    record_key: int,
    reference: weakref.ReferenceType[_base.ItemBankLifecycleRecord],
) -> None:
    """Discard one dead record seal without deleting a reused identity entry."""
    current = _CREATION_SEALS.get(record_key)
    if current is not None and current[0] is reference:
        _CREATION_SEALS.pop(record_key, None)


def _seal_record(record: _base.ItemBankLifecycleRecord) -> None:
    """Bind one factory-created object identity to its creation-time fingerprint."""
    record_key = id(record)
    fingerprint = vars(record)["_record_fingerprint"]
    reference = weakref.ref(
        record,
        lambda collected, key=record_key: _forget_creation_seal(key, collected),
    )
    _CREATION_SEALS[record_key] = (reference, fingerprint)


def _create_record(**kwargs: Any) -> _base.ItemBankLifecycleRecord:
    """Create one lifecycle record and retain its package-owned creation seal."""
    record = _ORIGINAL_CREATE_RECORD(**kwargs)
    _seal_record(record)
    return record


def _creation_fingerprint(record: _base.ItemBankLifecycleRecord) -> str | None:
    """Return the sealed fingerprint only for the exact still-live object identity."""
    sealed = _CREATION_SEALS.get(id(record))
    if sealed is None:
        return None
    reference, fingerprint = sealed
    if reference() is not record:
        return None
    return fingerprint


def _verify_current_record(record: Any) -> _base.ItemBankLifecycleRecord:
    """Replay mutable state and require its external factory-time identity seal."""
    verified = _ORIGINAL_VERIFY_CURRENT_RECORD(record)
    sealed_fingerprint = _creation_fingerprint(verified)
    if (
        sealed_fingerprint is None
        or vars(verified)["_record_fingerprint"] != sealed_fingerprint
    ):
        _base._raise_lifecycle_replay_mismatch()
    return verified


def _evidence_reference_state(
    reference: Any,
) -> dict[str, Any]:
    """Replay one standalone evidence identity without caller protocol dispatch."""
    if type(reference) is not _base.ItemBankEvidenceReference:
        raise ValueError("evidence reference must be an exact ItemBankEvidenceReference")
    state = vars(reference)
    field_names = tuple(state)
    if (
        any(type(name) is not str for name in field_names)
        or frozenset(field_names) != _base._EVIDENCE_REFERENCE_INSTANCE_FIELDS
        or type(state["evidence_kind"]) is not _base.ItemBankEvidenceKind
        or type(state["evidence_id"]) is not str
        or type(state["evidence_fingerprint"]) is not str
    ):
        raise ValueError("evidence reference no longer matches its normalized identity")
    _base._identifier(state["evidence_id"], "evidence_id")
    _base._fingerprint(state["evidence_fingerprint"], "evidence_fingerprint")
    return state


def _evidence_reference_to_dict(
    reference: _base.ItemBankEvidenceReference,
) -> dict[str, str]:
    """Serialize one evidence identity only after package-owned invariant replay."""
    state = _evidence_reference_state(reference)
    return {
        "evidence_kind": state["evidence_kind"].value,
        "evidence_id": state["evidence_id"],
        "evidence_fingerprint": state["evidence_fingerprint"],
    }


def _transition_evidence_references(
    values: Iterable[_base.ItemBankEvidenceReference],
) -> tuple[_base.ItemBankEvidenceReference, ...]:
    """Bound and replay newly supplied transition evidence before normalization."""
    try:
        raw = _base._bounded_values(
            values,
            "evidence_references",
            minimum=0,
            maximum=_base._MAX_EVIDENCE_REFERENCES,
        )
    except ValueError:
        raise _base.ItemBankLifecycleError(
            "invalid_evidence_references",
            "$.evidence_references",
            "evidence references must be a bounded collection",
        ) from None

    for index, reference in enumerate(raw):
        try:
            _evidence_reference_state(reference)
        except ValueError:
            raise _base.ItemBankLifecycleError(
                "invalid_evidence_reference",
                f"$.evidence_references[{index}]",
                "evidence reference no longer matches its normalized identity",
            ) from None

    return _base._normalize_evidence_references(
        raw,
        error_type=_base.ItemBankLifecycleError,
    )


def transition_item_bank_record(
    current_record: _base.ItemBankLifecycleRecord,
    target_state: _base.ItemBankLifecycleState,
    *,
    evidence_references: Iterable[_base.ItemBankEvidenceReference],
    transition_reason_id: str,
    approved_use_ids: Iterable[str] | None = None,
) -> _base.ItemBankLifecycleRecord:
    """Replay transition evidence and reject historical reactivation fingerprints."""
    additions = _transition_evidence_references(evidence_references)
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
    """Install lifecycle identity and fresh-evidence enforcement."""
    module._create_record = _create_record
    module._verify_current_record = _verify_current_record
    module.ItemBankEvidenceReference.to_dict = _evidence_reference_to_dict
    module.transition_item_bank_record = transition_item_bank_record


__all__: list[str] = []
