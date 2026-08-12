"""Immutable post-pilot lifecycle contracts for governed assessment items.

This module records evidence-bound lifecycle transitions only. It performs no
item calibration, fit, DIF, information, linking, exposure, drift, scoring, or
approval arithmetic. Numerical evidence remains owned by Rust-backed modules
and is referenced here by exact content fingerprint.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import re
from typing import Any, Iterable

from .audit import CandidateLifecycleState
from .models import (
    MAX_COLLECTION_VALUES,
    SCHEMA_VERSION,
    _bounded_values,
    _identifier,
    _identifier_tuple,
    _schema_version,
    _semantic_version,
    _sha256_hex,
    _text,
)
from .verified_pilot import PilotCandidateRecord

_MAX_EVIDENCE_REFERENCES = 64
_MAX_ERROR_MESSAGE_CHARACTERS = 512
_LIFECYCLE_PATH_PATTERN = re.compile(r"^\$(?:\.[a-z][a-z0-9_]*|\[[0-9]+\])*$")
_RECORD_CREATION_TOKEN = object()


class ItemBankLifecycleState(str, Enum):
    """Allowed post-pilot states for one versioned item-bank entry."""

    PILOTING = "piloting"
    CALIBRATED = "calibrated"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class ItemBankEvidenceKind(str, Enum):
    """Governed evidence classes that may support lifecycle transitions."""

    CALIBRATION = "calibration"
    ITEM_FIT = "item_fit"
    DIF = "dif"
    DIF_NOT_APPLICABLE = "dif_not_applicable"
    ITEM_INFORMATION = "item_information"
    LINKING = "linking"
    EXPOSURE = "exposure"
    DRIFT = "drift"
    APPROVAL = "approval"
    SUSPENSION = "suspension"
    RETIREMENT = "retirement"


class PolicyCriticality(str, Enum):
    """Operational criticality kept separate from psychometric information."""

    ORDINARY = "ordinary"
    REQUIRED = "required"
    CONJUNCTIVE_GATE = "conjunctive_gate"


class ItemBankLifecycleError(ValueError):
    """Stable source-text-free lifecycle validation failure."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store bounded machine-readable failure metadata."""
        self.code = _identifier(code, "code")
        self.path = _lifecycle_path(path)
        self.message = _text(
            message,
            "message",
            maximum=_MAX_ERROR_MESSAGE_CHARACTERS,
        )
        super().__init__(f"{self.code} at {self.path}: {self.message}")


def _lifecycle_path(value: Any) -> str:
    """Normalize a redacted JSON-style lifecycle field path."""
    normalized = _text(value, "path", maximum=512)
    if _LIFECYCLE_PATH_PATTERN.fullmatch(normalized) is None:
        raise ValueError("path must be a redacted JSON-style field path")
    return normalized


def _fingerprint(value: Any, name: str) -> str:
    """Normalize one complete lower-hexadecimal SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be 64 lower hexadecimal characters")
    return normalized


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    """Normalize an exact enum member or its serialized value."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


@dataclass(frozen=True)
class ItemBankEvidenceReference:
    """Source-text-free identity of one lifecycle evidence artifact."""

    evidence_kind: ItemBankEvidenceKind
    evidence_id: str
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        """Normalize exact evidence kind, identifier, and content identity."""
        object.__setattr__(
            self,
            "evidence_kind",
            _enum_value(self.evidence_kind, ItemBankEvidenceKind, "evidence_kind"),
        )
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _fingerprint(self.evidence_fingerprint, "evidence_fingerprint"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return one JSON-compatible evidence identity without raw content."""
        return {
            "evidence_kind": self.evidence_kind.value,
            "evidence_id": self.evidence_id,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


def _normalize_evidence_references(
    values: Iterable[ItemBankEvidenceReference],
    *,
    error_type: type[Exception] = ValueError,
) -> tuple[ItemBankEvidenceReference, ...]:
    """Normalize bounded exact evidence records and reject identity conflicts."""
    try:
        raw = _bounded_values(
            values,
            "evidence_references",
            minimum=0,
            maximum=_MAX_EVIDENCE_REFERENCES,
        )
    except ValueError as exc:
        if error_type is ItemBankLifecycleError:
            raise ItemBankLifecycleError(
                "invalid_evidence_references",
                "$.evidence_references",
                "evidence references must be a bounded collection",
            ) from None
        raise

    normalized: list[ItemBankEvidenceReference] = []
    fingerprints_by_id: dict[str, str] = {}
    exact_identities: set[tuple[ItemBankEvidenceKind, str, str]] = set()
    for index, reference in enumerate(raw):
        if type(reference) is not ItemBankEvidenceReference:
            if error_type is ItemBankLifecycleError:
                raise ItemBankLifecycleError(
                    "invalid_evidence_reference",
                    f"$.evidence_references[{index}]",
                    "evidence references must be exact ItemBankEvidenceReference values",
                )
            raise ValueError(
                f"evidence_references[{index}] must be an ItemBankEvidenceReference"
            )
        identity = (
            reference.evidence_kind,
            reference.evidence_id,
            reference.evidence_fingerprint,
        )
        if identity in exact_identities:
            if error_type is ItemBankLifecycleError:
                raise ItemBankLifecycleError(
                    "duplicate_evidence_reference",
                    "$.evidence_references",
                    "evidence references must not repeat an exact identity",
                )
            raise ValueError("evidence_references must not contain duplicates")
        exact_identities.add(identity)
        previous_fingerprint = fingerprints_by_id.get(reference.evidence_id)
        if (
            previous_fingerprint is not None
            and previous_fingerprint != reference.evidence_fingerprint
        ):
            if error_type is ItemBankLifecycleError:
                raise ItemBankLifecycleError(
                    "conflicting_evidence_identity",
                    "$.evidence_references",
                    "one evidence identifier cannot bind multiple fingerprints",
                )
            raise ValueError(
                "one evidence identifier cannot bind multiple fingerprints"
            )
        fingerprints_by_id[reference.evidence_id] = reference.evidence_fingerprint
        normalized.append(reference)
    return tuple(
        sorted(
            normalized,
            key=lambda reference: (
                reference.evidence_kind.value,
                reference.evidence_id,
                reference.evidence_fingerprint,
            ),
        )
    )


@dataclass(frozen=True)
class ItemBankLifecycleRecord:
    """Factory-sealed immutable lifecycle state for one exact item version."""

    item_id: str
    item_version: str
    candidate_fingerprint: str
    pilot_record_fingerprint: str
    audit_report_fingerprint: str
    blueprint_id: str
    rubric_id: str
    rubric_version: str
    lifecycle_state: ItemBankLifecycleState
    policy_criticality: PolicyCriticality
    approved_use_ids: tuple[str, ...]
    evidence_references: tuple[ItemBankEvidenceReference, ...]
    previous_record_fingerprint: str | None
    transition_reason_id: str
    schema_version: str = SCHEMA_VERSION
    _creation_token: InitVar[object | None] = None
    _record_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self, _creation_token: object | None) -> None:
        """Reject direct construction and seal normalized lifecycle content."""
        if _creation_token is not _RECORD_CREATION_TOKEN:
            raise ValueError(
                "ItemBankLifecycleRecord must be created by "
                "build_item_bank_pilot_record or transition_item_bank_record"
            )
        for name in ("item_id", "blueprint_id", "rubric_id", "transition_reason_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "item_version",
            _semantic_version(self.item_version, "item_version"),
        )
        for name in (
            "candidate_fingerprint",
            "pilot_record_fingerprint",
            "audit_report_fingerprint",
        ):
            object.__setattr__(self, name, _fingerprint(getattr(self, name), name))
        object.__setattr__(
            self,
            "rubric_version",
            _semantic_version(self.rubric_version, "rubric_version"),
        )
        object.__setattr__(
            self,
            "lifecycle_state",
            _enum_value(
                self.lifecycle_state,
                ItemBankLifecycleState,
                "lifecycle_state",
            ),
        )
        object.__setattr__(
            self,
            "policy_criticality",
            _enum_value(
                self.policy_criticality,
                PolicyCriticality,
                "policy_criticality",
            ),
        )
        object.__setattr__(
            self,
            "approved_use_ids",
            tuple(
                sorted(
                    _identifier_tuple(
                        self.approved_use_ids,
                        "approved_use_ids",
                        minimum=0,
                        maximum=MAX_COLLECTION_VALUES,
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _normalize_evidence_references(self.evidence_references),
        )
        if self.previous_record_fingerprint is not None:
            object.__setattr__(
                self,
                "previous_record_fingerprint",
                _fingerprint(
                    self.previous_record_fingerprint,
                    "previous_record_fingerprint",
                ),
            )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if self.lifecycle_state is ItemBankLifecycleState.PILOTING:
            if self.previous_record_fingerprint is not None:
                raise ValueError("piloting records must not have a previous record")
            if self.evidence_references:
                raise ValueError("piloting records must not add lifecycle evidence")
            if self.approved_use_ids:
                raise ValueError("piloting records must not declare approved uses")
            if self.transition_reason_id != "pilot_admission":
                raise ValueError("piloting records must use pilot_admission")
        else:
            if self.previous_record_fingerprint is None:
                raise ValueError("post-pilot records require a previous record fingerprint")
        if self.lifecycle_state in {
            ItemBankLifecycleState.APPROVED,
            ItemBankLifecycleState.ACTIVE,
            ItemBankLifecycleState.SUSPENDED,
            ItemBankLifecycleState.RETIRED,
        } and not self.approved_use_ids:
            raise ValueError("approved and later records require approved use identifiers")

        object.__setattr__(self, "_record_fingerprint", _sha256_hex(self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical lifecycle content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "item_version": self.item_version,
            "candidate_fingerprint": self.candidate_fingerprint,
            "pilot_record_fingerprint": self.pilot_record_fingerprint,
            "audit_report_fingerprint": self.audit_report_fingerprint,
            "blueprint_id": self.blueprint_id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "lifecycle_state": self.lifecycle_state.value,
            "policy_criticality": self.policy_criticality.value,
            "approved_use_ids": list(self.approved_use_ids),
            "evidence_references": [
                reference.to_dict() for reference in self.evidence_references
            ],
            "previous_record_fingerprint": self.previous_record_fingerprint,
            "transition_reason_id": self.transition_reason_id,
        }

    @property
    def record_fingerprint(self) -> str:
        """Return the creation-time complete SHA-256 identity of this record."""
        return self._record_fingerprint

    @property
    def record_id(self) -> str:
        """Return a descriptive 128-bit public lifecycle-record handle."""
        return f"item_bank_record_{self.record_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic public identities."""
        return {
            **self._content_dict(),
            "record_id": self.record_id,
            "record_fingerprint": self.record_fingerprint,
        }


def _verify_current_record(record: Any) -> ItemBankLifecycleRecord:
    """Replay one exact package record before granting transition authority."""
    if type(record) is not ItemBankLifecycleRecord:
        raise ItemBankLifecycleError(
            "invalid_lifecycle_record",
            "$.current_record",
            "current record must be an exact ItemBankLifecycleRecord",
        )
    if _sha256_hex(record._content_dict()) != record.record_fingerprint:
        raise ItemBankLifecycleError(
            "lifecycle_record_replay_mismatch",
            "$.current_record",
            "current lifecycle record no longer matches its creation-time identity",
        )
    return record


def _create_record(
    *,
    item_id: str,
    item_version: str,
    candidate_fingerprint: str,
    pilot_record_fingerprint: str,
    audit_report_fingerprint: str,
    blueprint_id: str,
    rubric_id: str,
    rubric_version: str,
    lifecycle_state: ItemBankLifecycleState,
    policy_criticality: PolicyCriticality,
    approved_use_ids: tuple[str, ...],
    evidence_references: tuple[ItemBankEvidenceReference, ...],
    previous_record_fingerprint: str | None,
    transition_reason_id: str,
) -> ItemBankLifecycleRecord:
    """Create one sealed normalized lifecycle record through a private token."""
    return ItemBankLifecycleRecord(
        item_id=item_id,
        item_version=item_version,
        candidate_fingerprint=candidate_fingerprint,
        pilot_record_fingerprint=pilot_record_fingerprint,
        audit_report_fingerprint=audit_report_fingerprint,
        blueprint_id=blueprint_id,
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        lifecycle_state=lifecycle_state,
        policy_criticality=policy_criticality,
        approved_use_ids=approved_use_ids,
        evidence_references=evidence_references,
        previous_record_fingerprint=previous_record_fingerprint,
        transition_reason_id=transition_reason_id,
        _creation_token=_RECORD_CREATION_TOKEN,
    )


def build_item_bank_pilot_record(
    pilot_record: PilotCandidateRecord,
    *,
    item_version: str,
    policy_criticality: PolicyCriticality = PolicyCriticality.ORDINARY,
) -> ItemBankLifecycleRecord:
    """Create the initial post-admission lifecycle record for one verified pilot."""
    if type(pilot_record) is not PilotCandidateRecord:
        raise TypeError("pilot_record must be an exact PilotCandidateRecord")
    if pilot_record.lifecycle_state is not CandidateLifecycleState.PILOT:
        raise ItemBankLifecycleError(
            "invalid_pilot_state",
            "$.pilot_record.lifecycle_state",
            "pilot record must have the verified pilot lifecycle state",
        )
    return _create_record(
        item_id=pilot_record.item_id,
        item_version=item_version,
        candidate_fingerprint=pilot_record.candidate_fingerprint,
        pilot_record_fingerprint=pilot_record.pilot_record_fingerprint,
        audit_report_fingerprint=pilot_record.audit_report_fingerprint,
        blueprint_id=pilot_record.blueprint_id,
        rubric_id=pilot_record.rubric_id,
        rubric_version=pilot_record.rubric_version,
        lifecycle_state=ItemBankLifecycleState.PILOTING,
        policy_criticality=_enum_value(
            policy_criticality,
            PolicyCriticality,
            "policy_criticality",
        ),
        approved_use_ids=(),
        evidence_references=(),
        previous_record_fingerprint=None,
        transition_reason_id="pilot_admission",
    )


_ALLOWED_TRANSITIONS: dict[
    ItemBankLifecycleState,
    frozenset[ItemBankLifecycleState],
] = {
    ItemBankLifecycleState.PILOTING: frozenset(
        {ItemBankLifecycleState.CALIBRATED}
    ),
    ItemBankLifecycleState.CALIBRATED: frozenset(
        {ItemBankLifecycleState.APPROVED}
    ),
    ItemBankLifecycleState.APPROVED: frozenset(
        {ItemBankLifecycleState.ACTIVE}
    ),
    ItemBankLifecycleState.ACTIVE: frozenset(
        {ItemBankLifecycleState.SUSPENDED, ItemBankLifecycleState.RETIRED}
    ),
    ItemBankLifecycleState.SUSPENDED: frozenset(
        {ItemBankLifecycleState.ACTIVE, ItemBankLifecycleState.RETIRED}
    ),
    ItemBankLifecycleState.RETIRED: frozenset(),
}


def _missing_required_kinds(
    current_state: ItemBankLifecycleState,
    target_state: ItemBankLifecycleState,
    supplied_kinds: set[ItemBankEvidenceKind],
) -> tuple[str, ...]:
    """Return missing newly supplied evidence kinds for one allowed transition."""
    if target_state is ItemBankLifecycleState.CALIBRATED:
        required = {
            ItemBankEvidenceKind.CALIBRATION,
            ItemBankEvidenceKind.ITEM_FIT,
            ItemBankEvidenceKind.ITEM_INFORMATION,
        }
        missing = [kind.value for kind in required - supplied_kinds]
        if not supplied_kinds.intersection(
            {ItemBankEvidenceKind.DIF, ItemBankEvidenceKind.DIF_NOT_APPLICABLE}
        ):
            missing.append("dif_or_dif_not_applicable")
        return tuple(sorted(missing))
    elif target_state is ItemBankLifecycleState.APPROVED:
        required = {ItemBankEvidenceKind.APPROVAL}
    elif (
        current_state is ItemBankLifecycleState.SUSPENDED
        and target_state is ItemBankLifecycleState.ACTIVE
    ):
        required = {ItemBankEvidenceKind.APPROVAL, ItemBankEvidenceKind.DRIFT}
    elif target_state is ItemBankLifecycleState.SUSPENDED:
        required = {ItemBankEvidenceKind.SUSPENSION}
        if not supplied_kinds.intersection(
            {ItemBankEvidenceKind.DIF, ItemBankEvidenceKind.DRIFT}
        ):
            return ("dif_or_drift",)
    elif target_state is ItemBankLifecycleState.RETIRED:
        required = {ItemBankEvidenceKind.RETIREMENT}
    else:
        required = set()
    return tuple(sorted(kind.value for kind in required - supplied_kinds))


def transition_item_bank_record(
    current_record: ItemBankLifecycleRecord,
    target_state: ItemBankLifecycleState,
    *,
    evidence_references: Iterable[ItemBankEvidenceReference],
    transition_reason_id: str,
    approved_use_ids: Iterable[str] | None = None,
) -> ItemBankLifecycleRecord:
    """Create one evidence-gated immutable successor to an exact current record."""
    current = _verify_current_record(current_record)
    try:
        target = _enum_value(target_state, ItemBankLifecycleState, "target_state")
    except ValueError as exc:
        raise ItemBankLifecycleError(
            "invalid_target_state",
            "$.target_state",
            "target state is not supported by this lifecycle version",
        ) from None
    if target not in _ALLOWED_TRANSITIONS[current.lifecycle_state]:
        raise ItemBankLifecycleError(
            "invalid_lifecycle_transition",
            "$.target_state",
            "requested transition is not allowed from the current lifecycle state",
        )

    additions = _normalize_evidence_references(
        evidence_references,
        error_type=ItemBankLifecycleError,
    )
    combined = _normalize_evidence_references(
        (*current.evidence_references, *additions),
        error_type=ItemBankLifecycleError,
    )
    supplied_kinds = {reference.evidence_kind for reference in additions}
    if (
        target is ItemBankLifecycleState.CALIBRATED
        and ItemBankEvidenceKind.DIF in supplied_kinds
        and ItemBankEvidenceKind.DIF_NOT_APPLICABLE in supplied_kinds
    ):
        raise ItemBankLifecycleError(
            "conflicting_dif_applicability",
            "$.evidence_references",
            "calibration requires exactly one DIF applicability evidence class",
        )
    missing = _missing_required_kinds(
        current.lifecycle_state,
        target,
        supplied_kinds,
    )
    if missing:
        raise ItemBankLifecycleError(
            "missing_transition_evidence",
            "$.evidence_references",
            f"transition requires newly supplied evidence: {', '.join(missing)}",
        )

    if approved_use_ids is None:
        normalized_uses = current.approved_use_ids
    else:
        try:
            normalized_uses = tuple(
                sorted(
                    _identifier_tuple(
                        approved_use_ids,
                        "approved_use_ids",
                        minimum=0,
                        maximum=MAX_COLLECTION_VALUES,
                    )
                )
            )
        except ValueError as exc:
            raise ItemBankLifecycleError(
                "invalid_approved_use",
                "$.approved_use_ids",
                "approved use identifiers are invalid",
            ) from None
    if current.approved_use_ids and normalized_uses != current.approved_use_ids:
        raise ItemBankLifecycleError(
            "approved_use_mutation",
            "$.approved_use_ids",
            "a successor cannot change the approved use scope",
        )
    if target in {
        ItemBankLifecycleState.APPROVED,
        ItemBankLifecycleState.ACTIVE,
        ItemBankLifecycleState.SUSPENDED,
        ItemBankLifecycleState.RETIRED,
    } and not normalized_uses:
        raise ItemBankLifecycleError(
            "missing_approved_use",
            "$.approved_use_ids",
            "approved and later states require at least one approved use identifier",
        )

    try:
        normalized_reason = _identifier(
            transition_reason_id,
            "transition_reason_id",
        )
    except ValueError as exc:
        raise ItemBankLifecycleError(
            "invalid_transition_reason",
            "$.transition_reason_id",
            "transition reason must be a descriptive identifier",
        ) from None

    return _create_record(
        item_id=current.item_id,
        item_version=current.item_version,
        candidate_fingerprint=current.candidate_fingerprint,
        pilot_record_fingerprint=current.pilot_record_fingerprint,
        audit_report_fingerprint=current.audit_report_fingerprint,
        blueprint_id=current.blueprint_id,
        rubric_id=current.rubric_id,
        rubric_version=current.rubric_version,
        lifecycle_state=target,
        policy_criticality=current.policy_criticality,
        approved_use_ids=normalized_uses,
        evidence_references=combined,
        previous_record_fingerprint=current.record_fingerprint,
        transition_reason_id=normalized_reason,
    )


__all__ = [
    "ItemBankEvidenceKind",
    "ItemBankEvidenceReference",
    "ItemBankLifecycleError",
    "ItemBankLifecycleRecord",
    "ItemBankLifecycleState",
    "PolicyCriticality",
    "build_item_bank_pilot_record",
    "transition_item_bank_record",
]
