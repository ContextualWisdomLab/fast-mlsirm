"""Immutable governance contracts for item-bank lifecycle decisions.

This module owns lifecycle state, provenance, and fail-closed transition policy.
It intentionally performs no calibration, linking, DIF, information, exposure,
or drift arithmetic; those numerical operations remain Rust-owned elsewhere in
``fast-mlsirm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


_MAX_IDENTITY_LENGTH = 256
_HEX_DIGITS = frozenset("0123456789abcdef")


class ItemLifecycleState(str, Enum):
    """Governed lifecycle states for a versioned item-bank entry."""

    DRAFT = "draft"
    AUDITED = "audited"
    SCREENED = "screened"
    PILOTING = "piloting"
    CALIBRATED = "calibrated"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class ItemEvidenceKind(str, Enum):
    """Evidence categories that authorize specific lifecycle transitions."""

    AUDIT = "audit"
    SCREENING = "screening"
    PILOT_DESIGN = "pilot_design"
    CALIBRATION = "calibration"
    APPROVAL = "approval"
    RELEASE = "release"
    LINKING = "linking"
    SUSPENSION = "suspension"
    RETIREMENT = "retirement"


@dataclass(frozen=True, slots=True)
class ItemLifecycleDecision:
    """One immutable state transition and its exact evidence fingerprints."""

    from_state: ItemLifecycleState
    to_state: ItemLifecycleState
    evidence_fingerprints: tuple[tuple[ItemEvidenceKind, str], ...]

    def __post_init__(self) -> None:
        """Validate the immutable decision payload without caller callbacks."""

        if type(self.from_state) is not ItemLifecycleState:
            raise TypeError("from_state must be an ItemLifecycleState")
        if type(self.to_state) is not ItemLifecycleState:
            raise TypeError("to_state must be an ItemLifecycleState")
        if type(self.evidence_fingerprints) is not tuple:
            raise TypeError("evidence_fingerprints must be a tuple")
        for pair in self.evidence_fingerprints:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("each evidence fingerprint entry must be a two-item tuple")
            kind, fingerprint = pair
            if type(kind) is not ItemEvidenceKind:
                raise TypeError("evidence kind must be an ItemEvidenceKind")
            _validate_fingerprint(fingerprint, f"{kind.value}_fingerprint")


@dataclass(frozen=True, slots=True)
class ItemBankEntry:
    """Versioned item-bank identity, provenance, lifecycle state, and history.

    The contract is deliberately persistence-neutral. Downstream applications
    may store it under their own tenant, authorization, retention, and audit
    controls, but ``fast-mlsirm`` does not introduce a database or authority
    service here.
    """

    item_id: str
    item_version: str
    rubric_fingerprint: str
    blueprint_fingerprint: str
    generation_contract_fingerprint: str
    state: ItemLifecycleState
    claims_cross_version_comparability: bool = False
    history: tuple[ItemLifecycleDecision, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and provenance as exact immutable built-in values."""

        _validate_identity(self.item_id, "item_id")
        _validate_identity(self.item_version, "item_version")
        _validate_fingerprint(self.rubric_fingerprint, "rubric_fingerprint")
        _validate_fingerprint(self.blueprint_fingerprint, "blueprint_fingerprint")
        _validate_fingerprint(
            self.generation_contract_fingerprint,
            "generation_contract_fingerprint",
        )
        if type(self.state) is not ItemLifecycleState:
            raise TypeError("state must be an ItemLifecycleState")
        if type(self.claims_cross_version_comparability) is not bool:
            raise TypeError("claims_cross_version_comparability must be a bool")
        if type(self.history) is not tuple:
            raise TypeError("history must be a tuple")
        for decision in self.history:
            if type(decision) is not ItemLifecycleDecision:
                raise TypeError("history entries must be ItemLifecycleDecision values")


_ALLOWED_TRANSITIONS: dict[
    ItemLifecycleState,
    dict[ItemLifecycleState, frozenset[ItemEvidenceKind]],
] = {
    ItemLifecycleState.DRAFT: {
        ItemLifecycleState.AUDITED: frozenset({ItemEvidenceKind.AUDIT}),
    },
    ItemLifecycleState.AUDITED: {
        ItemLifecycleState.SCREENED: frozenset({ItemEvidenceKind.SCREENING}),
    },
    ItemLifecycleState.SCREENED: {
        ItemLifecycleState.PILOTING: frozenset({ItemEvidenceKind.PILOT_DESIGN}),
    },
    ItemLifecycleState.PILOTING: {
        ItemLifecycleState.CALIBRATED: frozenset({ItemEvidenceKind.CALIBRATION}),
    },
    ItemLifecycleState.CALIBRATED: {
        ItemLifecycleState.APPROVED: frozenset({ItemEvidenceKind.APPROVAL}),
    },
    ItemLifecycleState.APPROVED: {
        ItemLifecycleState.ACTIVE: frozenset({ItemEvidenceKind.RELEASE}),
    },
    ItemLifecycleState.ACTIVE: {
        ItemLifecycleState.SUSPENDED: frozenset({ItemEvidenceKind.SUSPENSION}),
        ItemLifecycleState.RETIRED: frozenset({ItemEvidenceKind.RETIREMENT}),
    },
    ItemLifecycleState.SUSPENDED: {
        ItemLifecycleState.ACTIVE: frozenset(
            {ItemEvidenceKind.APPROVAL, ItemEvidenceKind.RELEASE}
        ),
        ItemLifecycleState.RETIRED: frozenset({ItemEvidenceKind.RETIREMENT}),
    },
    ItemLifecycleState.RETIRED: {},
}


def advance_item_bank_entry(
    entry: ItemBankEntry,
    target_state: ItemLifecycleState,
    *,
    evidence_fingerprints: dict[ItemEvidenceKind, str],
) -> ItemBankEntry:
    """Return a new entry after a valid evidence-backed lifecycle transition.

    The transition graph is intentionally sequential: a generated/draft item
    cannot become operational merely because a caller supplies JSON claiming
    approval. Cross-version comparability additionally requires linking
    evidence before the first activation.

    Parameters
    ----------
    entry:
        Existing immutable bank entry.
    target_state:
        Requested next lifecycle state.
    evidence_fingerprints:
        Exact SHA-256-style fingerprints keyed by the governed evidence kind.
        The mapping is copied into a deterministic immutable tuple.

    Returns
    -------
    ItemBankEntry
        A new entry with the requested state and an appended immutable decision.

    Raises
    ------
    TypeError
        If caller-controlled values use unsupported container or subclass types.
    ValueError
        If the transition is disallowed or required evidence is absent/invalid.
    """

    if type(entry) is not ItemBankEntry:
        raise TypeError("entry must be an ItemBankEntry")
    if type(target_state) is not ItemLifecycleState:
        raise TypeError("target_state must be an ItemLifecycleState")
    if type(evidence_fingerprints) is not dict:
        raise TypeError("evidence_fingerprints must be a built-in dict")
    if entry.state is ItemLifecycleState.RETIRED:
        raise ValueError("retired is terminal")

    allowed = _ALLOWED_TRANSITIONS[entry.state]
    required = allowed.get(target_state)
    if required is None:
        raise ValueError(
            f"transition {entry.state.value} -> {target_state.value} is not allowed"
        )

    normalized: list[tuple[ItemEvidenceKind, str]] = []
    present: set[ItemEvidenceKind] = set()
    for kind, fingerprint in evidence_fingerprints.items():
        if type(kind) is not ItemEvidenceKind:
            raise TypeError("evidence keys must be ItemEvidenceKind values")
        _validate_fingerprint(fingerprint, f"{kind.value}_fingerprint")
        normalized.append((kind, fingerprint))
        present.add(kind)

    effective_required = set(required)
    if (
        entry.claims_cross_version_comparability
        and entry.state is ItemLifecycleState.APPROVED
        and target_state is ItemLifecycleState.ACTIVE
    ):
        effective_required.add(ItemEvidenceKind.LINKING)

    missing = sorted(
        effective_required.difference(present),
        key=lambda kind: kind.value,
    )
    if missing:
        raise ValueError(
            "missing required evidence: " + ", ".join(kind.value for kind in missing)
        )

    normalized.sort(key=lambda pair: pair[0].value)
    decision = ItemLifecycleDecision(
        from_state=entry.state,
        to_state=target_state,
        evidence_fingerprints=tuple(normalized),
    )
    return ItemBankEntry(
        item_id=entry.item_id,
        item_version=entry.item_version,
        rubric_fingerprint=entry.rubric_fingerprint,
        blueprint_fingerprint=entry.blueprint_fingerprint,
        generation_contract_fingerprint=entry.generation_contract_fingerprint,
        state=target_state,
        claims_cross_version_comparability=entry.claims_cross_version_comparability,
        history=entry.history + (decision,),
    )


def _validate_identity(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a built-in str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must not be empty")
    if len(value) > _MAX_IDENTITY_LENGTH:
        raise ValueError(f"{field} must be at most {_MAX_IDENTITY_LENGTH} characters")
    if value != stripped or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a single trimmed line")


def _validate_fingerprint(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a built-in str")
    if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")


__all__ = [
    "ItemBankEntry",
    "ItemEvidenceKind",
    "ItemLifecycleDecision",
    "ItemLifecycleState",
    "advance_item_bank_entry",
]
