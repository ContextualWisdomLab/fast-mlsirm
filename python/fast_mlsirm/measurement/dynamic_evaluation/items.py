"""Criterion-bound dynamic item and run snapshots."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from ._common import (
    DYNAMIC_EVALUATION_ITEM_CONTRACT_ID,
    MAX_DYNAMIC_EVALUATION_CRITERIA,
    MAX_DYNAMIC_EVALUATION_ITEMS,
    MAX_DYNAMIC_EVALUATION_REFERENCES,
    DynamicEvaluationContractError,
    DynamicItemOrigin,
    EvaluationItemRole,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
    _ITEM_TOKEN,
    _SET_TOKEN,
    _enum,
    _error,
    _fingerprint,
    _optional_reference,
    _reference,
    _reference_tuple,
    _sha256,
)
from .criteria import EvaluationCriterionSetSnapshot


@dataclass(frozen=True)
class DynamicEvaluationItemSnapshot:
    """One immutable concrete item instance used or eligible for one run."""

    item_instance_ref: str
    blueprint_revision_ref: str
    content_ref: str
    content_sha256: str
    origin: DynamicItemOrigin
    role: EvaluationItemRole
    reference_semantics: ReferenceSemantics
    reference_status: ReferenceStatus
    rubric_revision_ref: str
    criterion_set_snapshot_ref: str
    criterion_set_sha256: str
    criterion_refs: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    generation_invocation_ref: str | None
    seed_ref: str | None
    regeneration_status: RegenerationStatus
    regeneration_evidence_ref: str | None
    adjudication_ref: str | None
    validation_evidence_refs: tuple[str, ...]
    _admission_token: InitVar[object | None] = None
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent direct construction and record the admitted item fingerprint."""
        if _admission_token is not _ITEM_TOKEN:
            raise ValueError(
                "DynamicEvaluationItemSnapshot must be created by "
                "build_dynamic_evaluation_item"
            )
        object.__setattr__(
            self, "_integrity_sha256", _fingerprint(self._payload_unchecked())
        )

    @property
    def contract_id(self) -> str:
        """Return the immutable Published Language identity."""
        return DYNAMIC_EVALUATION_ITEM_CONTRACT_ID

    def _payload_unchecked(self) -> dict[str, Any]:
        """Return item data without recursively checking its own fingerprint."""
        return {
            "contract_id": DYNAMIC_EVALUATION_ITEM_CONTRACT_ID,
            "item_instance_ref": self.item_instance_ref,
            "blueprint_revision_ref": self.blueprint_revision_ref,
            "content_ref": self.content_ref,
            "content_sha256": self.content_sha256,
            "origin": self.origin.value,
            "role": self.role.value,
            "reference_semantics": self.reference_semantics.value,
            "reference_status": self.reference_status.value,
            "rubric_revision_ref": self.rubric_revision_ref,
            "criterion_set_snapshot_ref": self.criterion_set_snapshot_ref,
            "criterion_set_sha256": self.criterion_set_sha256,
            "criterion_refs": list(self.criterion_refs),
            "provenance_refs": list(self.provenance_refs),
            "generation_invocation_ref": self.generation_invocation_ref,
            "seed_ref": self.seed_ref,
            "regeneration_status": self.regeneration_status.value,
            "regeneration_evidence_ref": self.regeneration_evidence_ref,
            "adjudication_ref": self.adjudication_ref,
            "validation_evidence_refs": list(self.validation_evidence_refs),
        }

    def _assert_integrity(self) -> None:
        """Reject object-level mutation after factory admission."""
        try:
            current = _fingerprint(self._payload_unchecked())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _error(
                "item_snapshot_integrity_mismatch",
                "$.items",
                "item snapshot changed after admission",
            ) from exc
        if current != self._integrity_sha256:
            raise _error(
                "item_snapshot_integrity_mismatch",
                "$.items",
                "item snapshot changed after admission",
            )

    @property
    def snapshot_sha256(self) -> str:
        """Return the deterministic immutable item fingerprint."""
        self._assert_integrity()
        return self._integrity_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible source-text-free item snapshot."""
        self._assert_integrity()
        return {**self._payload_unchecked(), "snapshot_sha256": self._integrity_sha256}


def build_dynamic_evaluation_item(
    *,
    item_instance_ref: str,
    blueprint_revision_ref: str,
    content_ref: str,
    content_sha256: str,
    origin: DynamicItemOrigin | str,
    role: EvaluationItemRole | str,
    reference_semantics: ReferenceSemantics | str,
    reference_status: ReferenceStatus | str,
    rubric_revision_ref: str,
    criterion_set_snapshot: EvaluationCriterionSetSnapshot | None,
    criterion_refs: tuple[str, ...] | list[str],
    provenance_refs: tuple[str, ...] | list[str],
    generation_invocation_ref: str | None,
    regeneration_status: RegenerationStatus | str,
    seed_ref: str | None = None,
    regeneration_evidence_ref: str | None = None,
    adjudication_ref: str | None = None,
    validation_evidence_refs: tuple[str, ...] | list[str] = (),
) -> DynamicEvaluationItemSnapshot:
    """Freeze one candidate item under an exact admitted criterion set."""
    if criterion_set_snapshot is None:
        raise _error(
            "criterion_set_required",
            "$.criterion_set_snapshot",
            "dynamic items require a non-empty immutable criterion-set snapshot",
        )
    if type(criterion_set_snapshot) is not EvaluationCriterionSetSnapshot:
        raise TypeError(
            "$.criterion_set_snapshot must be an exact "
            "EvaluationCriterionSetSnapshot"
        )
    criterion_set_snapshot._assert_integrity()
    normalized_blueprint_ref = _reference(
        blueprint_revision_ref, "$.blueprint_revision_ref"
    )
    normalized_rubric_ref = _reference(rubric_revision_ref, "$.rubric_revision_ref")
    if criterion_set_snapshot.blueprint_revision_ref != normalized_blueprint_ref:
        raise _error(
            "criterion_set_blueprint_mismatch",
            "$.criterion_set_snapshot",
            "criterion set must belong to the item blueprint revision",
        )
    if criterion_set_snapshot.rubric_revision_ref != normalized_rubric_ref:
        raise _error(
            "criterion_set_rubric_mismatch",
            "$.criterion_set_snapshot",
            "criterion set must use the item rubric revision",
        )
    normalized_criterion_refs = _reference_tuple(
        criterion_refs,
        "$.criterion_refs",
        maximum=MAX_DYNAMIC_EVALUATION_CRITERIA,
        allow_empty=False,
    )
    unknown_criteria = set(normalized_criterion_refs) - set(
        criterion_set_snapshot.criterion_refs
    )
    if unknown_criteria:
        raise _error(
            "item_criterion_not_registered",
            "$.criterion_refs",
            "item criteria must resolve in the frozen criterion set",
        )
    normalized_origin = _enum(origin, DynamicItemOrigin, "$.origin")
    normalized_role = _enum(role, EvaluationItemRole, "$.role")
    normalized_semantics = _enum(
        reference_semantics, ReferenceSemantics, "$.reference_semantics"
    )
    normalized_reference_status = _enum(
        reference_status, ReferenceStatus, "$.reference_status"
    )
    normalized_regeneration_status = _enum(
        regeneration_status, RegenerationStatus, "$.regeneration_status"
    )

    normalized_generation_ref = _optional_reference(
        generation_invocation_ref, "$.generation_invocation_ref"
    )
    generated_origins = {
        DynamicItemOrigin.GENERATED,
        DynamicItemOrigin.PERTURBATION,
        DynamicItemOrigin.SYNTHETIC_ADVERSARIAL,
    }
    if normalized_origin in generated_origins and normalized_generation_ref is None:
        raise _error(
            "generated_item_requires_invocation",
            "$.generation_invocation_ref",
            "generated, perturbation, and synthetic items require generation identity",
        )
    if (
        normalized_origin not in generated_origins
        and normalized_generation_ref is not None
    ):
        raise _error(
            "unexpected_generation_invocation",
            "$.generation_invocation_ref",
            "non-generated items cannot claim generation identity",
        )

    normalized_adjudication_ref = _optional_reference(
        adjudication_ref, "$.adjudication_ref"
    )
    normalized_validation_refs = _reference_tuple(
        validation_evidence_refs,
        "$.validation_evidence_refs",
        maximum=MAX_DYNAMIC_EVALUATION_REFERENCES,
        allow_empty=True,
    )
    if normalized_reference_status is ReferenceStatus.ADJUDICATED:
        if normalized_adjudication_ref is None:
            raise _error(
                "adjudicated_reference_requires_resolution",
                "$.adjudication_ref",
                "adjudicated reference status requires an immutable "
                "resolution reference",
            )
    elif normalized_adjudication_ref is not None:
        if normalized_reference_status is not ReferenceStatus.VALIDATED:
            raise _error(
                "unexpected_adjudication_resolution",
                "$.adjudication_ref",
                "only adjudicated or validated references may retain "
                "adjudication provenance",
            )

    if normalized_reference_status in {
        ReferenceStatus.VALIDATED,
        ReferenceStatus.INVALIDATED,
    }:
        if not normalized_validation_refs:
            raise _error(
                "validated_reference_requires_evidence",
                "$.validation_evidence_refs",
                "validated and invalidated references require validation evidence",
            )
    elif normalized_validation_refs:
        raise _error(
            "unexpected_validation_evidence",
            "$.validation_evidence_refs",
            "validation evidence is admitted only for validated or "
            "invalidated references",
        )

    if (
        normalized_role is EvaluationItemRole.ANCHOR
        and normalized_reference_status is not ReferenceStatus.VALIDATED
    ):
        raise _error(
            "anchor_requires_validated_reference",
            "$.role",
            "an anchor requires validated reference semantics, not adjudication alone",
        )

    normalized_regeneration_evidence_ref = _optional_reference(
        regeneration_evidence_ref, "$.regeneration_evidence_ref"
    )
    if normalized_regeneration_status is RegenerationStatus.VERIFIED:
        if normalized_regeneration_evidence_ref is None:
            raise _error(
                "verified_regeneration_requires_evidence",
                "$.regeneration_evidence_ref",
                "verified regeneration requires independent validation evidence",
            )
    elif normalized_regeneration_evidence_ref is not None:
        raise _error(
            "unexpected_regeneration_evidence",
            "$.regeneration_evidence_ref",
            "recorded inputs or a seed alone do not prove deterministic regeneration",
        )

    return DynamicEvaluationItemSnapshot(
        item_instance_ref=_reference(item_instance_ref, "$.item_instance_ref"),
        blueprint_revision_ref=normalized_blueprint_ref,
        content_ref=_reference(content_ref, "$.content_ref"),
        content_sha256=_sha256(content_sha256, "$.content_sha256"),
        origin=normalized_origin,
        role=normalized_role,
        reference_semantics=normalized_semantics,
        reference_status=normalized_reference_status,
        rubric_revision_ref=normalized_rubric_ref,
        criterion_set_snapshot_ref=criterion_set_snapshot.criterion_set_snapshot_ref,
        criterion_set_sha256=criterion_set_snapshot.snapshot_sha256,
        criterion_refs=tuple(sorted(normalized_criterion_refs)),
        provenance_refs=_reference_tuple(
            provenance_refs,
            "$.provenance_refs",
            maximum=MAX_DYNAMIC_EVALUATION_REFERENCES,
            allow_empty=False,
        ),
        generation_invocation_ref=normalized_generation_ref,
        seed_ref=_optional_reference(seed_ref, "$.seed_ref"),
        regeneration_status=normalized_regeneration_status,
        regeneration_evidence_ref=normalized_regeneration_evidence_ref,
        adjudication_ref=normalized_adjudication_ref,
        validation_evidence_refs=normalized_validation_refs,
        _admission_token=_ITEM_TOKEN,
    )


@dataclass(frozen=True)
class EvaluationItemSetSnapshot:
    """Immutable criterion-bound concrete item set resolved for one run."""

    run_snapshot_ref: str
    blueprint_revision_ref: str
    criterion_set_snapshot: EvaluationCriterionSetSnapshot
    items: tuple[DynamicEvaluationItemSnapshot, ...]
    linking_status: LinkingStatus
    linking_evidence_ref: str | None
    _admission_token: InitVar[object | None] = None
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent direct construction and record the admitted run fingerprint."""
        if _admission_token is not _SET_TOKEN:
            raise ValueError(
                "EvaluationItemSetSnapshot must be created by "
                "build_evaluation_item_set_snapshot"
            )
        object.__setattr__(
            self, "_integrity_sha256", _fingerprint(self._payload_unchecked())
        )

    @property
    def contract_id(self) -> str:
        """Return the immutable Published Language identity."""
        return DYNAMIC_EVALUATION_ITEM_CONTRACT_ID

    @property
    def criterion_set_snapshot_ref(self) -> str:
        """Return the exact criterion-set identity bound to this run."""
        self._assert_integrity()
        return self.criterion_set_snapshot.criterion_set_snapshot_ref

    @property
    def criterion_set_sha256(self) -> str:
        """Return the exact criterion-set digest bound to this run."""
        self._assert_integrity()
        return self.criterion_set_snapshot.snapshot_sha256

    @property
    def criterion_refs(self) -> tuple[str, ...]:
        """Return the full admitted criterion set for the run."""
        self._assert_integrity()
        return self.criterion_set_snapshot.criterion_refs

    @property
    def anchor_item_refs(self) -> tuple[str, ...]:
        """Return validated anchor identities present in this exact snapshot."""
        self._assert_integrity()
        return tuple(
            item.item_instance_ref
            for item in self.items
            if item.role is EvaluationItemRole.ANCHOR
        )

    def _payload_unchecked(self) -> dict[str, Any]:
        """Return run identity data without recursively checking its fingerprint."""
        return {
            "contract_id": DYNAMIC_EVALUATION_ITEM_CONTRACT_ID,
            "run_snapshot_ref": self.run_snapshot_ref,
            "blueprint_revision_ref": self.blueprint_revision_ref,
            "criterion_set_snapshot_ref": (
                self.criterion_set_snapshot.criterion_set_snapshot_ref
            ),
            "criterion_set_sha256": self.criterion_set_snapshot.snapshot_sha256,
            "criterion_refs": list(self.criterion_set_snapshot.criterion_refs),
            "criterion_set": self.criterion_set_snapshot.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "linking_status": self.linking_status.value,
            "linking_evidence_ref": self.linking_evidence_ref,
        }

    def _assert_integrity(self) -> None:
        """Reject item, criterion, or run mutation after factory admission."""
        try:
            if type(self.criterion_set_snapshot) is not EvaluationCriterionSetSnapshot:
                raise TypeError("run contains a foreign criterion set")
            self.criterion_set_snapshot._assert_integrity()
            if any(
                type(item) is not DynamicEvaluationItemSnapshot for item in self.items
            ):
                raise TypeError("run contains a foreign item")
            for item in self.items:
                item._assert_integrity()
            current = _fingerprint(self._payload_unchecked())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _error(
                "run_snapshot_integrity_mismatch",
                "$.run_snapshot",
                "run snapshot changed after admission",
            ) from exc
        if current != self._integrity_sha256:
            raise _error(
                "run_snapshot_integrity_mismatch",
                "$.run_snapshot",
                "run snapshot changed after admission",
            )

    @property
    def snapshot_sha256(self) -> str:
        """Return the deterministic immutable run fingerprint."""
        self._assert_integrity()
        return self._integrity_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned criterion-bound run snapshot."""
        self._assert_integrity()
        return {**self._payload_unchecked(), "snapshot_sha256": self._integrity_sha256}


def build_evaluation_item_set_snapshot(
    *,
    run_snapshot_ref: str,
    blueprint_revision_ref: str,
    items: (
        tuple[DynamicEvaluationItemSnapshot, ...]
        | list[DynamicEvaluationItemSnapshot]
    ),
    linking_status: LinkingStatus | str,
    criterion_set_snapshot: EvaluationCriterionSetSnapshot | None = None,
    linking_evidence_ref: str | None = None,
) -> EvaluationItemSetSnapshot:
    """Freeze concrete items only after binding explicit evaluation criteria."""
    if criterion_set_snapshot is None:
        raise _error(
            "criterion_set_required",
            "$.criterion_set_snapshot",
            "evaluation requires a non-empty immutable criterion-set snapshot",
        )
    if type(criterion_set_snapshot) is not EvaluationCriterionSetSnapshot:
        raise TypeError(
            "$.criterion_set_snapshot must be an exact EvaluationCriterionSetSnapshot"
        )
    criterion_set_snapshot._assert_integrity()

    if type(items) not in (tuple, list):
        raise TypeError("$.items must be a tuple or list")
    if not items:
        raise _error(
            "invalid_item_set",
            "$.items",
            "item set must contain at least one item",
        )
    if len(items) > MAX_DYNAMIC_EVALUATION_ITEMS:
        raise _error(
            "item_set_budget_exceeded",
            "$.items",
            f"item set may contain at most {MAX_DYNAMIC_EVALUATION_ITEMS} items",
        )
    normalized_items = tuple(items)
    if any(
        type(item) is not DynamicEvaluationItemSnapshot for item in normalized_items
    ):
        raise TypeError(
            "$.items must contain exact DynamicEvaluationItemSnapshot values"
        )
    for item in normalized_items:
        item._assert_integrity()

    normalized_blueprint_ref = _reference(
        blueprint_revision_ref, "$.blueprint_revision_ref"
    )
    if criterion_set_snapshot.blueprint_revision_ref != normalized_blueprint_ref:
        raise _error(
            "criterion_set_blueprint_mismatch",
            "$.criterion_set_snapshot",
            "criterion set must belong to the run blueprint revision",
        )
    if any(
        item.blueprint_revision_ref != normalized_blueprint_ref
        for item in normalized_items
    ):
        raise _error(
            "item_blueprint_mismatch",
            "$.items",
            "every item must belong to the run snapshot blueprint revision",
        )
    item_refs = [item.item_instance_ref for item in normalized_items]
    if len(set(item_refs)) != len(item_refs):
        raise _error(
            "duplicate_item_instance",
            "$.items",
            "one concrete item instance may appear only once in a run snapshot",
        )

    registered = set(criterion_set_snapshot.criterion_refs)
    covered: set[str] = set()
    for item in normalized_items:
        if (
            item.criterion_set_snapshot_ref
            != criterion_set_snapshot.criterion_set_snapshot_ref
            or item.criterion_set_sha256 != criterion_set_snapshot.snapshot_sha256
        ):
            raise _error(
                "item_criterion_set_mismatch",
                "$.items",
                "every item must retain the run criterion-set identity and digest",
            )
        covered.update(item.criterion_refs)
    if covered != registered:
        raise _error(
            "criterion_set_not_covered",
            "$.criteria",
            "every admitted criterion must be administered by at least one item",
        )

    normalized_linking_status = _enum(
        linking_status, LinkingStatus, "$.linking_status"
    )
    normalized_linking_evidence_ref = _optional_reference(
        linking_evidence_ref, "$.linking_evidence_ref"
    )
    anchors = tuple(
        item for item in normalized_items if item.role is EvaluationItemRole.ANCHOR
    )
    if normalized_linking_status is LinkingStatus.LINKED:
        if not anchors:
            raise _error(
                "linked_snapshot_requires_anchor",
                "$.linking_status",
                "cross-version linking requires at least one validated anchor",
            )
        if normalized_linking_evidence_ref is None:
            raise _error(
                "linked_snapshot_requires_evidence",
                "$.linking_evidence_ref",
                "linked status requires an immutable linking evidence reference",
            )
    elif normalized_linking_evidence_ref is not None:
        raise _error(
            "unexpected_linking_evidence",
            "$.linking_evidence_ref",
            "unavailable and within-run-only snapshots cannot claim linking evidence",
        )

    return EvaluationItemSetSnapshot(
        run_snapshot_ref=_reference(run_snapshot_ref, "$.run_snapshot_ref"),
        blueprint_revision_ref=normalized_blueprint_ref,
        criterion_set_snapshot=criterion_set_snapshot,
        items=normalized_items,
        linking_status=normalized_linking_status,
        linking_evidence_ref=normalized_linking_evidence_ref,
        _admission_token=_SET_TOKEN,
    )
