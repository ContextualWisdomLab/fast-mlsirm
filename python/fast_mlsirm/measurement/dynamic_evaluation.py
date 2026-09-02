"""Versioned dynamic-evaluation item and run-snapshot contracts.

The module freezes the concrete item instances used by one evaluation run while
keeping adjudication, validation, anchor promotion, and cross-version linking as
separate evidence states. It performs no item generation, scoring, calibration,
or statistical linking arithmetic.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Iterable, TypeVar

DYNAMIC_EVALUATION_ITEM_CONTRACT_ID = "fast_mlsirm_dynamic_evaluation_item/v1"
MAX_DYNAMIC_EVALUATION_ITEMS = 10_000
MAX_DYNAMIC_EVALUATION_REFERENCES = 256
MAX_DYNAMIC_EVALUATION_CRITERIA = 128
MAX_DYNAMIC_EVALUATION_REFERENCE_CHARS = 256
_ITEM_TOKEN = object()
_SET_TOKEN = object()
_ENUM_T = TypeVar("_ENUM_T", bound=Enum)


class DynamicItemOrigin(str, Enum):
    """Provenance class for one concrete evaluation item instance."""

    AUTHORED = "authored"
    GENERATED = "generated"
    PRODUCTION_SAMPLE = "production_sample"
    PERTURBATION = "perturbation"
    SYNTHETIC_ADVERSARIAL = "synthetic_adversarial"


class EvaluationItemRole(str, Enum):
    """Operational role of an item in an evaluation run."""

    CANDIDATE = "candidate"
    ANCHOR = "anchor"
    CHALLENGE = "challenge"
    PRODUCTION_SAMPLE = "production_sample"


class ReferenceSemantics(str, Enum):
    """Meaning of the evidence used to evaluate an item response."""

    EXACT = "exact"
    CONSTRAINT = "constraint"
    ACCEPTABLE_SET = "acceptable_set"
    RUBRIC = "rubric"
    PAIRWISE = "pairwise"
    OPEN_ENDED = "open_ended"


class ReferenceStatus(str, Enum):
    """Governance status of an item's response-reference semantics."""

    UNRESOLVED = "unresolved"
    PROVISIONAL = "provisional"
    ADJUDICATION_REQUIRED = "adjudication_required"
    ADJUDICATED = "adjudicated"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"


class RegenerationStatus(str, Enum):
    """Evidence level for regenerating item content from recorded inputs."""

    UNAVAILABLE = "unavailable"
    INPUTS_RECORDED = "inputs_recorded"
    VERIFIED = "verified"


class LinkingStatus(str, Enum):
    """Comparability claim permitted for one frozen evaluation item set."""

    UNAVAILABLE = "unavailable"
    WITHIN_RUN_ONLY = "within_run_only"
    LINKED = "linked"


class DynamicEvaluationContractError(ValueError):
    """Stable fail-closed error for dynamic-evaluation contract violations."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Retain bounded machine-readable rejection metadata."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def _error(code: str, path: str, message: str) -> DynamicEvaluationContractError:
    return DynamicEvaluationContractError(code, path, message)


def _enum(value: Any, enum_type: type[_ENUM_T], path: str) -> _ENUM_T:
    """Admit an exact enum or its exact string value without caller protocols."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise TypeError(f"{path} must be a {enum_type.__name__} or exact string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error("invalid_enum_value", path, f"unsupported {enum_type.__name__}") from exc


def _reference(value: Any, path: str) -> str:
    """Validate one exact opaque reference without normalization."""
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if (
        not value
        or len(value) > MAX_DYNAMIC_EVALUATION_REFERENCE_CHARS
        or value != value.strip()
        or value.startswith("\ufeff")
        or value.endswith("\ufeff")
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise _error(
            "invalid_reference",
            path,
            "reference must be 1..256 Unicode scalar values without "
            "boundary whitespace or controls",
        )
    return value


def _optional_reference(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _reference(value, path)


def _reference_tuple(
    value: Iterable[str],
    path: str,
    *,
    maximum: int,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Copy and validate a bounded unique reference collection."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{path} must be a tuple or list")
    normalized = tuple(
        _reference(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if (not allow_empty and not normalized) or len(normalized) > maximum:
        lower = 0 if allow_empty else 1
        raise _error(
            "invalid_reference_count",
            path,
            f"reference collection must contain {lower}..{maximum} entries",
        )
    if len(set(normalized)) != len(normalized):
        raise _error(
            "duplicate_reference", path, "reference collection must be unique"
        )
    return normalized


def _sha256(value: Any, path: str) -> str:
    """Validate one complete lowercase hexadecimal SHA-256 digest."""
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise _error(
            "invalid_sha256",
            path,
            "digest must be 64 lowercase hexadecimal characters",
        )
    return value


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
    criterion_refs: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    generation_invocation_ref: str | None
    seed_ref: str | None
    regeneration_status: RegenerationStatus
    regeneration_evidence_ref: str | None
    adjudication_ref: str | None
    validation_evidence_refs: tuple[str, ...]
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent direct construction that bypasses the public builder."""
        if _admission_token is not _ITEM_TOKEN:
            raise ValueError(
                "DynamicEvaluationItemSnapshot must be created by build_dynamic_evaluation_item"
            )

    @property
    def contract_id(self) -> str:
        """Return the immutable Published Language identity."""
        return DYNAMIC_EVALUATION_ITEM_CONTRACT_ID

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible source-text-free item snapshot."""
        return {
            "contract_id": self.contract_id,
            "item_instance_ref": self.item_instance_ref,
            "blueprint_revision_ref": self.blueprint_revision_ref,
            "content_ref": self.content_ref,
            "content_sha256": self.content_sha256,
            "origin": self.origin.value,
            "role": self.role.value,
            "reference_semantics": self.reference_semantics.value,
            "reference_status": self.reference_status.value,
            "rubric_revision_ref": self.rubric_revision_ref,
            "criterion_refs": list(self.criterion_refs),
            "provenance_refs": list(self.provenance_refs),
            "generation_invocation_ref": self.generation_invocation_ref,
            "seed_ref": self.seed_ref,
            "regeneration_status": self.regeneration_status.value,
            "regeneration_evidence_ref": self.regeneration_evidence_ref,
            "adjudication_ref": self.adjudication_ref,
            "validation_evidence_refs": list(self.validation_evidence_refs),
        }


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
    criterion_refs: tuple[str, ...] | list[str],
    provenance_refs: tuple[str, ...] | list[str],
    generation_invocation_ref: str | None,
    regeneration_status: RegenerationStatus | str,
    seed_ref: str | None = None,
    regeneration_evidence_ref: str | None = None,
    adjudication_ref: str | None = None,
    validation_evidence_refs: tuple[str, ...] | list[str] = (),
) -> DynamicEvaluationItemSnapshot:
    """Freeze one dynamic item without treating adjudication as anchor approval."""
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
    if normalized_origin not in generated_origins and normalized_generation_ref is not None:
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
                "adjudicated reference status requires an immutable resolution reference",
            )
    elif normalized_adjudication_ref is not None:
        if normalized_reference_status is not ReferenceStatus.VALIDATED:
            raise _error(
                "unexpected_adjudication_resolution",
                "$.adjudication_ref",
                "only adjudicated or validated references may retain adjudication provenance",
            )

    if normalized_reference_status in {ReferenceStatus.VALIDATED, ReferenceStatus.INVALIDATED}:
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
            "validation evidence is admitted only for validated or invalidated references",
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
        blueprint_revision_ref=_reference(
            blueprint_revision_ref, "$.blueprint_revision_ref"
        ),
        content_ref=_reference(content_ref, "$.content_ref"),
        content_sha256=_sha256(content_sha256, "$.content_sha256"),
        origin=normalized_origin,
        role=normalized_role,
        reference_semantics=normalized_semantics,
        reference_status=normalized_reference_status,
        rubric_revision_ref=_reference(rubric_revision_ref, "$.rubric_revision_ref"),
        criterion_refs=_reference_tuple(
            criterion_refs,
            "$.criterion_refs",
            maximum=MAX_DYNAMIC_EVALUATION_CRITERIA,
            allow_empty=False,
        ),
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
    """Immutable concrete item set resolved for one evaluation run."""

    run_snapshot_ref: str
    blueprint_revision_ref: str
    items: tuple[DynamicEvaluationItemSnapshot, ...]
    linking_status: LinkingStatus
    linking_evidence_ref: str | None
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent direct construction outside the aggregate builder."""
        if _admission_token is not _SET_TOKEN:
            raise ValueError(
                "EvaluationItemSetSnapshot must be created by build_evaluation_item_set_snapshot"
            )

    @property
    def contract_id(self) -> str:
        """Return the immutable Published Language identity."""
        return DYNAMIC_EVALUATION_ITEM_CONTRACT_ID

    @property
    def anchor_item_refs(self) -> tuple[str, ...]:
        """Return the validated anchor identities present in this exact snapshot."""
        return tuple(
            item.item_instance_ref
            for item in self.items
            if item.role is EvaluationItemRole.ANCHOR
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible run snapshot."""
        return {
            "contract_id": self.contract_id,
            "run_snapshot_ref": self.run_snapshot_ref,
            "blueprint_revision_ref": self.blueprint_revision_ref,
            "items": [item.to_dict() for item in self.items],
            "linking_status": self.linking_status.value,
            "linking_evidence_ref": self.linking_evidence_ref,
        }


def build_evaluation_item_set_snapshot(
    *,
    run_snapshot_ref: str,
    blueprint_revision_ref: str,
    items: tuple[DynamicEvaluationItemSnapshot, ...] | list[DynamicEvaluationItemSnapshot],
    linking_status: LinkingStatus | str,
    linking_evidence_ref: str | None = None,
) -> EvaluationItemSetSnapshot:
    """Freeze the concrete item set, including a zero-anchor cold start."""
    if not isinstance(items, (tuple, list)) or not items:
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
    if any(type(item) is not DynamicEvaluationItemSnapshot for item in normalized_items):
        raise TypeError("$.items must contain exact DynamicEvaluationItemSnapshot values")

    normalized_blueprint_ref = _reference(
        blueprint_revision_ref, "$.blueprint_revision_ref"
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

    normalized_linking_status = _enum(
        linking_status, LinkingStatus, "$.linking_status"
    )
    normalized_linking_evidence_ref = _optional_reference(
        linking_evidence_ref, "$.linking_evidence_ref"
    )
    anchors = tuple(
        item
        for item in normalized_items
        if item.role is EvaluationItemRole.ANCHOR
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
        items=normalized_items,
        linking_status=normalized_linking_status,
        linking_evidence_ref=normalized_linking_evidence_ref,
        _admission_token=_SET_TOKEN,
    )
