"""Typed model-specification and dependence-candidate compilation contracts.

This bounded context composes model metadata only. It does not evaluate a
likelihood, estimate a parameter, simulate responses, score respondents, or
perform any other psychometric arithmetic. Production numerical work remains
owned by the Rust core.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class DependenceKind(str, Enum):
    """Residual-dependence families kept distinct by the domain model."""

    LSIRM = "lsirm"
    MLSIRM = "mlsirm"
    DLSJM = "dlsjm"


class CapabilityStatus(str, Enum):
    """Scientific/implementation maturity of one compiled candidate."""

    SUPPORTED = "supported"
    RESEARCH_CANDIDATE = "research_candidate"
    UNSUPPORTED = "unsupported"


class MembershipClassification(str, Enum):
    """Higher-level classification topology of a generalized mixed model."""

    HIERARCHICAL = "hierarchical"
    CROSS_CLASSIFIED = "cross_classified"


class MembershipMultiplicity(str, Enum):
    """Whether one observation belongs to one or several units per structure."""

    SINGLE = "single"
    MULTIPLE = "multiple"


class MembershipWeightAuthority(str, Enum):
    """Authority for weights used by a multiple-membership formulation."""

    NOT_APPLICABLE = "not_applicable"
    EXPLICIT_NORMALIZED = "explicit_normalized"
    MODEL_ESTIMATED = "model_estimated"


_MANIFEST_SCHEMA_ID = "fast_mlsirm.model_specification.candidate_manifest"
_MANIFEST_SCHEMA_VERSION = "1.0.0"


def _exact_nonblank_string(value: object) -> bool:
    """Return whether ``value`` is an exact, non-blank built-in string."""
    return type(value) is str and bool(value.strip())


def _snapshot_builtin_sequence(value: object, field_name: str) -> tuple[object, ...]:
    """Copy an exact built-in list/tuple without invoking caller protocols."""
    if type(value) not in (list, tuple):
        raise TypeError(f"{field_name} must be a built-in list or tuple")
    return tuple(value)


def _immutable_exact_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """Copy a built-in list/tuple into an immutable tuple of exact strings."""
    snapshot = _snapshot_builtin_sequence(value, field_name)
    if any(type(item) is not str for item in snapshot):
        raise TypeError(f"{field_name} must contain only built-in strings")
    return snapshot  # type: ignore[return-value]


def _immutable_string_tuple(
    value: object,
    field_name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    """Copy an exact built-in list/tuple into an immutable exact-string tuple."""
    snapshot = _snapshot_builtin_sequence(value, field_name)
    if require_nonempty and not snapshot:
        raise ValueError(f"{field_name} must not be empty")
    if any(not _exact_nonblank_string(item) for item in snapshot):
        raise ValueError(f"{field_name} must contain only non-blank strings")
    return snapshot  # type: ignore[return-value]


def _immutable_dependence_set(value: object) -> frozenset[DependenceKind]:
    """Copy a built-in dependence collection without caller-owned mutability."""
    if type(value) not in (list, tuple, set, frozenset):
        raise TypeError(
            "compatible_dependence must be a built-in list, tuple, set, or frozenset"
        )
    snapshot = tuple(value)
    if any(type(item) is not DependenceKind for item in snapshot):
        raise ValueError("compatible_dependence must contain only DependenceKind values")
    return frozenset(snapshot)


def _nonempty_exact_string_tuple(values: object) -> bool:
    """Return whether evidence is a non-empty tuple of exact non-blank strings."""
    return (
        type(values) is tuple
        and bool(values)
        and all(_exact_nonblank_string(value) for value in values)
    )


def _primary_citations_are_complete(citations: object) -> bool:
    """Require a non-empty exact tuple of exact, non-blank citation identities."""
    return _nonempty_exact_string_tuple(citations)


def _scope_matches(value: object, candidate_id: str) -> bool:
    """Require support evidence to name the exact full candidate identity."""
    return type(value) is str and value == candidate_id


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize a JSON-shaped mapping deterministically for digesting."""
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class ResponseKernel:
    """Base conditional response-kernel identity and parameter ownership.

    ``compatible_dependence`` is declarative kernel metadata. The compiler does
    not contain a switch over response-family names, so a newly registered base
    kernel participates in dependence expansion without compiler modification.
    """

    family_id: str
    formulation_id: str
    response_scale: str
    parameter_blocks: tuple[str, ...]
    compatible_dependence: frozenset[DependenceKind]

    def __post_init__(self) -> None:
        """Seal caller-owned structural collections into immutable built-ins."""
        for field_name in ("family_id", "formulation_id", "response_scale"):
            if not _exact_nonblank_string(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a non-blank string")
        object.__setattr__(
            self,
            "parameter_blocks",
            _immutable_string_tuple(self.parameter_blocks, "parameter_blocks"),
        )
        object.__setattr__(
            self,
            "compatible_dependence",
            _immutable_dependence_set(self.compatible_dependence),
        )


@dataclass(frozen=True)
class DimensionalStructure:
    """Main-effect dimensional formulation, separate from dependence geometry."""

    formulation_id: str
    dimensions: int

    def __post_init__(self) -> None:
        """Reject malformed dimensional metadata before candidate compilation."""
        if not _exact_nonblank_string(self.formulation_id):
            raise ValueError("formulation_id must be a non-blank string")
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")


@dataclass(frozen=True)
class MembershipStructure:
    """Typed generalized-mixed membership and weight authority.

    Cross-classification describes non-nested classification axes; multiple
    membership describes belonging to more than one unit. They are orthogonal
    operators. Weight authority is explicit so a caller cannot hide an
    unspecified weighting rule behind a free-form label.
    """

    classification: MembershipClassification
    multiplicity: MembershipMultiplicity
    weight_authority: MembershipWeightAuthority
    classification_axes: tuple[str, ...]
    weight_recovery_metric: str | None = None

    def __post_init__(self) -> None:
        """Validate topology, multiplicity, and weight-source invariants."""
        if type(self.classification) is not MembershipClassification:
            raise TypeError("classification must be a MembershipClassification")
        if type(self.multiplicity) is not MembershipMultiplicity:
            raise TypeError("multiplicity must be a MembershipMultiplicity")
        if type(self.weight_authority) is not MembershipWeightAuthority:
            raise TypeError("weight_authority must be a MembershipWeightAuthority")
        axes = _immutable_string_tuple(
            self.classification_axes,
            "classification_axes",
            require_nonempty=True,
        )
        object.__setattr__(self, "classification_axes", axes)
        if (
            self.classification is MembershipClassification.CROSS_CLASSIFIED
            and len(axes) < 2
        ):
            raise ValueError("cross-classified membership requires at least two axes")
        if self.multiplicity is MembershipMultiplicity.SINGLE:
            if self.weight_authority is not MembershipWeightAuthority.NOT_APPLICABLE:
                raise ValueError("single membership cannot have membership weights")
            if self.weight_recovery_metric is not None:
                raise ValueError("single membership cannot have a weight recovery metric")
            return
        if self.weight_authority is MembershipWeightAuthority.NOT_APPLICABLE:
            raise ValueError("multiple membership requires weight authority")
        if self.weight_authority is MembershipWeightAuthority.MODEL_ESTIMATED:
            if not _exact_nonblank_string(self.weight_recovery_metric):
                raise ValueError(
                    "model-estimated membership weights require a recovery metric"
                )
        elif self.weight_recovery_metric is not None:
            raise ValueError(
                "explicit normalized membership weights cannot declare an estimator recovery metric"
            )

    def to_manifest(self) -> dict[str, object]:
        """Return the JSON-shaped membership contract."""
        return {
            "classification": self.classification.value,
            "multiplicity": self.multiplicity.value,
            "weight_authority": self.weight_authority.value,
            "classification_axes": list(self.classification_axes),
            "weight_recovery_metric": self.weight_recovery_metric,
        }


@dataclass(frozen=True)
class GeneralizedMixedStructure:
    """Declarative generalized mixed-model structure attached to a kernel."""

    formulation_id: str
    fixed_effects: tuple[str, ...] = ()
    random_effects: tuple[str, ...] = ()
    membership: MembershipStructure = MembershipStructure(
        classification=MembershipClassification.HIERARCHICAL,
        multiplicity=MembershipMultiplicity.SINGLE,
        weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
        classification_axes=("group",),
    )

    def __post_init__(self) -> None:
        """Seal generalized-mixed identity before it can enter candidate IDs."""
        if not _exact_nonblank_string(self.formulation_id):
            raise ValueError("formulation_id must be a non-blank string")
        if type(self.membership) is not MembershipStructure:
            raise TypeError("membership must be a MembershipStructure")
        object.__setattr__(
            self,
            "fixed_effects",
            _immutable_string_tuple(self.fixed_effects, "fixed_effects"),
        )
        object.__setattr__(
            self,
            "random_effects",
            _immutable_string_tuple(self.random_effects, "random_effects"),
        )


@dataclass(frozen=True)
class EstimationPlan:
    """Estimator/backend evidence scoped to exactly one compiled candidate."""

    estimator_id: str
    computational_backend: str
    implemented: bool
    applies_to_candidate_id: str

    def __post_init__(self) -> None:
        """Keep estimator evidence JSON-safe while permitting incomplete records."""
        for field_name in (
            "estimator_id",
            "computational_backend",
            "applies_to_candidate_id",
        ):
            if type(getattr(self, field_name)) is not str:
                raise TypeError(f"{field_name} must be a built-in string")
        if type(self.implemented) is not bool:
            raise TypeError("implemented must be a bool")


@dataclass(frozen=True)
class IdentificationContract:
    """Identification evidence scoped to exactly one compiled candidate."""

    rules: tuple[str, ...]
    verified: bool
    applies_to_candidate_id: str

    def __post_init__(self) -> None:
        """Seal rule evidence while allowing an explicitly incomplete research record."""
        object.__setattr__(
            self,
            "rules",
            _immutable_exact_string_tuple(self.rules, "rules"),
        )
        if type(self.verified) is not bool:
            raise TypeError("verified must be a bool")
        if type(self.applies_to_candidate_id) is not str:
            raise TypeError("applies_to_candidate_id must be a built-in string")


@dataclass(frozen=True)
class RecoveryContract:
    """Known-truth recovery evidence scoped to exactly one compiled candidate."""

    required_metrics: tuple[str, ...]
    passing: bool
    applies_to_candidate_id: str

    def __post_init__(self) -> None:
        """Seal recovery-metric evidence while preserving research-candidate states."""
        object.__setattr__(
            self,
            "required_metrics",
            _immutable_exact_string_tuple(self.required_metrics, "required_metrics"),
        )
        if type(self.passing) is not bool:
            raise TypeError("passing must be a bool")
        if type(self.applies_to_candidate_id) is not str:
            raise TypeError("applies_to_candidate_id must be a built-in string")


@dataclass(frozen=True)
class ModelSpecification:
    """Base model aggregate before residual-dependence expansion."""

    response_kernel: ResponseKernel
    dimensional_structure: DimensionalStructure
    mixed_structure: GeneralizedMixedStructure
    estimation_plan: EstimationPlan
    identification_contract: IdentificationContract
    recovery_contract: RecoveryContract


@dataclass(frozen=True)
class DependenceStructure:
    """One formulation-qualified residual-dependence parameter block."""

    kind: DependenceKind
    formulation_id: str
    parameter_blocks: tuple[str, ...]
    baseline_citations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Seal dependence identity and documentary baseline references."""
        if type(self.kind) is not DependenceKind:
            raise TypeError("kind must be a DependenceKind")
        if not _exact_nonblank_string(self.formulation_id):
            raise ValueError("formulation_id must be a non-blank string")
        object.__setattr__(
            self,
            "parameter_blocks",
            _immutable_string_tuple(self.parameter_blocks, "parameter_blocks"),
        )
        object.__setattr__(
            self,
            "baseline_citations",
            _immutable_string_tuple(
                self.baseline_citations,
                "baseline_citations",
                require_nonempty=True,
            ),
        )


@dataclass(frozen=True)
class CandidateIdentity:
    """Full immutable scientific identity of one compiled model candidate.

    Evidence and implementation state are intentionally excluded: the same
    scientific specification keeps its identity while its maturity advances.
    Every structural axis that can change the represented model is included.
    """

    response_family_id: str
    response_formulation_id: str
    response_scale: str
    response_parameter_blocks: tuple[str, ...]
    dimensional_formulation_id: str
    dimensions: int
    mixed_formulation_id: str
    fixed_effects: tuple[str, ...]
    random_effects: tuple[str, ...]
    membership_classification: MembershipClassification
    membership_multiplicity: MembershipMultiplicity
    membership_weight_authority: MembershipWeightAuthority
    membership_classification_axes: tuple[str, ...]
    membership_weight_recovery_metric: str | None
    dependence_kind: DependenceKind
    dependence_formulation_id: str
    dependence_parameter_blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Guarantee identity fields cannot retain externally mutable sequences."""
        for field_name in (
            "response_family_id",
            "response_formulation_id",
            "response_scale",
            "dimensional_formulation_id",
            "mixed_formulation_id",
            "dependence_formulation_id",
        ):
            if not _exact_nonblank_string(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a non-blank string")
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if type(self.dependence_kind) is not DependenceKind:
            raise TypeError("dependence_kind must be a DependenceKind")
        membership = MembershipStructure(
            classification=self.membership_classification,
            multiplicity=self.membership_multiplicity,
            weight_authority=self.membership_weight_authority,
            classification_axes=self.membership_classification_axes,
            weight_recovery_metric=self.membership_weight_recovery_metric,
        )
        object.__setattr__(
            self,
            "membership_classification_axes",
            membership.classification_axes,
        )
        for field_name in (
            "response_parameter_blocks",
            "fixed_effects",
            "random_effects",
            "dependence_parameter_blocks",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_string_tuple(getattr(self, field_name), field_name),
            )

    def membership_manifest(self) -> dict[str, object]:
        """Return membership identity in the shared manifest shape."""
        return MembershipStructure(
            classification=self.membership_classification,
            multiplicity=self.membership_multiplicity,
            weight_authority=self.membership_weight_authority,
            classification_axes=self.membership_classification_axes,
            weight_recovery_metric=self.membership_weight_recovery_metric,
        ).to_manifest()

    def to_manifest(self) -> dict[str, object]:
        """Return the complete JSON-shaped structural identity."""
        return {
            "response_kernel": {
                "family_id": self.response_family_id,
                "formulation_id": self.response_formulation_id,
                "response_scale": self.response_scale,
                "parameter_blocks": list(self.response_parameter_blocks),
            },
            "dimensional_structure": {
                "formulation_id": self.dimensional_formulation_id,
                "dimensions": self.dimensions,
            },
            "mixed_structure": {
                "formulation_id": self.mixed_formulation_id,
                "fixed_effects": list(self.fixed_effects),
                "random_effects": list(self.random_effects),
                "membership": self.membership_manifest(),
            },
            "dependence": {
                "kind": self.dependence_kind.value,
                "formulation_id": self.dependence_formulation_id,
                "parameter_blocks": list(self.dependence_parameter_blocks),
            },
        }

    def canonical_json(self) -> str:
        """Serialize identity with fixed JSON ordering and separators."""
        return _canonical_json_bytes(self.to_manifest()).decode("utf-8")

    def canonical_id(self) -> str:
        """Return a readable formulation prefix plus full SHA-256 identity."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return (
            f"{self.response_formulation_id}__{self.dependence_formulation_id}"
            f"__spec_sha256_{digest}"
        )


_MISSING_SUPPORT_REQUIREMENTS = (
    "generative_equation_required",
    "rust_estimator_required",
    "identification_evidence_required",
    "primary_citation_required",
    "membership_weight_recovery_required",
    "passing_recovery_evidence_required",
)


@dataclass(frozen=True)
class CapabilityEvidence:
    """Documentary evidence specific to one full candidate identity.

    Estimator implementation, identification verification, and recovery status
    remain on their owning candidate-scoped value objects; duplicating those
    booleans here would create competing sources of model truth.
    """

    generative_equation_id: str | None = None
    primary_citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Seal documentary collections without treating incompleteness as support."""
        if self.generative_equation_id is not None and type(self.generative_equation_id) is not str:
            raise TypeError("generative_equation_id must be a built-in string or None")
        object.__setattr__(
            self,
            "primary_citations",
            _immutable_exact_string_tuple(self.primary_citations, "primary_citations"),
        )


@dataclass(frozen=True)
class CompiledModelCandidate:
    """Immutable result of base + mixed + dependence composition."""

    identity: CandidateIdentity
    response_kernel: ResponseKernel
    dimensional_structure: DimensionalStructure
    mixed_structure: GeneralizedMixedStructure
    dependence: DependenceStructure
    estimation_plan: EstimationPlan
    identification_contract: IdentificationContract
    recovery_contract: RecoveryContract
    status: CapabilityStatus
    missing_requirements: tuple[str, ...]
    generative_equation_id: str | None
    primary_citations: tuple[str, ...]
    temporal_boundary: str = "tepp_owned"

    @property
    def canonical_id(self) -> str:
        """Expose the canonical ID derived from the single identity owner."""
        return self.identity.canonical_id()

    def _manifest_payload(self) -> dict[str, object]:
        """Build the versioned manifest payload before self-digesting."""
        return {
            "manifest_schema_id": _MANIFEST_SCHEMA_ID,
            "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
            "canonical_id": self.canonical_id,
            "identity": self.identity.to_manifest(),
            "status": self.status.value,
            "response_kernel": {
                "family_id": self.response_kernel.family_id,
                "formulation_id": self.response_kernel.formulation_id,
                "response_scale": self.response_kernel.response_scale,
                "parameter_blocks": list(self.response_kernel.parameter_blocks),
            },
            "dimensional_structure": {
                "formulation_id": self.dimensional_structure.formulation_id,
                "dimensions": self.dimensional_structure.dimensions,
            },
            "mixed_structure": {
                "formulation_id": self.mixed_structure.formulation_id,
                "fixed_effects": list(self.mixed_structure.fixed_effects),
                "random_effects": list(self.mixed_structure.random_effects),
                "membership": self.mixed_structure.membership.to_manifest(),
            },
            "dependence": {
                "kind": self.dependence.kind.value,
                "formulation_id": self.dependence.formulation_id,
                "parameter_blocks": list(self.dependence.parameter_blocks),
                "baseline_citations": list(self.dependence.baseline_citations),
            },
            "estimation_plan": {
                "estimator_id": self.estimation_plan.estimator_id,
                "computational_backend": self.estimation_plan.computational_backend,
                "implemented": self.estimation_plan.implemented,
                "applies_to_candidate_id": self.estimation_plan.applies_to_candidate_id,
            },
            "identification": {
                "rules": list(self.identification_contract.rules),
                "verified": self.identification_contract.verified,
                "applies_to_candidate_id": (
                    self.identification_contract.applies_to_candidate_id
                ),
            },
            "recovery": {
                "required_metrics": list(self.recovery_contract.required_metrics),
                "passing": self.recovery_contract.passing,
                "applies_to_candidate_id": self.recovery_contract.applies_to_candidate_id,
            },
            "generative_equation_id": self.generative_equation_id,
            "primary_citations": list(self.primary_citations),
            "missing_requirements": list(self.missing_requirements),
            "temporal_boundary": self.temporal_boundary,
        }

    def to_manifest(self) -> dict[str, object]:
        """Serialize the candidate with schema version and deterministic digest."""
        payload = self._manifest_payload()
        digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return {**payload, "manifest_sha256": digest}


_DEPENDENCE_TEMPLATES = MappingProxyType(
    {
        DependenceKind.LSIRM: DependenceStructure(
            kind=DependenceKind.LSIRM,
            formulation_id="lsirm_jeon_et_al_2021_extension",
            parameter_blocks=(
                "person_interaction_position",
                "item_interaction_position",
                "interaction_strength",
            ),
            baseline_citations=("10.1007/s11336-021-09762-5",),
        ),
        DependenceKind.MLSIRM: DependenceStructure(
            kind=DependenceKind.MLSIRM,
            formulation_id="mlsirm_kang_jeon_2025_extension",
            parameter_blocks=(
                "person_interaction_position",
                "item_interaction_position",
                "interaction_strength",
            ),
            baseline_citations=("10.1017/psy.2025.5",),
        ),
        DependenceKind.DLSJM: DependenceStructure(
            kind=DependenceKind.DLSJM,
            formulation_id="dlsjm_jin_jeon_2019_extension",
            parameter_blocks=(
                "item_dependence_position",
                "person_dependence_position",
            ),
            baseline_citations=("10.1007/s11336-018-9630-0",),
        ),
    }
)
_DEPENDENCE_ORDER = (
    DependenceKind.LSIRM,
    DependenceKind.MLSIRM,
    DependenceKind.DLSJM,
)


def _candidate_identity(
    base: ModelSpecification,
    dependence: DependenceStructure,
) -> CandidateIdentity:
    """Build the one structural identity used by IDs and evidence lookup."""
    membership = base.mixed_structure.membership
    return CandidateIdentity(
        response_family_id=base.response_kernel.family_id,
        response_formulation_id=base.response_kernel.formulation_id,
        response_scale=base.response_kernel.response_scale,
        response_parameter_blocks=base.response_kernel.parameter_blocks,
        dimensional_formulation_id=base.dimensional_structure.formulation_id,
        dimensions=base.dimensional_structure.dimensions,
        mixed_formulation_id=base.mixed_structure.formulation_id,
        fixed_effects=base.mixed_structure.fixed_effects,
        random_effects=base.mixed_structure.random_effects,
        membership_classification=membership.classification,
        membership_multiplicity=membership.multiplicity,
        membership_weight_authority=membership.weight_authority,
        membership_classification_axes=membership.classification_axes,
        membership_weight_recovery_metric=membership.weight_recovery_metric,
        dependence_kind=dependence.kind,
        dependence_formulation_id=dependence.formulation_id,
        dependence_parameter_blocks=dependence.parameter_blocks,
    )


def _missing_support_requirements(
    base: ModelSpecification,
    candidate_id: str,
    evidence: CapabilityEvidence | None,
) -> tuple[str, ...]:
    """Derive promotion gates from candidate-scoped canonical owners."""
    has_equation = (
        evidence is not None
        and _exact_nonblank_string(evidence.generative_equation_id)
    )
    has_rust_estimator = (
        base.estimation_plan.implemented is True
        and _exact_nonblank_string(base.estimation_plan.estimator_id)
        and type(base.estimation_plan.computational_backend) is str
        and base.estimation_plan.computational_backend == "rust"
        and _scope_matches(base.estimation_plan.applies_to_candidate_id, candidate_id)
    )
    has_identification = (
        base.identification_contract.verified is True
        and _nonempty_exact_string_tuple(base.identification_contract.rules)
        and _scope_matches(
            base.identification_contract.applies_to_candidate_id,
            candidate_id,
        )
    )
    has_citations = (
        evidence is not None
        and _primary_citations_are_complete(evidence.primary_citations)
    )
    membership = base.mixed_structure.membership
    membership_recovery_metric = membership.weight_recovery_metric
    has_membership_weight_recovery = (
        membership.weight_authority is not MembershipWeightAuthority.MODEL_ESTIMATED
        or (
            _exact_nonblank_string(membership_recovery_metric)
            and membership_recovery_metric in base.recovery_contract.required_metrics
        )
    )
    has_recovery = (
        base.recovery_contract.passing is True
        and _nonempty_exact_string_tuple(base.recovery_contract.required_metrics)
        and _scope_matches(base.recovery_contract.applies_to_candidate_id, candidate_id)
    )
    checks = (
        (has_equation, _MISSING_SUPPORT_REQUIREMENTS[0]),
        (has_rust_estimator, _MISSING_SUPPORT_REQUIREMENTS[1]),
        (has_identification, _MISSING_SUPPORT_REQUIREMENTS[2]),
        (has_citations, _MISSING_SUPPORT_REQUIREMENTS[3]),
        (has_membership_weight_recovery, _MISSING_SUPPORT_REQUIREMENTS[4]),
        (has_recovery, _MISSING_SUPPORT_REQUIREMENTS[5]),
    )
    return tuple(requirement for satisfied, requirement in checks if not satisfied)


def _published_equation_id(evidence: CapabilityEvidence | None) -> str | None:
    """Publish only an admitted exact equation identity."""
    if evidence is None or not _exact_nonblank_string(evidence.generative_equation_id):
        return None
    return evidence.generative_equation_id


def _published_primary_citations(
    evidence: CapabilityEvidence | None,
) -> tuple[str, ...]:
    """Publish only a complete citation-evidence tuple used by promotion."""
    if evidence is None or not _primary_citations_are_complete(evidence.primary_citations):
        return ()
    return evidence.primary_citations


def _compile_one(
    base: ModelSpecification,
    kind: DependenceKind,
    evidence_by_candidate_id: Mapping[str, CapabilityEvidence],
) -> CompiledModelCandidate:
    dependence = _DEPENDENCE_TEMPLATES[kind]
    identity = _candidate_identity(base, dependence)
    candidate_id = identity.canonical_id()
    evidence = evidence_by_candidate_id.get(candidate_id)

    if kind not in base.response_kernel.compatible_dependence:
        status = CapabilityStatus.UNSUPPORTED
        missing = ("base_kernel_declares_dependence_incompatible",)
    elif kind is DependenceKind.MLSIRM and base.dimensional_structure.dimensions < 2:
        status = CapabilityStatus.UNSUPPORTED
        missing = ("multidimensional_main_effects_required",)
    else:
        missing = _missing_support_requirements(base, candidate_id, evidence)
        status = (
            CapabilityStatus.RESEARCH_CANDIDATE
            if missing
            else CapabilityStatus.SUPPORTED
        )

    return CompiledModelCandidate(
        identity=identity,
        response_kernel=base.response_kernel,
        dimensional_structure=base.dimensional_structure,
        mixed_structure=base.mixed_structure,
        dependence=dependence,
        estimation_plan=base.estimation_plan,
        identification_contract=base.identification_contract,
        recovery_contract=base.recovery_contract,
        status=status,
        missing_requirements=missing,
        generative_equation_id=_published_equation_id(evidence),
        primary_citations=_published_primary_citations(evidence),
    )


def compile_dependence_candidates(
    base: ModelSpecification,
    *,
    evidence_by_candidate_id: Mapping[str, CapabilityEvidence] | None = None,
) -> tuple[CompiledModelCandidate, ...]:
    """Compile LSIRM, MLSIRM, and DLSJM variants without family-specific branching.

    Every dependence family is materialized. Scientific incompatibility becomes
    an explicit ``unsupported`` candidate; absent full-candidate-specific
    equation, implementation, identification, citation, or recovery evidence
    becomes ``research_candidate``. The compiler never substitutes the
    local-independent base model for a requested dependence structure.
    """
    evidence = {} if evidence_by_candidate_id is None else evidence_by_candidate_id
    return tuple(_compile_one(base, kind, evidence) for kind in _DEPENDENCE_ORDER)


__all__ = [
    "CandidateIdentity",
    "CapabilityEvidence",
    "CapabilityStatus",
    "CompiledModelCandidate",
    "DependenceKind",
    "DependenceStructure",
    "DimensionalStructure",
    "EstimationPlan",
    "GeneralizedMixedStructure",
    "IdentificationContract",
    "MembershipClassification",
    "MembershipMultiplicity",
    "MembershipStructure",
    "MembershipWeightAuthority",
    "ModelSpecification",
    "RecoveryContract",
    "ResponseKernel",
    "compile_dependence_candidates",
]
