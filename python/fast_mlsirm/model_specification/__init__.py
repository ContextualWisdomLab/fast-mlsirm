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


@dataclass(frozen=True)
class DimensionalStructure:
    """Main-effect dimensional formulation, separate from dependence geometry."""

    formulation_id: str
    dimensions: int

    def __post_init__(self) -> None:
        """Reject an impossible dimension count before candidate compilation."""
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")


@dataclass(frozen=True)
class GeneralizedMixedStructure:
    """Declarative generalized mixed-model structure attached to a kernel."""

    formulation_id: str
    fixed_effects: tuple[str, ...] = ()
    random_effects: tuple[str, ...] = ()
    membership: str = "single"


@dataclass(frozen=True)
class EstimationPlan:
    """Estimator/backend evidence scoped to exactly one compiled candidate."""

    estimator_id: str
    computational_backend: str
    implemented: bool
    applies_to_candidate_id: str


@dataclass(frozen=True)
class IdentificationContract:
    """Identification evidence scoped to exactly one compiled candidate."""

    rules: tuple[str, ...]
    verified: bool
    applies_to_candidate_id: str


@dataclass(frozen=True)
class RecoveryContract:
    """Known-truth recovery evidence scoped to exactly one compiled candidate."""

    required_metrics: tuple[str, ...]
    passing: bool
    applies_to_candidate_id: str


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
    membership: str
    dependence_kind: DependenceKind
    dependence_formulation_id: str
    dependence_parameter_blocks: tuple[str, ...]

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
                "membership": self.membership,
            },
            "dependence": {
                "kind": self.dependence_kind.value,
                "formulation_id": self.dependence_formulation_id,
                "parameter_blocks": list(self.dependence_parameter_blocks),
            },
        }

    def canonical_json(self) -> str:
        """Serialize identity with fixed JSON ordering and separators."""
        return json.dumps(
            self.to_manifest(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

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

    def to_manifest(self) -> dict[str, object]:
        """Serialize the candidate deterministically using JSON-shaped values."""
        return {
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
                "membership": self.mixed_structure.membership,
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
        membership=base.mixed_structure.membership,
        dependence_kind=dependence.kind,
        dependence_formulation_id=dependence.formulation_id,
        dependence_parameter_blocks=dependence.parameter_blocks,
    )


def _exact_nonblank_string(value: object) -> bool:
    """Return whether ``value`` is an exact, non-blank built-in string."""
    return type(value) is str and bool(value.strip())


def _primary_citations_are_complete(citations: object) -> bool:
    """Require a non-empty exact tuple of exact, non-blank citation identities."""
    return (
        type(citations) is tuple
        and bool(citations)
        and all(_exact_nonblank_string(citation) for citation in citations)
    )


def _scope_matches(value: object, candidate_id: str) -> bool:
    """Require support evidence to name the exact full candidate identity."""
    return type(value) is str and value == candidate_id


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
        and type(base.estimation_plan.computational_backend) is str
        and base.estimation_plan.computational_backend == "rust"
        and _scope_matches(base.estimation_plan.applies_to_candidate_id, candidate_id)
    )
    has_identification = (
        base.identification_contract.verified is True
        and _scope_matches(
            base.identification_contract.applies_to_candidate_id,
            candidate_id,
        )
    )
    has_citations = (
        evidence is not None
        and _primary_citations_are_complete(evidence.primary_citations)
    )
    has_recovery = (
        base.recovery_contract.passing is True
        and _scope_matches(base.recovery_contract.applies_to_candidate_id, candidate_id)
    )
    checks = (
        (has_equation, _MISSING_SUPPORT_REQUIREMENTS[0]),
        (has_rust_estimator, _MISSING_SUPPORT_REQUIREMENTS[1]),
        (has_identification, _MISSING_SUPPORT_REQUIREMENTS[2]),
        (has_citations, _MISSING_SUPPORT_REQUIREMENTS[3]),
        (has_recovery, _MISSING_SUPPORT_REQUIREMENTS[4]),
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
    "ModelSpecification",
    "RecoveryContract",
    "ResponseKernel",
    "compile_dependence_candidates",
]
