"""Typed model-specification and dependence-candidate compilation contracts.

This bounded context composes model metadata only. It does not evaluate a
likelihood, estimate a parameter, simulate responses, score respondents, or
perform any other psychometric arithmetic. Production numerical work remains
owned by the Rust core.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class GeneralizedMixedStructure:
    """Declarative generalized mixed-model structure attached to a kernel."""

    formulation_id: str
    fixed_effects: tuple[str, ...] = ()
    random_effects: tuple[str, ...] = ()
    membership: str = "single"


@dataclass(frozen=True)
class EstimationPlan:
    """Estimator/backend declaration without numerical implementation logic."""

    estimator_id: str
    computational_backend: str
    implemented: bool


@dataclass(frozen=True)
class IdentificationContract:
    """Identification rules required before a model can support inference."""

    rules: tuple[str, ...]
    verified: bool


@dataclass(frozen=True)
class RecoveryContract:
    """Known-truth recovery metrics required before production support."""

    required_metrics: tuple[str, ...]
    passing: bool


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


_MISSING_SUPPORT_REQUIREMENTS = (
    "generative_equation_required",
    "rust_estimator_required",
    "identification_evidence_required",
    "primary_citation_required",
    "passing_recovery_evidence_required",
)


@dataclass(frozen=True)
class CapabilityEvidence:
    """Exact evidence needed to promote a research candidate to supported."""

    generative_equation_id: str | None = None
    primary_citations: tuple[str, ...] = ()
    rust_estimator_implemented: bool = False
    identification_verified: bool = False
    recovery_passed: bool = False

    def missing_requirements(self) -> tuple[str, ...]:
        """Return every absent support gate in deterministic contract order."""
        checks = (
            (self.generative_equation_id is not None, _MISSING_SUPPORT_REQUIREMENTS[0]),
            (self.rust_estimator_implemented, _MISSING_SUPPORT_REQUIREMENTS[1]),
            (self.identification_verified, _MISSING_SUPPORT_REQUIREMENTS[2]),
            (bool(self.primary_citations), _MISSING_SUPPORT_REQUIREMENTS[3]),
            (self.recovery_passed, _MISSING_SUPPORT_REQUIREMENTS[4]),
        )
        return tuple(requirement for satisfied, requirement in checks if not satisfied)


@dataclass(frozen=True)
class CompiledModelCandidate:
    """Immutable result of base + mixed + dependence composition."""

    canonical_id: str
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

    def to_manifest(self) -> dict[str, object]:
        """Serialize the candidate deterministically using JSON-shaped values."""
        return {
            "canonical_id": self.canonical_id,
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
            },
            "estimation_plan": {
                "estimator_id": self.estimation_plan.estimator_id,
                "computational_backend": self.estimation_plan.computational_backend,
                "implemented": self.estimation_plan.implemented,
            },
            "identification": {
                "rules": list(self.identification_contract.rules),
                "verified": self.identification_contract.verified,
            },
            "recovery": {
                "required_metrics": list(self.recovery_contract.required_metrics),
                "passing": self.recovery_contract.passing,
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


def _compile_one(
    base: ModelSpecification,
    kind: DependenceKind,
    evidence: CapabilityEvidence | None,
) -> CompiledModelCandidate:
    dependence = _DEPENDENCE_TEMPLATES[kind]
    canonical_id = f"{base.response_kernel.formulation_id}__{dependence.formulation_id}"

    if kind not in base.response_kernel.compatible_dependence:
        status = CapabilityStatus.UNSUPPORTED
        missing = ("base_kernel_declares_dependence_incompatible",)
    elif kind is DependenceKind.MLSIRM and base.dimensional_structure.dimensions < 2:
        status = CapabilityStatus.UNSUPPORTED
        missing = ("multidimensional_main_effects_required",)
    else:
        missing = (
            _MISSING_SUPPORT_REQUIREMENTS
            if evidence is None
            else evidence.missing_requirements()
        )
        status = (
            CapabilityStatus.RESEARCH_CANDIDATE
            if missing
            else CapabilityStatus.SUPPORTED
        )

    citations = dependence.baseline_citations
    equation_id = None
    if evidence is not None:
        equation_id = evidence.generative_equation_id
        if evidence.primary_citations:
            citations = evidence.primary_citations

    return CompiledModelCandidate(
        canonical_id=canonical_id,
        response_kernel=base.response_kernel,
        dimensional_structure=base.dimensional_structure,
        mixed_structure=base.mixed_structure,
        dependence=dependence,
        estimation_plan=base.estimation_plan,
        identification_contract=base.identification_contract,
        recovery_contract=base.recovery_contract,
        status=status,
        missing_requirements=missing,
        generative_equation_id=equation_id,
        primary_citations=citations,
    )


def compile_dependence_candidates(
    base: ModelSpecification,
    *,
    evidence_by_kind: Mapping[DependenceKind, CapabilityEvidence] | None = None,
) -> tuple[CompiledModelCandidate, ...]:
    """Compile LSIRM, MLSIRM, and DLSJM variants without family-specific branching.

    Every dependence family is materialized. Scientific incompatibility becomes
    an explicit ``unsupported`` candidate; absent implementation/identification/
    recovery evidence becomes ``research_candidate``. The compiler never
    substitutes the local-independent base model for a requested dependence
    structure.
    """
    evidence = {} if evidence_by_kind is None else evidence_by_kind
    return tuple(_compile_one(base, kind, evidence.get(kind)) for kind in _DEPENDENCE_ORDER)


__all__ = [
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
