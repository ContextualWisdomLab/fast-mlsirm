from __future__ import annotations

from dataclasses import replace

from fast_mlsirm.model_specification import (
    CapabilityEvidence,
    CapabilityStatus,
    DependenceKind,
    DimensionalStructure,
    EstimationPlan,
    GeneralizedMixedStructure,
    IdentificationContract,
    MembershipClassification,
    MembershipMultiplicity,
    MembershipStructure,
    MembershipWeightAuthority,
    ModelSpecification,
    RecoveryContract,
    ResponseKernel,
    compile_dependence_candidates,
)


_LSIRM_PRIMARY_DOI = "10.1007/s11336-021-09762-5"


def _base_spec() -> ModelSpecification:
    """Build an LSIRM-compatible research candidate with incomplete support evidence."""
    return ModelSpecification(
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=("discrimination", "difficulty"),
            compatible_dependence=frozenset({DependenceKind.LSIRM}),
        ),
        dimensional_structure=DimensionalStructure(
            formulation_id="confirmatory",
            dimensions=2,
        ),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="hierarchical",
            membership=MembershipStructure(
                classification=MembershipClassification.HIERARCHICAL,
                multiplicity=MembershipMultiplicity.SINGLE,
                weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
                classification_axes=("group",),
            ),
        ),
        estimation_plan=EstimationPlan(
            estimator_id="lsirm_mmle",
            computational_backend="rust",
            implemented=False,
            applies_to_candidate_id="",
        ),
        identification_contract=IdentificationContract(
            rules=("trait_location_scale", "dependence_geometry_alignment"),
            verified=False,
            applies_to_candidate_id="",
        ),
        recovery_contract=RecoveryContract(
            required_metrics=("bias", "rmse", "coverage", "convergence"),
            passing=False,
            applies_to_candidate_id="",
        ),
    )


def _ready_spec() -> tuple[ModelSpecification, str]:
    """Scope estimator, identification, and recovery evidence to the exact LSIRM candidate."""
    base = _base_spec()
    candidate_id = compile_dependence_candidates(base)[0].canonical_id
    ready = replace(
        base,
        estimation_plan=replace(
            base.estimation_plan,
            implemented=True,
            applies_to_candidate_id=candidate_id,
        ),
        identification_contract=replace(
            base.identification_contract,
            verified=True,
            applies_to_candidate_id=candidate_id,
        ),
        recovery_contract=replace(
            base.recovery_contract,
            passing=True,
            applies_to_candidate_id=candidate_id,
        ),
    )
    return ready, candidate_id


def test_unrelated_citation_cannot_promote_lsirm_candidate() -> None:
    """A nonblank but unrelated citation must not satisfy LSIRM primary-paper evidence."""
    ready, candidate_id = _ready_spec()
    evidence = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.0000/unrelated",),
    )

    candidate = compile_dependence_candidates(
        ready,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == ("primary_citation_required",)


def test_named_lsirm_primary_citation_can_satisfy_citation_gate() -> None:
    """The dependence family's canonical primary paper satisfies the citation gate."""
    ready, candidate_id = _ready_spec()
    evidence = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=(_LSIRM_PRIMARY_DOI,),
    )

    candidate = compile_dependence_candidates(
        ready,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0]

    assert candidate.status is CapabilityStatus.SUPPORTED
    assert candidate.missing_requirements == ()
