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


def _base_specification() -> ModelSpecification:
    return ModelSpecification(
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=("discrimination", "difficulty"),
            compatible_dependence=frozenset(
                {DependenceKind.LSIRM, DependenceKind.MLSIRM, DependenceKind.DLSJM}
            ),
        ),
        dimensional_structure=DimensionalStructure(
            formulation_id="confirmatory",
            dimensions=2,
        ),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory",
            fixed_effects=("person_covariates",),
            random_effects=("group_intercept",),
            membership=MembershipStructure(
                classification=MembershipClassification.HIERARCHICAL,
                multiplicity=MembershipMultiplicity.SINGLE,
                weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
                classification_axes=("group",),
            ),
        ),
        estimation_plan=EstimationPlan("research_mmle", "rust", False, "base"),
        identification_contract=IdentificationContract(
            ("trait_location_scale",), False, "base"
        ),
        recovery_contract=RecoveryContract(("bias", "rmse"), False, "base"),
    )


def test_compiled_manifest_never_publishes_another_candidates_support_records() -> None:
    """Candidate manifests must not digest support evidence scoped to a sibling model."""
    base = _base_specification()
    lsirm_id = compile_dependence_candidates(base)[0].canonical_id
    lsirm_ready = replace(
        base,
        estimation_plan=EstimationPlan("mmle", "rust", True, lsirm_id),
        identification_contract=IdentificationContract(
            ("trait_location_scale", "dependence_geometry_alignment"),
            True,
            lsirm_id,
        ),
        recovery_contract=RecoveryContract(("bias", "rmse"), True, lsirm_id),
    )
    candidates = compile_dependence_candidates(
        lsirm_ready,
        evidence_by_candidate_id={
            lsirm_id: CapabilityEvidence(
                generative_equation_id="2plm_lsirm_eq_v1",
                primary_citations=("10.1007/s11336-021-09762-5",),
            )
        },
    )

    lsirm = candidates[0]
    assert lsirm.status is CapabilityStatus.SUPPORTED
    assert lsirm.estimation_plan == lsirm_ready.estimation_plan
    assert lsirm.identification_contract == lsirm_ready.identification_contract
    assert lsirm.recovery_contract == lsirm_ready.recovery_contract

    for candidate in candidates[1:]:
        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert candidate.estimation_plan == EstimationPlan(
            "", "", False, candidate.canonical_id
        )
        assert candidate.identification_contract == IdentificationContract(
            (), False, candidate.canonical_id
        )
        assert candidate.recovery_contract == RecoveryContract(
            (), False, candidate.canonical_id
        )

        manifest = candidate.to_manifest()
        assert manifest["estimation_plan"]["applies_to_candidate_id"] == candidate.canonical_id
        assert manifest["identification"]["applies_to_candidate_id"] == candidate.canonical_id
        assert manifest["recovery"]["applies_to_candidate_id"] == candidate.canonical_id
