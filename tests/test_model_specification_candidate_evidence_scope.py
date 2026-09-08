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


def _lsirm_evidence() -> CapabilityEvidence:
    return CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )


def _ready_for_candidate(
    base: ModelSpecification,
    candidate_id: str,
    *,
    recovery_metrics: tuple[str, ...] = ("bias", "rmse"),
) -> ModelSpecification:
    return replace(
        base,
        estimation_plan=EstimationPlan("mmle", "rust", True, candidate_id),
        identification_contract=IdentificationContract(
            ("trait_location_scale", "dependence_geometry_alignment"),
            True,
            candidate_id,
        ),
        recovery_contract=RecoveryContract(
            recovery_metrics,
            True,
            candidate_id,
        ),
    )


def test_compiled_manifest_never_publishes_another_candidates_support_records() -> None:
    """Candidate manifests must not digest support evidence scoped to a sibling model."""
    base = _base_specification()
    lsirm_id = compile_dependence_candidates(base)[0].canonical_id
    lsirm_ready = _ready_for_candidate(base, lsirm_id)
    candidates = compile_dependence_candidates(
        lsirm_ready,
        evidence_by_candidate_id={lsirm_id: _lsirm_evidence()},
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
        assert (
            manifest["estimation_plan"]["applies_to_candidate_id"]
            == candidate.canonical_id
        )
        assert (
            manifest["identification"]["applies_to_candidate_id"]
            == candidate.canonical_id
        )
        assert manifest["recovery"]["applies_to_candidate_id"] == candidate.canonical_id


def test_membership_recovery_cannot_be_borrowed_from_a_sibling_candidate() -> None:
    """Candidate scope also governs model-estimated membership-weight recovery."""
    base = replace(
        _base_specification(),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory_multiple_membership",
            fixed_effects=("person_covariates",),
            random_effects=("group_intercept",),
            membership=MembershipStructure(
                classification=MembershipClassification.HIERARCHICAL,
                multiplicity=MembershipMultiplicity.MULTIPLE,
                weight_authority=MembershipWeightAuthority.MODEL_ESTIMATED,
                classification_axes=("group",),
                weight_recovery_metric="membership_weight_rmse",
            ),
        ),
    )
    lsirm_id = compile_dependence_candidates(base)[0].canonical_id
    lsirm_ready = _ready_for_candidate(
        base,
        lsirm_id,
        recovery_metrics=("bias", "rmse", "membership_weight_rmse"),
    )
    candidates = compile_dependence_candidates(
        lsirm_ready,
        evidence_by_candidate_id={lsirm_id: _lsirm_evidence()},
    )

    assert candidates[0].status is CapabilityStatus.SUPPORTED
    for candidate in candidates[1:]:
        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert "membership_weight_recovery_required" in candidate.missing_requirements
        assert "passing_recovery_evidence_required" in candidate.missing_requirements
        assert candidate.recovery_contract == RecoveryContract(
            (), False, candidate.canonical_id
        )
