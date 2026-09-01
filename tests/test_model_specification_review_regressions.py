from __future__ import annotations

from dataclasses import replace
from typing import cast

from fast_mlsirm.model_specification import (
    CapabilityEvidence,
    CapabilityStatus,
    DependenceKind,
    DimensionalStructure,
    EstimationPlan,
    GeneralizedMixedStructure,
    IdentificationContract,
    ModelSpecification,
    RecoveryContract,
    ResponseKernel,
    compile_dependence_candidates,
)


_ALL_DEPENDENCE = frozenset(
    {DependenceKind.LSIRM, DependenceKind.MLSIRM, DependenceKind.DLSJM}
)


def _base_spec() -> ModelSpecification:
    return ModelSpecification(
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=("discrimination", "difficulty"),
            compatible_dependence=_ALL_DEPENDENCE,
        ),
        dimensional_structure=DimensionalStructure(
            formulation_id="confirmatory",
            dimensions=2,
        ),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory_multiple_membership",
            fixed_effects=("person_covariates", "item_covariates"),
            random_effects=("group_intercept",),
            membership="multiple_membership",
        ),
        estimation_plan=EstimationPlan(
            estimator_id="research_mmle",
            computational_backend="rust",
            implemented=False,
            applies_to_candidate_id="2plm_logistic",
        ),
        identification_contract=IdentificationContract(
            rules=("trait_location_scale", "dependence_geometry_alignment"),
            verified=False,
            applies_to_candidate_id="2plm_logistic",
        ),
        recovery_contract=RecoveryContract(
            required_metrics=("bias", "mae", "rmse", "coverage", "convergence"),
            passing=False,
            applies_to_candidate_id="2plm_logistic",
        ),
    )


def _complete_evidence() -> CapabilityEvidence:
    return CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )


def _ready_for_lsirm(base: ModelSpecification) -> tuple[ModelSpecification, str]:
    candidate_id = compile_dependence_candidates(base)[0].canonical_id
    return (
        replace(
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
        ),
        candidate_id,
    )


def test_structural_collections_are_frozen_at_domain_boundaries() -> None:
    parameter_blocks = ["discrimination", "difficulty"]
    compatible_dependence = {
        DependenceKind.LSIRM,
        DependenceKind.MLSIRM,
        DependenceKind.DLSJM,
    }
    fixed_effects = ["person_covariates", "item_covariates"]
    random_effects = ["group_intercept"]

    base = replace(
        _base_spec(),
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=cast(tuple[str, ...], parameter_blocks),
            compatible_dependence=cast(
                frozenset[DependenceKind], compatible_dependence
            ),
        ),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory_multiple_membership",
            fixed_effects=cast(tuple[str, ...], fixed_effects),
            random_effects=cast(tuple[str, ...], random_effects),
            membership="multiple_membership",
        ),
    )
    candidate = compile_dependence_candidates(base)[0]
    canonical_id = candidate.canonical_id
    manifest = candidate.to_manifest()

    parameter_blocks.append("guessing")
    compatible_dependence.clear()
    fixed_effects.append("occasion")
    random_effects.clear()

    assert candidate.canonical_id == canonical_id
    assert candidate.to_manifest() == manifest
    assert candidate.response_kernel.parameter_blocks == (
        "discrimination",
        "difficulty",
    )
    assert candidate.response_kernel.compatible_dependence == _ALL_DEPENDENCE
    assert candidate.mixed_structure.fixed_effects == (
        "person_covariates",
        "item_covariates",
    )
    assert candidate.mixed_structure.random_effects == ("group_intercept",)


def test_support_requires_substantive_candidate_scoped_records() -> None:
    base, candidate_id = _ready_for_lsirm(_base_spec())
    evidence = {candidate_id: _complete_evidence()}

    blank_estimator = replace(
        base,
        estimation_plan=replace(base.estimation_plan, estimator_id="  "),
    )
    assert compile_dependence_candidates(
        blank_estimator,
        evidence_by_candidate_id=evidence,
    )[0].missing_requirements == ("rust_estimator_required",)

    for rules in ((), ("  ",)):
        blank_identification = replace(
            base,
            identification_contract=replace(base.identification_contract, rules=rules),
        )
        candidate = compile_dependence_candidates(
            blank_identification,
            evidence_by_candidate_id=evidence,
        )[0]
        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert candidate.missing_requirements == ("identification_evidence_required",)

    for required_metrics in ((), ("  ",)):
        blank_recovery = replace(
            base,
            recovery_contract=replace(
                base.recovery_contract,
                required_metrics=required_metrics,
            ),
        )
        candidate = compile_dependence_candidates(
            blank_recovery,
            evidence_by_candidate_id=evidence,
        )[0]
        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert candidate.missing_requirements == (
            "passing_recovery_evidence_required",
        )
