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


def _membership() -> MembershipStructure:
    return MembershipStructure(
        classification=MembershipClassification.HIERARCHICAL,
        multiplicity=MembershipMultiplicity.MULTIPLE,
        weight_authority=MembershipWeightAuthority.EXPLICIT_NORMALIZED,
        classification_axes=("group",),
    )


def _mutable_base() -> tuple[
    ModelSpecification,
    list[str],
    set[DependenceKind],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    parameter_blocks = ["discrimination", "difficulty"]
    compatible_dependence = {
        DependenceKind.LSIRM,
        DependenceKind.MLSIRM,
        DependenceKind.DLSJM,
    }
    fixed_effects = ["person_covariates", "item_covariates"]
    random_effects = ["group_intercept"]
    identification_rules = ["trait_location_scale", "dependence_geometry_alignment"]
    recovery_metrics = ["bias", "mae", "rmse", "coverage", "convergence"]
    base = ModelSpecification(
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=parameter_blocks,  # type: ignore[arg-type]
            compatible_dependence=compatible_dependence,  # type: ignore[arg-type]
        ),
        dimensional_structure=DimensionalStructure(
            formulation_id="confirmatory",
            dimensions=2,
        ),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory_multiple_membership",
            fixed_effects=fixed_effects,  # type: ignore[arg-type]
            random_effects=random_effects,  # type: ignore[arg-type]
            membership=_membership(),
        ),
        estimation_plan=EstimationPlan(
            estimator_id="research_mmle",
            computational_backend="rust",
            implemented=False,
            applies_to_candidate_id="2plm_logistic",
        ),
        identification_contract=IdentificationContract(
            rules=identification_rules,  # type: ignore[arg-type]
            verified=False,
            applies_to_candidate_id="2plm_logistic",
        ),
        recovery_contract=RecoveryContract(
            required_metrics=recovery_metrics,  # type: ignore[arg-type]
            passing=False,
            applies_to_candidate_id="2plm_logistic",
        ),
    )
    return (
        base,
        parameter_blocks,
        compatible_dependence,
        fixed_effects,
        random_effects,
        identification_rules,
        recovery_metrics,
    )


def test_structural_collections_are_sealed_before_candidate_identity_is_created() -> None:
    (
        base,
        parameter_blocks,
        compatible_dependence,
        fixed_effects,
        random_effects,
        identification_rules,
        recovery_metrics,
    ) = _mutable_base()
    candidate = compile_dependence_candidates(base)[0]
    before_id = candidate.canonical_id
    before_manifest = candidate.to_manifest()

    parameter_blocks.append("guessing")
    compatible_dependence.clear()
    fixed_effects.append("group_predictors")
    random_effects.append("item_intercept")
    identification_rules.append("mutated_rule")
    recovery_metrics.append("mutated_metric")

    assert candidate.canonical_id == before_id
    assert candidate.to_manifest() == before_manifest
    assert base.response_kernel.parameter_blocks == ("discrimination", "difficulty")
    assert base.response_kernel.compatible_dependence == frozenset(
        {
            DependenceKind.LSIRM,
            DependenceKind.MLSIRM,
            DependenceKind.DLSJM,
        }
    )
    assert base.mixed_structure.fixed_effects == (
        "person_covariates",
        "item_covariates",
    )
    assert base.mixed_structure.random_effects == ("group_intercept",)
    assert base.mixed_structure.membership.classification is MembershipClassification.HIERARCHICAL
    assert base.mixed_structure.membership.multiplicity is MembershipMultiplicity.MULTIPLE
    assert base.mixed_structure.membership.weight_authority is MembershipWeightAuthority.EXPLICIT_NORMALIZED
    assert base.identification_contract.rules == (
        "trait_location_scale",
        "dependence_geometry_alignment",
    )
    assert base.recovery_contract.required_metrics == (
        "bias",
        "mae",
        "rmse",
        "coverage",
        "convergence",
    )


def _candidate_ready_base() -> tuple[ModelSpecification, str]:
    base, *_ = _mutable_base()
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


def _evidence() -> CapabilityEvidence:
    return CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )


def test_blank_estimator_identity_cannot_promote_candidate() -> None:
    base, candidate_id = _candidate_ready_base()
    candidate = compile_dependence_candidates(
        replace(
            base,
            estimation_plan=replace(base.estimation_plan, estimator_id=""),
        ),
        evidence_by_candidate_id={candidate_id: _evidence()},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == ("rust_estimator_required",)


def test_empty_or_blank_identification_rules_cannot_promote_candidate() -> None:
    base, candidate_id = _candidate_ready_base()
    for rules in ((), ("",)):
        candidate = compile_dependence_candidates(
            replace(
                base,
                identification_contract=replace(
                    base.identification_contract,
                    rules=rules,
                ),
            ),
            evidence_by_candidate_id={candidate_id: _evidence()},
        )[0]

        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert candidate.missing_requirements == ("identification_evidence_required",)


def test_empty_or_blank_recovery_metrics_cannot_promote_candidate() -> None:
    base, candidate_id = _candidate_ready_base()
    for required_metrics in ((), ("",)):
        candidate = compile_dependence_candidates(
            replace(
                base,
                recovery_contract=replace(
                    base.recovery_contract,
                    required_metrics=required_metrics,
                ),
            ),
            evidence_by_candidate_id={candidate_id: _evidence()},
        )[0]

        assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
        assert candidate.missing_requirements == ("passing_recovery_evidence_required",)
