from __future__ import annotations

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


def _base_spec(*, family: str = "2plm", dimensions: int = 2) -> ModelSpecification:
    return ModelSpecification(
        response_kernel=ResponseKernel(
            family_id=family,
            formulation_id=f"{family}_logistic",
            response_scale="dichotomous",
            parameter_blocks=("discrimination", "difficulty"),
            compatible_dependence=_ALL_DEPENDENCE,
        ),
        dimensional_structure=DimensionalStructure(
            formulation_id="confirmatory",
            dimensions=dimensions,
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
        ),
        identification_contract=IdentificationContract(
            rules=("trait_location_scale", "dependence_geometry_alignment"),
            verified=False,
        ),
        recovery_contract=RecoveryContract(
            required_metrics=(
                "bias",
                "mae",
                "rmse",
                "coverage",
                "convergence",
            ),
            passing=False,
        ),
    )


def test_new_compatible_kernel_auto_expands_all_dependence_families() -> None:
    candidates = compile_dependence_candidates(_base_spec(family="new_binary_kernel"))

    assert tuple(candidate.dependence.kind for candidate in candidates) == (
        DependenceKind.LSIRM,
        DependenceKind.MLSIRM,
        DependenceKind.DLSJM,
    )
    assert {candidate.status for candidate in candidates} == {
        CapabilityStatus.RESEARCH_CANDIDATE
    }
    assert tuple(candidate.canonical_id for candidate in candidates) == (
        "new_binary_kernel_logistic__lsirm_jeon_et_al_2021_extension",
        "new_binary_kernel_logistic__mlsirm_kang_jeon_2025_extension",
        "new_binary_kernel_logistic__dlsjm_jin_jeon_2019_extension",
    )


def test_unidimensional_kernel_keeps_mlsirm_as_typed_unsupported_candidate() -> None:
    candidates = compile_dependence_candidates(_base_spec(dimensions=1))
    mlsirm = next(
        candidate for candidate in candidates if candidate.dependence.kind is DependenceKind.MLSIRM
    )

    assert mlsirm.status is CapabilityStatus.UNSUPPORTED
    assert mlsirm.missing_requirements == ("multidimensional_main_effects_required",)


def test_dlsjm_is_not_an_lsirm_alias() -> None:
    candidates = compile_dependence_candidates(_base_spec())
    lsirm = next(
        candidate for candidate in candidates if candidate.dependence.kind is DependenceKind.LSIRM
    )
    dlsjm = next(
        candidate for candidate in candidates if candidate.dependence.kind is DependenceKind.DLSJM
    )

    assert dlsjm.dependence.formulation_id == "dlsjm_jin_jeon_2019_extension"
    assert dlsjm.dependence.parameter_blocks == (
        "item_dependence_position",
        "person_dependence_position",
    )
    assert lsirm.dependence.parameter_blocks == (
        "person_interaction_position",
        "item_interaction_position",
        "interaction_strength",
    )
    assert dlsjm.dependence.parameter_blocks != lsirm.dependence.parameter_blocks


def test_expansion_preserves_base_parameters_and_serializes_stably() -> None:
    base = _base_spec()
    first = compile_dependence_candidates(base)
    second = compile_dependence_candidates(base)

    assert [candidate.to_manifest() for candidate in first] == [
        candidate.to_manifest() for candidate in second
    ]
    for candidate in first:
        assert candidate.response_kernel.parameter_blocks == (
            "discrimination",
            "difficulty",
        )
        assert candidate.to_manifest()["response_kernel"]["parameter_blocks"] == [
            "discrimination",
            "difficulty",
        ]
        assert candidate.to_manifest()["temporal_boundary"] == "tepp_owned"


def test_incompatible_dependence_is_classified_not_dropped_or_fallback() -> None:
    base = _base_spec()
    restricted = ModelSpecification(
        response_kernel=ResponseKernel(
            family_id=base.response_kernel.family_id,
            formulation_id=base.response_kernel.formulation_id,
            response_scale=base.response_kernel.response_scale,
            parameter_blocks=base.response_kernel.parameter_blocks,
            compatible_dependence=frozenset({DependenceKind.LSIRM}),
        ),
        dimensional_structure=base.dimensional_structure,
        mixed_structure=base.mixed_structure,
        estimation_plan=base.estimation_plan,
        identification_contract=base.identification_contract,
        recovery_contract=base.recovery_contract,
    )

    candidates = compile_dependence_candidates(restricted)

    assert len(candidates) == 3
    assert next(
        candidate for candidate in candidates if candidate.dependence.kind is DependenceKind.LSIRM
    ).status is CapabilityStatus.RESEARCH_CANDIDATE
    for kind in (DependenceKind.MLSIRM, DependenceKind.DLSJM):
        candidate = next(item for item in candidates if item.dependence.kind is kind)
        assert candidate.status is CapabilityStatus.UNSUPPORTED
        assert candidate.missing_requirements == ("base_kernel_declares_dependence_incompatible",)


def test_supported_requires_complete_equation_estimator_identification_citation_and_recovery() -> None:
    base = _base_spec()
    incomplete = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
        rust_estimator_implemented=True,
        identification_verified=True,
        recovery_passed=False,
    )

    candidate = compile_dependence_candidates(
        base,
        evidence_by_kind={DependenceKind.LSIRM: incomplete},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == ("passing_recovery_evidence_required",)

    complete = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
        rust_estimator_implemented=True,
        identification_verified=True,
        recovery_passed=True,
    )
    supported = compile_dependence_candidates(
        base,
        evidence_by_kind={DependenceKind.LSIRM: complete},
    )[0]

    assert supported.status is CapabilityStatus.SUPPORTED
    assert supported.missing_requirements == ()
    assert supported.generative_equation_id == "2plm_lsirm_eq_v1"
    assert supported.primary_citations == ("10.1007/s11336-021-09762-5",)
