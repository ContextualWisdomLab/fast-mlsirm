from __future__ import annotations

from dataclasses import fields, replace

import pytest

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
_LSIRM_ID = "2plm_logistic__lsirm_jeon_et_al_2021_extension"


def _base_spec(
    *,
    family: str = "2plm",
    dimensions: int = 2,
    support_ready: bool = False,
    support_formulation_id: str | None = None,
) -> ModelSpecification:
    base_formulation_id = f"{family}_logistic"
    evidence_scope = (
        base_formulation_id
        if support_formulation_id is None
        else support_formulation_id
    )
    return ModelSpecification(
        response_kernel=ResponseKernel(
            family_id=family,
            formulation_id=base_formulation_id,
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
            implemented=support_ready,
            applies_to_formulation_id=evidence_scope,
        ),
        identification_contract=IdentificationContract(
            rules=("trait_location_scale", "dependence_geometry_alignment"),
            verified=support_ready,
            applies_to_formulation_id=evidence_scope,
        ),
        recovery_contract=RecoveryContract(
            required_metrics=(
                "bias",
                "mae",
                "rmse",
                "coverage",
                "convergence",
            ),
            passing=support_ready,
            applies_to_formulation_id=evidence_scope,
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
        candidate
        for candidate in candidates
        if candidate.dependence.kind is DependenceKind.MLSIRM
    )

    assert mlsirm.status is CapabilityStatus.UNSUPPORTED
    assert mlsirm.missing_requirements == ("multidimensional_main_effects_required",)


def test_nonpositive_dimensionality_fails_closed() -> None:
    with pytest.raises(ValueError, match="dimensions must be >= 1"):
        DimensionalStructure(formulation_id="confirmatory", dimensions=0)


def test_dlsjm_is_not_an_lsirm_alias() -> None:
    candidates = compile_dependence_candidates(_base_spec())
    lsirm = next(
        candidate
        for candidate in candidates
        if candidate.dependence.kind is DependenceKind.LSIRM
    )
    dlsjm = next(
        candidate
        for candidate in candidates
        if candidate.dependence.kind is DependenceKind.DLSJM
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
        candidate
        for candidate in candidates
        if candidate.dependence.kind is DependenceKind.LSIRM
    ).status is CapabilityStatus.RESEARCH_CANDIDATE
    for kind in (DependenceKind.MLSIRM, DependenceKind.DLSJM):
        candidate = next(item for item in candidates if item.dependence.kind is kind)
        assert candidate.status is CapabilityStatus.UNSUPPORTED
        assert candidate.missing_requirements == (
            "base_kernel_declares_dependence_incompatible",
        )


def test_capability_evidence_has_no_duplicate_estimator_or_recovery_truth() -> None:
    assert tuple(field.name for field in fields(CapabilityEvidence)) == (
        "generative_equation_id",
        "primary_citations",
    )


def test_generic_base_evidence_cannot_promote_a_dependence_extension() -> None:
    evidence = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )

    candidate = compile_dependence_candidates(
        _base_spec(support_ready=True),
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == (
        "rust_estimator_required",
        "identification_evidence_required",
        "passing_recovery_evidence_required",
    )


def test_supported_requires_formulation_scoped_rust_identification_and_recovery() -> None:
    evidence = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )
    base = _base_spec(
        support_ready=True,
        support_formulation_id=_LSIRM_ID,
    )

    supported = compile_dependence_candidates(
        base,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]

    assert supported.status is CapabilityStatus.SUPPORTED
    assert supported.missing_requirements == ()
    assert supported.generative_equation_id == "2plm_lsirm_eq_v1"
    assert supported.primary_citations == ("10.1007/s11336-021-09762-5",)
    assert supported.estimation_plan.applies_to_formulation_id == _LSIRM_ID
    assert supported.identification_contract.applies_to_formulation_id == _LSIRM_ID
    assert supported.recovery_contract.applies_to_formulation_id == _LSIRM_ID

    not_implemented = replace(
        base,
        estimation_plan=replace(base.estimation_plan, implemented=False),
    )
    candidate = compile_dependence_candidates(
        not_implemented,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]
    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == ("rust_estimator_required",)

    wrong_backend = replace(
        base,
        estimation_plan=replace(base.estimation_plan, computational_backend="numpy"),
    )
    candidate = compile_dependence_candidates(
        wrong_backend,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]
    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == ("rust_estimator_required",)

    unidentified = replace(
        base,
        identification_contract=replace(base.identification_contract, verified=False),
    )
    candidate = compile_dependence_candidates(
        unidentified,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]
    assert candidate.missing_requirements == ("identification_evidence_required",)

    unrecovered = replace(
        base,
        recovery_contract=replace(base.recovery_contract, passing=False),
    )
    candidate = compile_dependence_candidates(
        unrecovered,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0]
    assert candidate.missing_requirements == ("passing_recovery_evidence_required",)


def test_scope_mismatch_fails_each_support_gate_independently() -> None:
    evidence = CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )
    base = _base_spec(
        support_ready=True,
        support_formulation_id=_LSIRM_ID,
    )

    wrong_estimator_scope = replace(
        base,
        estimation_plan=replace(
            base.estimation_plan,
            applies_to_formulation_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_estimator_scope,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0].missing_requirements == ("rust_estimator_required",)

    wrong_identification_scope = replace(
        base,
        identification_contract=replace(
            base.identification_contract,
            applies_to_formulation_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_identification_scope,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0].missing_requirements == ("identification_evidence_required",)

    wrong_recovery_scope = replace(
        base,
        recovery_contract=replace(
            base.recovery_contract,
            applies_to_formulation_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_recovery_scope,
        evidence_by_kind={DependenceKind.LSIRM: evidence},
    )[0].missing_requirements == ("passing_recovery_evidence_required",)


def test_blank_equation_or_citation_cannot_promote_candidate() -> None:
    base = _base_spec(
        support_ready=True,
        support_formulation_id=_LSIRM_ID,
    )
    blank = CapabilityEvidence(generative_equation_id="", primary_citations=("",))

    candidate = compile_dependence_candidates(
        base,
        evidence_by_kind={DependenceKind.LSIRM: blank},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == (
        "generative_equation_required",
        "primary_citation_required",
    )
