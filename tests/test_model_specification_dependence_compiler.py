from __future__ import annotations

from dataclasses import fields, replace

import pytest

from fast_mlsirm.model_specification import (
    CandidateIdentity,
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


def _base_spec(
    *,
    family: str = "2plm",
    dimensions: int = 2,
    fixed_effects: tuple[str, ...] = ("person_covariates", "item_covariates"),
    random_effects: tuple[str, ...] = ("group_intercept",),
    membership: str = "multiple_membership",
) -> ModelSpecification:
    base_formulation_id = f"{family}_logistic"
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
            fixed_effects=fixed_effects,
            random_effects=random_effects,
            membership=membership,
        ),
        estimation_plan=EstimationPlan(
            estimator_id="research_mmle",
            computational_backend="rust",
            implemented=False,
            applies_to_candidate_id=base_formulation_id,
        ),
        identification_contract=IdentificationContract(
            rules=("trait_location_scale", "dependence_geometry_alignment"),
            verified=False,
            applies_to_candidate_id=base_formulation_id,
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
            applies_to_candidate_id=base_formulation_id,
        ),
    )


def _ready_for_candidate(base: ModelSpecification, candidate_id: str) -> ModelSpecification:
    return replace(
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


def _lsirm_id(base: ModelSpecification) -> str:
    return compile_dependence_candidates(base)[0].canonical_id


def _complete_lsirm_evidence() -> CapabilityEvidence:
    return CapabilityEvidence(
        generative_equation_id="2plm_lsirm_eq_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
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
    assert candidates[0].canonical_id.startswith(
        "new_binary_kernel_logistic__lsirm_jeon_et_al_2021_extension__spec_sha256_"
    )
    assert candidates[1].canonical_id.startswith(
        "new_binary_kernel_logistic__mlsirm_kang_jeon_2025_extension__spec_sha256_"
    )
    assert candidates[2].canonical_id.startswith(
        "new_binary_kernel_logistic__dlsjm_jin_jeon_2019_extension__spec_sha256_"
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


def test_candidate_identity_covers_dimensional_and_mixed_structure() -> None:
    baseline = _base_spec()
    more_dimensions = _base_spec(dimensions=3)
    different_fixed_effects = _base_spec(fixed_effects=("person_covariates",))
    different_membership = _base_spec(membership="single")

    identities = {
        _lsirm_id(baseline),
        _lsirm_id(more_dimensions),
        _lsirm_id(different_fixed_effects),
        _lsirm_id(different_membership),
    }

    assert len(identities) == 4


def test_candidate_identity_is_typed_and_serializes_stably() -> None:
    base = _base_spec()
    first = compile_dependence_candidates(base)[0]
    second = compile_dependence_candidates(base)[0]

    assert isinstance(first.identity, CandidateIdentity)
    assert first.identity == second.identity
    assert first.identity.to_manifest() == second.identity.to_manifest()
    assert first.identity.canonical_json() == second.identity.canonical_json()
    assert first.canonical_id == second.canonical_id
    assert first.to_manifest()["identity"] == first.identity.to_manifest()


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
    restricted = replace(
        base,
        response_kernel=replace(
            base.response_kernel,
            compatible_dependence=frozenset({DependenceKind.LSIRM}),
        ),
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
    base = _base_spec()
    candidate_id = _lsirm_id(base)
    base_ready_only = replace(
        base,
        estimation_plan=replace(base.estimation_plan, implemented=True),
        identification_contract=replace(base.identification_contract, verified=True),
        recovery_contract=replace(base.recovery_contract, passing=True),
    )

    candidate = compile_dependence_candidates(
        base_ready_only,
        evidence_by_candidate_id={candidate_id: _complete_lsirm_evidence()},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == (
        "rust_estimator_required",
        "identification_evidence_required",
        "passing_recovery_evidence_required",
    )


def test_supported_requires_exact_candidate_scoped_evidence() -> None:
    base = _base_spec()
    candidate_id = _lsirm_id(base)
    ready = _ready_for_candidate(base, candidate_id)
    evidence = _complete_lsirm_evidence()

    supported = compile_dependence_candidates(
        ready,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0]

    assert supported.status is CapabilityStatus.SUPPORTED
    assert supported.missing_requirements == ()
    assert supported.generative_equation_id == "2plm_lsirm_eq_v1"
    assert supported.primary_citations == ("10.1007/s11336-021-09762-5",)
    assert supported.estimation_plan.applies_to_candidate_id == candidate_id
    assert supported.identification_contract.applies_to_candidate_id == candidate_id
    assert supported.recovery_contract.applies_to_candidate_id == candidate_id

    not_implemented = replace(
        ready,
        estimation_plan=replace(ready.estimation_plan, implemented=False),
    )
    assert compile_dependence_candidates(
        not_implemented,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("rust_estimator_required",)

    wrong_backend = replace(
        ready,
        estimation_plan=replace(ready.estimation_plan, computational_backend="numpy"),
    )
    assert compile_dependence_candidates(
        wrong_backend,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("rust_estimator_required",)

    unidentified = replace(
        ready,
        identification_contract=replace(ready.identification_contract, verified=False),
    )
    assert compile_dependence_candidates(
        unidentified,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("identification_evidence_required",)

    unrecovered = replace(
        ready,
        recovery_contract=replace(ready.recovery_contract, passing=False),
    )
    assert compile_dependence_candidates(
        unrecovered,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("passing_recovery_evidence_required",)


def test_evidence_for_one_full_candidate_cannot_promote_another() -> None:
    baseline = _base_spec()
    baseline_id = _lsirm_id(baseline)
    changed = _base_spec(dimensions=3)
    changed_id = _lsirm_id(changed)
    changed_ready = _ready_for_candidate(changed, changed_id)

    candidate = compile_dependence_candidates(
        changed_ready,
        evidence_by_candidate_id={baseline_id: _complete_lsirm_evidence()},
    )[0]

    assert baseline_id != changed_id
    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == (
        "generative_equation_required",
        "primary_citation_required",
    )


def test_scope_mismatch_fails_each_support_gate_independently() -> None:
    base = _base_spec()
    candidate_id = _lsirm_id(base)
    ready = _ready_for_candidate(base, candidate_id)
    evidence = _complete_lsirm_evidence()

    wrong_estimator_scope = replace(
        ready,
        estimation_plan=replace(
            ready.estimation_plan,
            applies_to_candidate_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_estimator_scope,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("rust_estimator_required",)

    wrong_identification_scope = replace(
        ready,
        identification_contract=replace(
            ready.identification_contract,
            applies_to_candidate_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_identification_scope,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("identification_evidence_required",)

    wrong_recovery_scope = replace(
        ready,
        recovery_contract=replace(
            ready.recovery_contract,
            applies_to_candidate_id="2plm_logistic",
        ),
    )
    assert compile_dependence_candidates(
        wrong_recovery_scope,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("passing_recovery_evidence_required",)


def test_blank_equation_or_citation_cannot_promote_candidate() -> None:
    base = _base_spec()
    candidate_id = _lsirm_id(base)
    ready = _ready_for_candidate(base, candidate_id)
    blank = CapabilityEvidence(generative_equation_id="", primary_citations=("",))

    candidate = compile_dependence_candidates(
        ready,
        evidence_by_candidate_id={candidate_id: blank},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert candidate.missing_requirements == (
        "generative_equation_required",
        "primary_citation_required",
    )


def test_structural_collections_are_snapshotted_before_candidate_identity() -> None:
    parameter_blocks = ["discrimination", "difficulty"]
    compatible_dependence = [DependenceKind.LSIRM]
    fixed_effects = ["person_covariates"]
    random_effects = ["group_intercept"]

    base = ModelSpecification(
        response_kernel=ResponseKernel(
            family_id="2plm",
            formulation_id="2plm_logistic",
            response_scale="dichotomous",
            parameter_blocks=parameter_blocks,
            compatible_dependence=compatible_dependence,
        ),
        dimensional_structure=DimensionalStructure("confirmatory", 2),
        mixed_structure=GeneralizedMixedStructure(
            "explanatory",
            fixed_effects=fixed_effects,
            random_effects=random_effects,
            membership="single",
        ),
        estimation_plan=EstimationPlan("research_mmle", "rust", False, "base"),
        identification_contract=IdentificationContract(("trait_scale",), False, "base"),
        recovery_contract=RecoveryContract(("rmse",), False, "base"),
    )
    candidate = compile_dependence_candidates(base)[0]
    candidate_id = candidate.canonical_id

    parameter_blocks.append("caller_mutation")
    compatible_dependence.append(DependenceKind.DLSJM)
    fixed_effects.append("caller_mutation")
    random_effects.clear()

    assert base.response_kernel.parameter_blocks == ("discrimination", "difficulty")
    assert base.response_kernel.compatible_dependence == frozenset({DependenceKind.LSIRM})
    assert base.mixed_structure.fixed_effects == ("person_covariates",)
    assert base.mixed_structure.random_effects == ("group_intercept",)
    assert candidate.canonical_id == candidate_id
    assert candidate.identity.canonical_id() == candidate_id


def test_supported_status_requires_substantive_support_owner_records() -> None:
    base = _base_spec()
    candidate_id = _lsirm_id(base)
    ready = _ready_for_candidate(base, candidate_id)
    evidence = _complete_lsirm_evidence()

    missing_estimator_identity = replace(
        ready,
        estimation_plan=replace(ready.estimation_plan, estimator_id=""),
    )
    assert compile_dependence_candidates(
        missing_estimator_identity,
        evidence_by_candidate_id={candidate_id: evidence},
    )[0].missing_requirements == ("rust_estimator_required",)

    for rules in ((), ("",)):
        missing_identification = replace(
            ready,
            identification_contract=replace(ready.identification_contract, rules=rules),
        )
        assert compile_dependence_candidates(
            missing_identification,
            evidence_by_candidate_id={candidate_id: evidence},
        )[0].missing_requirements == ("identification_evidence_required",)

    for metrics in ((), ("",)):
        missing_recovery = replace(
            ready,
            recovery_contract=replace(
                ready.recovery_contract,
                required_metrics=metrics,
            ),
        )
        assert compile_dependence_candidates(
            missing_recovery,
            evidence_by_candidate_id={candidate_id: evidence},
        )[0].missing_requirements == ("passing_recovery_evidence_required",)
