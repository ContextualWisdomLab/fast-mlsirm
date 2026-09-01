from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

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


def _membership(
    *,
    classification: MembershipClassification = MembershipClassification.CROSS_CLASSIFIED,
    multiplicity: MembershipMultiplicity = MembershipMultiplicity.MULTIPLE,
    weight_authority: MembershipWeightAuthority = MembershipWeightAuthority.EXPLICIT_NORMALIZED,
    weight_recovery_metric: str | None = None,
) -> MembershipStructure:
    return MembershipStructure(
        classification=classification,
        multiplicity=multiplicity,
        weight_authority=weight_authority,
        classification_axes=("organization", "project"),
        weight_recovery_metric=weight_recovery_metric,
    )


def _base(membership: MembershipStructure) -> ModelSpecification:
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
        dimensional_structure=DimensionalStructure("confirmatory", 2),
        mixed_structure=GeneralizedMixedStructure(
            formulation_id="explanatory_cross_classified",
            fixed_effects=("person_covariates",),
            random_effects=("organization_intercept", "project_intercept"),
            membership=membership,
        ),
        estimation_plan=EstimationPlan("research_mmle", "rust", False, "base"),
        identification_contract=IdentificationContract(("trait_scale",), False, "base"),
        recovery_contract=RecoveryContract(("rmse",), False, "base"),
    )


def _ready(base: ModelSpecification, candidate_id: str) -> ModelSpecification:
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


def _evidence() -> CapabilityEvidence:
    return CapabilityEvidence(
        generative_equation_id="2plm_lsirm_membership_v1",
        primary_citations=("10.1007/s11336-021-09762-5",),
    )


def test_free_form_membership_string_is_rejected() -> None:
    with pytest.raises(TypeError, match="membership must be a MembershipStructure"):
        GeneralizedMixedStructure(
            formulation_id="explanatory",
            membership="multiple_membership",  # type: ignore[arg-type]
        )


def test_cross_classification_and_multiple_membership_are_orthogonal() -> None:
    cross_classified_single = _membership(
        multiplicity=MembershipMultiplicity.SINGLE,
        weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
    )
    cross_classified_multiple = _membership()

    first = compile_dependence_candidates(_base(cross_classified_single))[0]
    second = compile_dependence_candidates(_base(cross_classified_multiple))[0]

    assert first.canonical_id != second.canonical_id
    assert first.identity.membership_classification is MembershipClassification.CROSS_CLASSIFIED
    assert first.identity.membership_multiplicity is MembershipMultiplicity.SINGLE
    assert second.identity.membership_classification is MembershipClassification.CROSS_CLASSIFIED
    assert second.identity.membership_multiplicity is MembershipMultiplicity.MULTIPLE


def test_multiple_membership_requires_explicit_weight_authority() -> None:
    with pytest.raises(ValueError, match="multiple membership requires weight authority"):
        _membership(weight_authority=MembershipWeightAuthority.NOT_APPLICABLE)

    with pytest.raises(ValueError, match="single membership cannot have membership weights"):
        _membership(
            multiplicity=MembershipMultiplicity.SINGLE,
            weight_authority=MembershipWeightAuthority.EXPLICIT_NORMALIZED,
        )


def test_cross_classification_requires_two_named_axes() -> None:
    with pytest.raises(
        ValueError,
        match="cross-classified membership requires at least two distinct axes",
    ):
        MembershipStructure(
            classification=MembershipClassification.CROSS_CLASSIFIED,
            multiplicity=MembershipMultiplicity.SINGLE,
            weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
            classification_axes=("organization",),
        )


def test_model_estimated_weights_require_named_recovery_metric() -> None:
    with pytest.raises(ValueError, match="model-estimated membership weights require a recovery metric"):
        _membership(
            weight_authority=MembershipWeightAuthority.MODEL_ESTIMATED,
            weight_recovery_metric=None,
        )

    model_estimated = _membership(
        weight_authority=MembershipWeightAuthority.MODEL_ESTIMATED,
        weight_recovery_metric="membership_weight_rmse",
    )
    base = _base(model_estimated)
    candidate_id = compile_dependence_candidates(base)[0].canonical_id
    ready_without_weight_recovery = _ready(base, candidate_id)
    candidate = compile_dependence_candidates(
        ready_without_weight_recovery,
        evidence_by_candidate_id={candidate_id: _evidence()},
    )[0]

    assert candidate.status is CapabilityStatus.RESEARCH_CANDIDATE
    assert "membership_weight_recovery_required" in candidate.missing_requirements

    recovered = replace(
        ready_without_weight_recovery,
        recovery_contract=replace(
            ready_without_weight_recovery.recovery_contract,
            required_metrics=("rmse", "membership_weight_rmse"),
        ),
    )
    supported = compile_dependence_candidates(
        recovered,
        evidence_by_candidate_id={candidate_id: _evidence()},
    )[0]
    assert supported.status is CapabilityStatus.SUPPORTED


def test_manifest_publishes_typed_membership_and_self_digest() -> None:
    candidate = compile_dependence_candidates(_base(_membership()))[0]
    manifest = candidate.to_manifest()

    assert manifest["manifest_schema_id"] == "fast_mlsirm.model_specification.candidate_manifest"
    assert manifest["manifest_schema_version"] == "1.0.0"
    membership = manifest["mixed_structure"]["membership"]
    assert membership == {
        "classification": "cross_classified",
        "multiplicity": "multiple",
        "weight_authority": "explicit_normalized",
        "classification_axes": ["organization", "project"],
        "weight_recovery_metric": None,
    }

    digest = manifest.pop("manifest_sha256")
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert digest == hashlib.sha256(canonical).hexdigest()
