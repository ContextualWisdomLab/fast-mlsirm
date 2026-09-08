from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from fast_mlsirm.model_specification import (
    CandidateSupportRecords,
    DependenceKind,
    DimensionalStructure,
    EstimationPlan,
    GeneralizedMixedStructure,
    IdentificationContract,
    ModelSpecification,
    RecoveryContract,
    ResponseKernel,
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
        dimensional_structure=DimensionalStructure("confirmatory", 2),
        mixed_structure=GeneralizedMixedStructure("explanatory"),
        estimation_plan=EstimationPlan("research", "rust", False, "base"),
        identification_contract=IdentificationContract(("location_scale",), False, "base"),
        recovery_contract=RecoveryContract(("bias", "rmse"), False, "base"),
    )


def _support(candidate_id: str, estimator_id: str = "mmle") -> CandidateSupportRecords:
    return CandidateSupportRecords(
        EstimationPlan(estimator_id, "rust", True, candidate_id),
        IdentificationContract(("location_scale",), True, candidate_id),
        RecoveryContract(("bias", "rmse"), True, candidate_id),
    )


def test_candidate_support_records_require_one_exact_scope() -> None:
    """All three support value objects must name the same non-blank candidate."""
    with pytest.raises(TypeError, match="estimation_plan"):
        CandidateSupportRecords(  # type: ignore[arg-type]
            object(),
            IdentificationContract(("location_scale",), True, "candidate"),
            RecoveryContract(("bias",), True, "candidate"),
        )
    with pytest.raises(TypeError, match="identification_contract"):
        CandidateSupportRecords(  # type: ignore[arg-type]
            EstimationPlan("mmle", "rust", True, "candidate"),
            object(),
            RecoveryContract(("bias",), True, "candidate"),
        )
    with pytest.raises(TypeError, match="recovery_contract"):
        CandidateSupportRecords(  # type: ignore[arg-type]
            EstimationPlan("mmle", "rust", True, "candidate"),
            IdentificationContract(("location_scale",), True, "candidate"),
            object(),
        )
    with pytest.raises(ValueError, match="non-blank"):
        _support("")
    with pytest.raises(ValueError, match="identification support scope"):
        CandidateSupportRecords(
            EstimationPlan("mmle", "rust", True, "candidate-a"),
            IdentificationContract(("location_scale",), True, "candidate-b"),
            RecoveryContract(("bias",), True, "candidate-a"),
        )
    with pytest.raises(ValueError, match="recovery support scope"):
        CandidateSupportRecords(
            EstimationPlan("mmle", "rust", True, "candidate-a"),
            IdentificationContract(("location_scale",), True, "candidate-a"),
            RecoveryContract(("bias",), True, "candidate-b"),
        )


def test_support_catalog_is_snapshotted_and_replace_safe() -> None:
    """Caller mutation and dataclass replacement cannot rewrite admitted support."""
    base = _base_specification()
    support = _support("candidate-a")
    caller_owned = {"candidate-a": support}
    specification = replace(base, support_records_by_candidate_id=caller_owned)

    caller_owned.clear()
    assert specification.support_records_by_candidate_id == {"candidate-a": support}
    assert type(specification.support_records_by_candidate_id) is MappingProxyType

    replaced = replace(specification, dimensional_structure=DimensionalStructure("confirmatory", 3))
    assert replaced.support_records_by_candidate_id == {"candidate-a": support}
    assert type(replaced.support_records_by_candidate_id) is MappingProxyType


def test_support_catalog_rejects_callback_capable_or_malformed_inputs() -> None:
    """Admission fails closed for mapping subclasses, malformed keys, and wrong scopes."""
    class CallbackDict(dict[str, CandidateSupportRecords]):
        pass

    base = _base_specification()
    with pytest.raises(TypeError, match="built-in dict or sealed mapping"):
        replace(base, support_records_by_candidate_id=CallbackDict())
    with pytest.raises(TypeError, match="keys must be non-blank"):
        replace(base, support_records_by_candidate_id={"": _support("candidate-a")})
    with pytest.raises(TypeError, match="values must be CandidateSupportRecords"):
        replace(  # type: ignore[arg-type]
            base,
            support_records_by_candidate_id={"candidate-a": object()},
        )
    with pytest.raises(ValueError, match="key must match"):
        replace(
            base,
            support_records_by_candidate_id={"candidate-a": _support("candidate-b")},
        )


def test_support_catalog_rejects_conflicting_legacy_singleton_truth() -> None:
    """Migration cannot publish two different support records for one candidate."""
    candidate_id = "candidate-a"
    legacy = replace(
        _base_specification(),
        estimation_plan=EstimationPlan("legacy-mmle", "rust", True, candidate_id),
        identification_contract=IdentificationContract(
            ("location_scale",), True, candidate_id
        ),
        recovery_contract=RecoveryContract(("bias", "rmse"), True, candidate_id),
    )

    same = CandidateSupportRecords(
        legacy.estimation_plan,
        legacy.identification_contract,
        legacy.recovery_contract,
    )
    compatible = replace(legacy, support_records_by_candidate_id={candidate_id: same})
    assert compatible.support_records_by_candidate_id == {candidate_id: same}

    with pytest.raises(ValueError, match="cannot conflict"):
        replace(
            legacy,
            support_records_by_candidate_id={
                candidate_id: _support(candidate_id, estimator_id="different-mmle")
            },
        )
