from __future__ import annotations

import pytest

from fast_mlsirm.model_specification import (
    CapabilityEvidence,
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


def _base_spec() -> ModelSpecification:
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
        ),
        estimation_plan=EstimationPlan(
            estimator_id="",
            computational_backend="",
            implemented=False,
            applies_to_candidate_id="",
        ),
        identification_contract=IdentificationContract(
            rules=(),
            verified=False,
            applies_to_candidate_id="",
        ),
        recovery_contract=RecoveryContract(
            required_metrics=(),
            passing=False,
            applies_to_candidate_id="",
        ),
    )


class _MutatingEvidenceMap(dict[str, CapabilityEvidence]):
    def __init__(self, base: ModelSpecification) -> None:
        super().__init__()
        self.base = base
        self.get_calls = 0

    def get(  # type: ignore[override]
        self,
        key: str,
        default: CapabilityEvidence | None = None,
    ) -> CapabilityEvidence | None:
        self.get_calls += 1
        object.__setattr__(self.base.dimensional_structure, "dimensions", 1)
        return default


def test_compiler_rejects_mapping_subclass_before_lookup_callback() -> None:
    """Evidence admission must not let a mapping callback split candidate identity."""
    base = _base_spec()
    evidence = _MutatingEvidenceMap(base)

    with pytest.raises(
        TypeError,
        match="evidence_by_candidate_id must be a built-in dict",
    ):
        compile_dependence_candidates(base, evidence_by_candidate_id=evidence)

    assert evidence.get_calls == 0
    assert base.dimensional_structure.dimensions == 2


def test_compiler_accepts_exact_builtin_evidence_dict() -> None:
    """Ordinary exact dictionaries remain the supported public evidence carrier."""
    candidates = compile_dependence_candidates(
        _base_spec(),
        evidence_by_candidate_id={},
    )

    assert len(candidates) == 3
