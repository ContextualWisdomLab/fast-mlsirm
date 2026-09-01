from __future__ import annotations

import json

import pytest

from fast_mlsirm.model_specification import (
    CapabilityEvidence,
    EstimationPlan,
    IdentificationContract,
    RecoveryContract,
)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: EstimationPlan(
                estimator_id=object(),  # type: ignore[arg-type]
                computational_backend="rust",
                implemented=False,
                applies_to_candidate_id="candidate",
            ),
            "estimator_id",
        ),
        (
            lambda: EstimationPlan(
                estimator_id="research_mmle",
                computational_backend=object(),  # type: ignore[arg-type]
                implemented=False,
                applies_to_candidate_id="candidate",
            ),
            "computational_backend",
        ),
        (
            lambda: EstimationPlan(
                estimator_id="research_mmle",
                computational_backend="rust",
                implemented=1,  # type: ignore[arg-type]
                applies_to_candidate_id="candidate",
            ),
            "implemented",
        ),
        (
            lambda: IdentificationContract(
                rules=(object(),),  # type: ignore[arg-type]
                verified=False,
                applies_to_candidate_id="candidate",
            ),
            "rules",
        ),
        (
            lambda: RecoveryContract(
                required_metrics=(object(),),  # type: ignore[arg-type]
                passing=False,
                applies_to_candidate_id="candidate",
            ),
            "required_metrics",
        ),
        (
            lambda: CapabilityEvidence(
                generative_equation_id=object(),  # type: ignore[arg-type]
                primary_citations=(),
            ),
            "generative_equation_id",
        ),
        (
            lambda: CapabilityEvidence(
                generative_equation_id=None,
                primary_citations=(object(),),  # type: ignore[arg-type]
            ),
            "primary_citations",
        ),
    ],
)
def test_manifest_metadata_rejects_non_json_domain_values(factory, field_name: str) -> None:
    """Domain evidence must not admit opaque objects into a JSON-shaped manifest."""
    with pytest.raises((TypeError, ValueError), match=field_name):
        factory()


def test_incomplete_research_metadata_remains_json_serializable() -> None:
    """Incomplete evidence uses typed empty values rather than opaque sentinels."""
    values = [
        EstimationPlan("", "", False, ""),
        IdentificationContract((), False, ""),
        RecoveryContract((), False, ""),
        CapabilityEvidence(None, ()),
    ]

    payload = [
        {
            key: value
            for key, value in vars(record).items()
        }
        for record in values
    ]
    json.dumps(payload, sort_keys=True)
