"""Enterprise audit and strict-schema contract for rubric item authoring."""

from __future__ import annotations

import re

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    compile_item_blueprints,
)


def _levels() -> tuple[RubricLevel, ...]:
    """Return a valid three-level ordinal scale."""
    return (
        RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
        RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
        RubricLevel(2, "full_support", "Full support.", ("complete support",)),
    )


def _rubric(response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING) -> RubricSpecification:
    """Return a versioned enterprise rubric fixture."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=response_format,
        levels=_levels(),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        locale="en-US",
        rubric_version="1.2.3",
    )


def _blueprint(response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING):
    """Compile one deterministic blueprint."""
    return compile_item_blueprints(
        _rubric(response_format),
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=20260803,
        ),
    )[0]


def test_rubric_revision_is_distinct_from_serialization_schema_version():
    """Human governance revision and wire-schema evolution are independently traceable."""
    rubric = _rubric()
    assert rubric.schema_version == "1.0"
    assert rubric.rubric_version == "1.2.3"
    assert rubric.to_dict()["rubric_version"] == "1.2.3"


@pytest.mark.parametrize("version", ["1", "1.2", "v1.2.3", "01.2.3", "1.02.3"])
def test_rubric_revision_requires_canonical_semantic_version(version):
    """Rubric revisions reject ambiguous or non-canonical version strings."""
    with pytest.raises(ValueError, match="rubric_version.*semantic version"):
        RubricSpecification(
            rubric_id="faithfulness_rubric",
            construct_id="evidence_grounding",
            construct_definition="Degree to which claims are supported.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("claim_verification",),
            evidence_requirements=("Quote the supporting source span.",),
            rubric_version=version,
        )


def test_blueprint_exposes_full_sha256_fingerprint_in_addition_to_short_id():
    """A display id never replaces the full audit identity."""
    blueprint = _blueprint()
    assert re.fullmatch(r"[0-9a-f]{64}", blueprint.blueprint_fingerprint)
    assert blueprint.to_dict()["blueprint_fingerprint"] == blueprint.blueprint_fingerprint
    assert blueprint.blueprint_id == f"item_blueprint_{blueprint.blueprint_fingerprint[:16]}"


def test_generation_contract_exposes_full_sha256_fingerprint():
    """Generation provenance remains collision-resistant at the contract layer."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    contract = build_generation_contract(rubric, blueprint)
    assert re.fullmatch(r"[0-9a-f]{64}", contract["contract_fingerprint"])
    assert contract["contract_id"] == (
        f"generation_contract_{contract['contract_fingerprint'][:16]}"
    )
    assert contract["blueprint"]["blueprint_fingerprint"] == (
        blueprint.blueprint_fingerprint
    )


def test_structured_output_schema_declares_json_schema_draft_2020_12():
    """Provider adapters can select the intended validation dialect unambiguously."""
    contract = build_generation_contract(_rubric(), _blueprint())
    assert contract["output_schema"]["$schema"] == (
        "https://json-schema.org/draft/2020-12/schema"
    )


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        (
            ResponseFormat.CONSTRUCTED_RESPONSE,
            {
                "oneOf": [
                    {"type": "null"},
                    {"type": "string", "minLength": 1},
                ]
            },
        ),
        (
            ResponseFormat.SELECTED_RESPONSE,
            {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        ),
        (ResponseFormat.BINARY_JUDGMENT, {"type": "boolean"}),
        (
            ResponseFormat.ORDINAL_RATING,
            {"type": "integer", "enum": [0, 1, 2]},
        ),
        (
            ResponseFormat.PAIRWISE_COMPARISON,
            {
                "type": "string",
                "enum": [
                    "left_candidate",
                    "right_candidate",
                    "tie",
                    "insufficient_evidence",
                ],
            },
        ),
    ],
)
def test_answer_key_schema_is_closed_for_each_response_format(
    response_format,
    expected,
):
    """The strict output claim contains no unconstrained answer-key escape hatch."""
    rubric = _rubric(response_format)
    blueprint = _blueprint(response_format)
    contract = build_generation_contract(rubric, blueprint)
    assert contract["output_schema"]["properties"]["answer_key"] == expected
