"""Response-format-specific contracts for generated assessment items."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    compile_item_blueprints,
)


def _properties(response_format: ResponseFormat) -> dict:
    """Return output-schema properties for one response format."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported by evidence.",
        response_format=response_format,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
            RubricLevel(2, "full_support", "Full support.", ("complete support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote supporting evidence.",),
    )
    blueprint = compile_item_blueprints(rubric, BlueprintPlan())[0]
    return build_generation_contract(rubric, blueprint)["output_schema"]["properties"]


@pytest.mark.parametrize(
    ("response_format", "minimum", "maximum"),
    [
        (ResponseFormat.CONSTRUCTED_RESPONSE, 0, 0),
        (ResponseFormat.SELECTED_RESPONSE, 2, 32),
        (ResponseFormat.BINARY_JUDGMENT, 0, 0),
        (ResponseFormat.ORDINAL_RATING, 0, 0),
        (ResponseFormat.PAIRWISE_COMPARISON, 2, 2),
    ],
)
def test_option_cardinality_matches_response_format(
    response_format: ResponseFormat,
    minimum: int,
    maximum: int,
):
    """Each response format exposes only its valid option count."""
    options = _properties(response_format)["options"]
    assert options["minItems"] == minimum
    assert options["maxItems"] == maximum
    assert options["uniqueItems"] is True


def test_constructed_response_answer_key_is_bounded():
    """Constructed responses use a finite structured reference contract."""
    answer_key = _properties(ResponseFormat.CONSTRUCTED_RESPONSE)["answer_key"]
    assert answer_key["additionalProperties"] is False
    assert answer_key["required"] == [
        "reference_response",
        "accepted_variants",
        "rationale",
    ]
    assert answer_key["properties"]["reference_response"]["maxLength"] == 8_192
    assert answer_key["properties"]["accepted_variants"]["maxItems"] == 32


def test_choice_answer_keys_reference_option_ids():
    """Selected and pairwise items use explicit option identifiers."""
    selected = _properties(ResponseFormat.SELECTED_RESPONSE)["answer_key"]
    option_ids = selected["properties"]["option_ids"]
    assert option_ids["minItems"] == 1
    assert option_ids["maxItems"] == 32
    assert option_ids["uniqueItems"] is True

    pairwise = _properties(ResponseFormat.PAIRWISE_COMPARISON)["answer_key"]
    assert pairwise["required"] == ["preferred_option_id", "rationale"]
    assert pairwise["properties"]["preferred_option_id"]["maxLength"] == 128


def test_binary_and_ordinal_answer_keys_have_typed_values():
    """Judgment and rating keys cannot contain unrestricted JSON values."""
    binary = _properties(ResponseFormat.BINARY_JUDGMENT)["answer_key"]
    assert binary["properties"]["value"] == {"type": "boolean"}

    ordinal = _properties(ResponseFormat.ORDINAL_RATING)["answer_key"]
    assert ordinal["properties"]["score"] == {
        "type": "integer",
        "enum": [0, 1, 2],
    }


def test_all_nested_text_and_collections_are_bounded():
    """Nested provider fields retain explicit size limits."""
    properties = _properties(ResponseFormat.ORDINAL_RATING)
    assert properties["stimulus"]["items"]["minLength"] == 1
    assert properties["safety_notes"]["items"]["minLength"] == 1
    assert properties["answer_key"]["properties"]["rationale"]["maxLength"] == 8_192
