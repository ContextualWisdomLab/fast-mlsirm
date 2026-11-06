"""Criterion semantics and exact item/run binding contracts."""

from __future__ import annotations

import pytest

from fast_mlsirm.measurement import dynamic_evaluation as de


def _category(ref: str, order: int) -> de.EvaluationCategoryDefinition:
    """Build one content-addressed response category."""
    return de.build_evaluation_category_definition(
        category_ref=ref,
        definition_ref=f"definition_{ref}",
        definition_sha256=("a" if order == 0 else "b") * 64,
        order_index=order,
    )


def _criterion() -> de.EvaluationCriterionDefinition:
    """Build one criterion with explicit evidence and response rules."""
    return de.build_evaluation_criterion_definition(
        criterion_ref="criterion_evidence_support",
        criterion_revision_ref="criterion_evidence_support_revision_1",
        definition_ref="criterion_definition_1",
        definition_sha256="c" * 64,
        admissible_evidence_rule_ref="admissible_evidence_rule_1",
        admissible_evidence_rule_sha256="d" * 64,
        exclusion_rule_ref="exclusion_rule_1",
        exclusion_rule_sha256="e" * 64,
        response_semantics_ref="response_semantics_1",
        response_semantics_sha256="f" * 64,
        abstention_rule_ref="abstention_rule_1",
        abstention_rule_sha256="1" * 64,
        not_observable_rule_ref="not_observable_rule_1",
        not_observable_rule_sha256="2" * 64,
        category_definitions=(
            _category("category_not_supported", 0),
            _category("category_supported", 1),
        ),
    )


def _criterion_set(
    *,
    snapshot_ref: str = "criterion_set_snapshot_1",
    revision_ref: str = "criterion_set_revision_1",
    blueprint_ref: str = "evaluation_blueprint_revision_1",
    rubric_ref: str = "rubric_revision_1",
) -> de.EvaluationCriterionSetSnapshot:
    """Freeze one explicit criterion set for an evaluation blueprint."""
    return de.build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref=snapshot_ref,
        criterion_set_revision_ref=revision_ref,
        blueprint_revision_ref=blueprint_ref,
        rubric_revision_ref=rubric_ref,
        intended_use_ref="intended_use_model_response_evaluation_1",
        construct_ref="construct_evidence_grounding_1",
        population_scope_ref="population_scope_enterprise_responses_1",
        language_scope_ref="language_scope_multilingual_1",
        domain_scope_ref="domain_scope_enterprise_qa_1",
        criteria=(_criterion(),),
    )


def _item(
    criterion_set: de.EvaluationCriterionSetSnapshot,
    *,
    criterion_refs: tuple[str, ...] = ("criterion_evidence_support",),
    rubric_ref: str = "rubric_revision_1",
) -> de.DynamicEvaluationItemSnapshot:
    """Build one candidate item under an exact criterion-set digest."""
    return de.build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_alpha",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_alpha",
        content_sha256="3" * 64,
        origin=de.DynamicItemOrigin.GENERATED,
        role=de.EvaluationItemRole.CANDIDATE,
        reference_semantics=de.ReferenceSemantics.RUBRIC,
        reference_status=de.ReferenceStatus.PROVISIONAL,
        rubric_revision_ref=rubric_ref,
        criterion_set_snapshot=criterion_set,
        criterion_refs=criterion_refs,
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
    )


def test_criterion_requires_digest_bound_meaning_rules_and_categories() -> None:
    """Every judgment rule and response category is content-addressed."""
    criterion = _criterion()
    payload = criterion.to_dict()
    assert payload["admissible_evidence_rule_sha256"] == "d" * 64
    assert payload["exclusion_rule_sha256"] == "e" * 64
    assert payload["response_semantics_sha256"] == "f" * 64
    assert payload["abstention_rule_sha256"] == "1" * 64
    assert payload["not_observable_rule_sha256"] == "2" * 64
    assert [row["category_ref"] for row in payload["category_definitions"]] == [
        "category_not_supported",
        "category_supported",
    ]


def test_run_and_item_require_an_explicit_nonempty_criterion_set() -> None:
    """Neither a dynamic item nor a run is evaluable without frozen criteria."""
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_alpha",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_alpha",
            content_sha256="3" * 64,
            origin=de.DynamicItemOrigin.GENERATED,
            role=de.EvaluationItemRole.CANDIDATE,
            reference_semantics=de.ReferenceSemantics.RUBRIC,
            reference_status=de.ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_set_snapshot=None,
            criterion_refs=("criterion_evidence_support",),
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
        )
    assert caught.value.code == "criterion_set_required"

    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_set_snapshot(
            criterion_set_snapshot_ref="criterion_set_snapshot_empty",
            criterion_set_revision_ref="criterion_set_revision_empty",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            rubric_revision_ref="rubric_revision_1",
            intended_use_ref="intended_use_model_response_evaluation_1",
            construct_ref="construct_evidence_grounding_1",
            population_scope_ref="population_scope_enterprise_responses_1",
            language_scope_ref="language_scope_multilingual_1",
            domain_scope_ref="domain_scope_enterprise_qa_1",
            criteria=(),
        )
    assert caught.value.code == "invalid_criterion_set"

    criterion_set = _criterion_set()
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_missing_criteria",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(criterion_set),),
            criterion_set_snapshot=None,
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_required"


def test_item_cannot_reference_unknown_criterion_or_foreign_rubric() -> None:
    """Item admission rejects invented criteria and rubric substitution."""
    criterion_set = _criterion_set()
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        _item(criterion_set, criterion_refs=("criterion_unknown",))
    assert caught.value.code == "item_criterion_not_registered"

    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        _item(criterion_set, rubric_ref="rubric_revision_2")
    assert caught.value.code == "criterion_set_rubric_mismatch"


def test_item_and_run_bind_the_same_criterion_identity_and_digest() -> None:
    """A run rejects items created under another criterion-set revision."""
    criterion_set = _criterion_set()
    item = _item(criterion_set)
    assert item.criterion_set_snapshot_ref == (
        criterion_set.criterion_set_snapshot_ref
    )
    assert item.criterion_set_sha256 == criterion_set.snapshot_sha256

    foreign_set = _criterion_set(
        snapshot_ref="criterion_set_snapshot_2",
        revision_ref="criterion_set_revision_2",
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="run_snapshot_foreign_criteria",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(item,),
            criterion_set_snapshot=foreign_set,
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "item_criterion_set_mismatch"


def test_run_publishes_exact_criterion_identity_digest_and_rules() -> None:
    """Every run discloses the exact criteria and rules actually administered."""
    criterion_set = _criterion_set()
    snapshot = de.build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(criterion_set),),
        criterion_set_snapshot=criterion_set,
        linking_status=de.LinkingStatus.WITHIN_RUN_ONLY,
    )

    payload = snapshot.to_dict()
    assert payload["criterion_set_snapshot_ref"] == (
        criterion_set.criterion_set_snapshot_ref
    )
    assert payload["criterion_set_sha256"] == criterion_set.snapshot_sha256
    assert payload["criterion_refs"] == ["criterion_evidence_support"]
    criterion_payload = payload["criterion_set"]["criteria"][0]
    assert criterion_payload["admissible_evidence_rule_ref"]
    assert criterion_payload["admissible_evidence_rule_sha256"]
    assert criterion_payload["abstention_rule_ref"]
    assert criterion_payload["category_definitions"]


def test_category_mutation_invalidates_criterion_set_and_run_admission() -> None:
    """Changing category meaning invalidates the criterion-set digest."""
    criterion_set = _criterion_set()
    category = criterion_set.criteria[0].category_definitions[0]
    object.__setattr__(category, "definition_ref", "mutated_definition")

    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        criterion_set.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"

    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_mutated_criteria",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(),
            criterion_set_snapshot=criterion_set,
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_integrity_mismatch"


def test_run_requires_every_admitted_criterion_to_be_administered() -> None:
    """A run cannot silently omit a criterion declared by its criterion set."""
    second = de.build_evaluation_criterion_definition(
        criterion_ref="criterion_safety",
        criterion_revision_ref="criterion_safety_revision_1",
        definition_ref="criterion_safety_definition_1",
        definition_sha256="4" * 64,
        admissible_evidence_rule_ref="criterion_safety_admissible_rule_1",
        admissible_evidence_rule_sha256="5" * 64,
        exclusion_rule_ref="criterion_safety_exclusion_rule_1",
        exclusion_rule_sha256="6" * 64,
        response_semantics_ref="criterion_safety_response_semantics_1",
        response_semantics_sha256="7" * 64,
        abstention_rule_ref="criterion_safety_abstention_rule_1",
        abstention_rule_sha256="8" * 64,
        not_observable_rule_ref="criterion_safety_not_observable_rule_1",
        not_observable_rule_sha256="9" * 64,
        category_definitions=(
            de.build_evaluation_category_definition(
                category_ref="criterion_safety_not_satisfied",
                definition_ref="criterion_safety_not_satisfied_definition_1",
                definition_sha256="a" * 64,
                order_index=0,
            ),
            de.build_evaluation_category_definition(
                category_ref="criterion_safety_satisfied",
                definition_ref="criterion_safety_satisfied_definition_1",
                definition_sha256="b" * 64,
                order_index=1,
            ),
        ),
    )
    criterion_set = de.build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_two_criteria",
        criterion_set_revision_ref="criterion_set_revision_two_criteria",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_model_response_evaluation_1",
        construct_ref="construct_evidence_grounding_1",
        population_scope_ref="population_scope_enterprise_responses_1",
        language_scope_ref="language_scope_multilingual_1",
        domain_scope_ref="domain_scope_enterprise_qa_1",
        criteria=(_criterion(), second),
    )
    item = _item(
        criterion_set,
        criterion_refs=("criterion_evidence_support",),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_partial_criteria",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(item,),
            criterion_set_snapshot=criterion_set,
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_not_covered"
