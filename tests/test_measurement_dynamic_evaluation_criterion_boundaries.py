"""Integrity and resource boundaries for criterion-bound evaluation."""

from __future__ import annotations

import pytest

from fast_mlsirm.measurement import dynamic_evaluation as de


def _category(
    *,
    category_ref: str,
    order_index: int | None,
    marker: str,
) -> de.EvaluationCategoryDefinition:
    """Build one content-addressed category for boundary tests."""
    return de.build_evaluation_category_definition(
        category_ref=category_ref,
        definition_ref=f"definition_{category_ref}",
        definition_sha256=marker * 64,
        order_index=order_index,
    )


def _criterion(
    *,
    criterion_ref: str = "criterion_evidence_support",
    criterion_revision_ref: str = "criterion_evidence_support_revision_1",
    marker: str = "c",
    categories: tuple[de.EvaluationCategoryDefinition, ...] | None = None,
) -> de.EvaluationCriterionDefinition:
    """Build one explicit criterion with content-addressed judgment rules."""
    if categories is None:
        categories = (
            _category(
                category_ref=f"{criterion_ref}_not_satisfied",
                order_index=0,
                marker="6",
            ),
            _category(
                category_ref=f"{criterion_ref}_satisfied",
                order_index=1,
                marker="7",
            ),
        )
    return de.build_evaluation_criterion_definition(
        criterion_ref=criterion_ref,
        criterion_revision_ref=criterion_revision_ref,
        definition_ref=f"definition_{criterion_ref}",
        definition_sha256=marker * 64,
        admissible_evidence_rule_ref=f"admissible_{criterion_ref}",
        admissible_evidence_rule_sha256="1" * 64,
        exclusion_rule_ref=f"exclusion_{criterion_ref}",
        exclusion_rule_sha256="2" * 64,
        response_semantics_ref=f"semantics_{criterion_ref}",
        response_semantics_sha256="3" * 64,
        abstention_rule_ref=f"abstention_{criterion_ref}",
        abstention_rule_sha256="4" * 64,
        not_observable_rule_ref=f"not_observable_{criterion_ref}",
        not_observable_rule_sha256="5" * 64,
        category_definitions=categories,
    )


def _criterion_set() -> de.EvaluationCriterionSetSnapshot:
    """Build one criterion set for run-integrity tests."""
    return de.build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_model_response_evaluation_1",
        construct_ref="construct_evidence_grounding_1",
        population_scope_ref="population_scope_enterprise_responses_1",
        language_scope_ref="language_scope_multilingual_1",
        domain_scope_ref="domain_scope_enterprise_qa_1",
        criteria=(_criterion(),),
    )


def _item() -> de.DynamicEvaluationItemSnapshot:
    """Build one item bound to the fixture criterion-set digest."""
    criterion_set = _criterion_set()
    return de.build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_alpha",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_alpha",
        content_sha256="a" * 64,
        origin=de.DynamicItemOrigin.GENERATED,
        role=de.EvaluationItemRole.CANDIDATE,
        reference_semantics=de.ReferenceSemantics.RUBRIC,
        reference_status=de.ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=criterion_set,
        criterion_refs=("criterion_evidence_support",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
    )


def _criterion_set_kwargs() -> dict[str, object]:
    """Return reusable criterion-set arguments for boundary mutations."""
    return {
        "criterion_set_snapshot_ref": "criterion_set_snapshot_1",
        "criterion_set_revision_ref": "criterion_set_revision_1",
        "blueprint_revision_ref": "evaluation_blueprint_revision_1",
        "rubric_revision_ref": "rubric_revision_1",
        "intended_use_ref": "intended_use_model_response_evaluation_1",
        "construct_ref": "construct_evidence_grounding_1",
        "population_scope_ref": "population_scope_enterprise_responses_1",
        "language_scope_ref": "language_scope_multilingual_1",
        "domain_scope_ref": "domain_scope_enterprise_qa_1",
    }


def _criterion_direct_kwargs() -> dict[str, object]:
    """Return a complete direct-construction payload for sealing tests."""
    categories = (
        _category(category_ref="category_no", order_index=0, marker="6"),
        _category(category_ref="category_yes", order_index=1, marker="7"),
    )
    return {
        "criterion_ref": "criterion_evidence_support",
        "criterion_revision_ref": "criterion_revision_1",
        "definition_ref": "definition_1",
        "definition_sha256": "c" * 64,
        "admissible_evidence_rule_ref": "admissible_1",
        "admissible_evidence_rule_sha256": "1" * 64,
        "exclusion_rule_ref": "exclusion_1",
        "exclusion_rule_sha256": "2" * 64,
        "response_semantics_ref": "semantics_1",
        "response_semantics_sha256": "3" * 64,
        "abstention_rule_ref": "abstention_1",
        "abstention_rule_sha256": "4" * 64,
        "not_observable_rule_ref": "not_observable_1",
        "not_observable_rule_sha256": "5" * 64,
        "category_definitions": categories,
    }


def test_category_and_criterion_factories_are_sealed_and_detect_mutation() -> None:
    """Meaning-bearing values cannot bypass admission or change afterward."""
    with pytest.raises(ValueError, match="build_evaluation_category_definition"):
        de.EvaluationCategoryDefinition(  # type: ignore[call-arg]
            category_ref="category_yes",
            definition_ref="category_definition_1",
            definition_sha256="a" * 64,
            order_index=0,
        )

    category = _category(category_ref="category_yes", order_index=0, marker="a")
    assert category.snapshot_sha256 == category.to_dict()["snapshot_sha256"]
    object.__setattr__(category, "definition_ref", "mutated_category")
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        category.to_dict()
    assert caught.value.code == "category_definition_integrity_mismatch"

    malformed_category = _category(
        category_ref="category_no", order_index=0, marker="b"
    )
    object.__setattr__(malformed_category, "definition_sha256", object())
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        malformed_category.to_dict()
    assert caught.value.code == "category_definition_integrity_mismatch"

    with pytest.raises(ValueError, match="build_evaluation_criterion_definition"):
        de.EvaluationCriterionDefinition(  # type: ignore[call-arg]
            **_criterion_direct_kwargs()
        )

    admitted = _criterion()
    assert admitted.snapshot_sha256 == admitted.to_dict()["snapshot_sha256"]
    object.__setattr__(admitted, "definition_ref", "changed_definition")
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        admitted.to_dict()
    assert caught.value.code == "criterion_definition_integrity_mismatch"

    malformed = _criterion(marker="d")
    object.__setattr__(malformed, "definition_sha256", object())
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        malformed.to_dict()
    assert caught.value.code == "criterion_definition_integrity_mismatch"

    with pytest.raises(ValueError, match="build_evaluation_criterion_set_snapshot"):
        de.EvaluationCriterionSetSnapshot(  # type: ignore[call-arg]
            **_criterion_set_kwargs(),
            criteria=(_criterion(),),
        )


def test_category_admission_rejects_order_and_collection_defects() -> None:
    """Category order, identity, and resource constraints fail closed."""
    with pytest.raises(TypeError, match="order_index"):
        de.build_evaluation_category_definition(
            category_ref="category_invalid",
            definition_ref="definition_invalid",
            definition_sha256="a" * 64,
            order_index=True,
        )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_category_definition(
            category_ref="category_invalid",
            definition_ref="definition_invalid",
            definition_sha256="a" * 64,
            order_index=-1,
        )
    assert caught.value.code == "invalid_category_order"

    base = _criterion_direct_kwargs()
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": ()}
        )
    assert caught.value.code == "invalid_category_set"

    too_many = tuple(
        _category(
            category_ref=f"category_{index}",
            order_index=None,
            marker=f"{index % 16:x}",
        )
        for index in range(65)
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": too_many}
        )
    assert caught.value.code == "category_set_budget_exceeded"

    with pytest.raises(TypeError, match="EvaluationCategoryDefinition"):
        de.build_evaluation_criterion_definition(
            **{
                **base,
                "category_definitions": (
                    _category(
                        category_ref="category_yes",
                        order_index=0,
                        marker="a",
                    ),
                    object(),
                ),
            }
        )

    duplicate_ref = (
        _category(category_ref="category_same", order_index=0, marker="a"),
        _category(category_ref="category_same", order_index=1, marker="b"),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": duplicate_ref}
        )
    assert caught.value.code == "duplicate_category_definition"

    partial_order = (
        _category(category_ref="category_one", order_index=0, marker="a"),
        _category(category_ref="category_two", order_index=None, marker="b"),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": partial_order}
        )
    assert caught.value.code == "partial_category_order"

    duplicate_order = (
        _category(category_ref="category_one", order_index=0, marker="a"),
        _category(category_ref="category_two", order_index=0, marker="b"),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": duplicate_order}
        )
    assert caught.value.code == "duplicate_category_order"

    non_contiguous = (
        _category(category_ref="category_one", order_index=0, marker="a"),
        _category(category_ref="category_two", order_index=2, marker="b"),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_definition(
            **{**base, "category_definitions": non_contiguous}
        )
    assert caught.value.code == "non_contiguous_category_order"

    unordered = (
        _category(category_ref="category_z", order_index=None, marker="a"),
        _category(category_ref="category_a", order_index=None, marker="b"),
    )
    criterion = de.build_evaluation_criterion_definition(
        **{**base, "category_definitions": unordered}
    )
    assert [row.category_ref for row in criterion.category_definitions] == [
        "category_a",
        "category_z",
    ]


def test_criterion_set_rejects_budget_type_and_duplicate_identity_defects() -> None:
    """Criterion-set resource and identity defects fail before run admission."""
    too_many = tuple(
        _criterion(
            criterion_ref=f"criterion_{index}",
            criterion_revision_ref=f"criterion_revision_{index}",
            marker=f"{index % 16:x}",
        )
        for index in range(129)
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=too_many,
        )
    assert caught.value.code == "criterion_set_budget_exceeded"

    with pytest.raises(TypeError, match="exact EvaluationCriterionDefinition"):
        de.build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(_criterion(), object()),  # type: ignore[arg-type]
        )

    duplicate_ref = _criterion(
        criterion_revision_ref="criterion_other_revision",
        marker="e",
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(_criterion(), duplicate_ref),
        )
    assert caught.value.code == "duplicate_criterion_definition"

    first = _criterion()
    duplicate_revision = _criterion(
        criterion_ref="criterion_safety",
        criterion_revision_ref=first.criterion_revision_ref,
        marker="f",
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_criterion_set_snapshot(
            **_criterion_set_kwargs(),
            criteria=(first, duplicate_revision),
        )
    assert caught.value.code == "duplicate_criterion_revision"


def test_nested_mutation_and_foreign_values_invalidate_composite_snapshots() -> None:
    """Nested or foreign domain values invalidate criterion and run digests."""
    criterion_set = _criterion_set()
    assert criterion_set.criterion_refs == ("criterion_evidence_support",)
    assert criterion_set.snapshot_sha256 == (
        criterion_set.to_dict()["snapshot_sha256"]
    )

    nested = criterion_set.criteria[0].category_definitions[0]
    object.__setattr__(nested, "definition_ref", "changed_definition")
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        criterion_set.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"

    foreign = _criterion_set()
    object.__setattr__(foreign, "criteria", (object(),))
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        foreign.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"

    wrong_set = de.build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_wrong_set",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=_criterion_set(),
        linking_status=de.LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(wrong_set, "criterion_set_snapshot", object())
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        wrong_set.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"

    wrong_items = de.build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_wrong_items",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=_criterion_set(),
        linking_status=de.LinkingStatus.UNAVAILABLE,
    )
    object.__setattr__(wrong_items, "items", (object(),))
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        wrong_items.to_dict()
    assert caught.value.code == "run_snapshot_integrity_mismatch"


def test_run_properties_preserve_binding_and_wrong_domain_types_fail() -> None:
    """Run accessors expose exact criteria and reject wrong domain values."""
    criterion_set = _criterion_set()
    snapshot = de.build_evaluation_item_set_snapshot(
        run_snapshot_ref="evaluation_run_snapshot_properties",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        items=(_item(),),
        criterion_set_snapshot=criterion_set,
        linking_status=de.LinkingStatus.UNAVAILABLE,
    )
    assert snapshot.criterion_set_snapshot_ref == (
        criterion_set.criterion_set_snapshot_ref
    )
    assert snapshot.criterion_set_sha256 == criterion_set.snapshot_sha256
    assert snapshot.criterion_refs == ("criterion_evidence_support",)
    assert snapshot.snapshot_sha256 == snapshot.to_dict()["snapshot_sha256"]

    with pytest.raises(TypeError, match="EvaluationCriterionSetSnapshot"):
        de.build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_wrong_set",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_wrong_set",
            content_sha256="a" * 64,
            origin=de.DynamicItemOrigin.GENERATED,
            role=de.EvaluationItemRole.CANDIDATE,
            reference_semantics=de.ReferenceSemantics.RUBRIC,
            reference_status=de.ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_set_snapshot=object(),  # type: ignore[arg-type]
            criterion_refs=("criterion_evidence_support",),
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
        )

    with pytest.raises(TypeError, match="EvaluationCriterionSetSnapshot"):
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_wrong_set_type",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(_item(),),
            criterion_set_snapshot=object(),  # type: ignore[arg-type]
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )


def test_nested_foreign_category_and_top_level_set_mutation_are_detected() -> None:
    """Foreign nested values and top-level set changes invalidate fingerprints."""
    criterion = _criterion()
    object.__setattr__(criterion, "category_definitions", (object(),))
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        criterion.to_dict()
    assert caught.value.code == "criterion_definition_integrity_mismatch"

    criterion_set = _criterion_set()
    object.__setattr__(criterion_set, "construct_ref", "construct_changed")
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        criterion_set.to_dict()
    assert caught.value.code == "criterion_set_integrity_mismatch"


def test_item_and_run_reject_foreign_blueprint_criterion_sets() -> None:
    """Blueprint identity must agree before criteria can govern an item or run."""
    foreign_set = de.build_evaluation_criterion_set_snapshot(
        **{
            **_criterion_set_kwargs(),
            "criterion_set_snapshot_ref": "criterion_set_snapshot_foreign",
            "criterion_set_revision_ref": "criterion_set_revision_foreign",
            "blueprint_revision_ref": "evaluation_blueprint_revision_2",
        },
        criteria=(_criterion(),),
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_foreign_blueprint",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_foreign_blueprint",
            content_sha256="a" * 64,
            origin=de.DynamicItemOrigin.GENERATED,
            role=de.EvaluationItemRole.CANDIDATE,
            reference_semantics=de.ReferenceSemantics.RUBRIC,
            reference_status=de.ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_set_snapshot=foreign_set,
            criterion_refs=("criterion_evidence_support",),
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
        )
    assert caught.value.code == "criterion_set_blueprint_mismatch"

    foreign_item = de.build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_foreign_blueprint",
        blueprint_revision_ref="evaluation_blueprint_revision_2",
        content_ref="content_foreign_blueprint",
        content_sha256="a" * 64,
        origin=de.DynamicItemOrigin.GENERATED,
        role=de.EvaluationItemRole.CANDIDATE,
        reference_semantics=de.ReferenceSemantics.RUBRIC,
        reference_status=de.ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=foreign_set,
        criterion_refs=("criterion_evidence_support",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
    )
    with pytest.raises(de.DynamicEvaluationContractError) as caught:
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="evaluation_run_snapshot_foreign_blueprint",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=(foreign_item,),
            criterion_set_snapshot=foreign_set,
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
    assert caught.value.code == "criterion_set_blueprint_mismatch"
