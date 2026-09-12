"""Callback-free container admission for dynamic evaluation contracts."""

from __future__ import annotations

from typing import Any

import pytest

from fast_mlsirm.measurement import dynamic_evaluation as de


class _CallbackList(list[Any]):
    """Expose any attempt to execute a caller-defined container protocol."""

    def __len__(self) -> int:
        """Fail if package admission asks a foreign container for cardinality."""
        raise AssertionError("caller-defined __len__ must not execute")

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Fail if package admission iterates a foreign container."""
        raise AssertionError("caller-defined __iter__ must not execute")



def _category() -> de.EvaluationCategoryDefinition:
    """Build one valid category using only exact built-in carriers."""
    return de.build_evaluation_category_definition(
        category_ref="category_supported",
        definition_ref="category_supported_definition_1",
        definition_sha256="a" * 64,
        order_index=0,
    )



def _criterion() -> de.EvaluationCriterionDefinition:
    """Build one valid criterion using only exact built-in carriers."""
    return de.build_evaluation_criterion_definition(
        criterion_ref="criterion_accuracy",
        criterion_revision_ref="criterion_accuracy_revision_1",
        definition_ref="criterion_accuracy_definition_1",
        definition_sha256="b" * 64,
        admissible_evidence_rule_ref="criterion_accuracy_admissible_1",
        admissible_evidence_rule_sha256="c" * 64,
        exclusion_rule_ref="criterion_accuracy_exclusion_1",
        exclusion_rule_sha256="d" * 64,
        response_semantics_ref="criterion_accuracy_response_1",
        response_semantics_sha256="e" * 64,
        abstention_rule_ref="criterion_accuracy_abstention_1",
        abstention_rule_sha256="f" * 64,
        not_observable_rule_ref="criterion_accuracy_not_observable_1",
        not_observable_rule_sha256="1" * 64,
        category_definitions=(_category(),),
    )



def _criterion_set() -> de.EvaluationCriterionSetSnapshot:
    """Build one valid criterion set using only exact built-in carriers."""
    return de.build_evaluation_criterion_set_snapshot(
        criterion_set_snapshot_ref="criterion_set_snapshot_1",
        criterion_set_revision_ref="criterion_set_revision_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        rubric_revision_ref="rubric_revision_1",
        intended_use_ref="intended_use_1",
        construct_ref="construct_1",
        population_scope_ref="population_scope_1",
        language_scope_ref="language_scope_1",
        domain_scope_ref="domain_scope_1",
        criteria=(_criterion(),),
    )



def _item() -> de.DynamicEvaluationItemSnapshot:
    """Build one valid item using only exact built-in carriers."""
    return de.build_dynamic_evaluation_item(
        item_instance_ref="evaluation_item_1",
        blueprint_revision_ref="evaluation_blueprint_revision_1",
        content_ref="content_1",
        content_sha256="2" * 64,
        origin=de.DynamicItemOrigin.GENERATED,
        role=de.EvaluationItemRole.CANDIDATE,
        reference_semantics=de.ReferenceSemantics.RUBRIC,
        reference_status=de.ReferenceStatus.PROVISIONAL,
        rubric_revision_ref="rubric_revision_1",
        criterion_set_snapshot=_criterion_set(),
        criterion_refs=("criterion_accuracy",),
        provenance_refs=("source_snapshot_1",),
        generation_invocation_ref="generation_invocation_1",
        regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
    )



def test_reference_collections_reject_subclasses_without_callbacks() -> None:
    """Reference collection admission rejects foreign list protocols immediately."""
    hostile = _CallbackList(["criterion_accuracy"])

    with pytest.raises(TypeError, match="criterion_refs must be a tuple or list"):
        de.build_dynamic_evaluation_item(
            item_instance_ref="evaluation_item_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            content_ref="content_1",
            content_sha256="2" * 64,
            origin=de.DynamicItemOrigin.GENERATED,
            role=de.EvaluationItemRole.CANDIDATE,
            reference_semantics=de.ReferenceSemantics.RUBRIC,
            reference_status=de.ReferenceStatus.PROVISIONAL,
            rubric_revision_ref="rubric_revision_1",
            criterion_set_snapshot=_criterion_set(),
            criterion_refs=hostile,
            provenance_refs=("source_snapshot_1",),
            generation_invocation_ref="generation_invocation_1",
            regeneration_status=de.RegenerationStatus.INPUTS_RECORDED,
        )



def test_category_collection_rejects_subclasses_without_callbacks() -> None:
    """Criterion admission rejects foreign category-container protocols immediately."""
    hostile = _CallbackList([_category()])

    with pytest.raises(TypeError, match="category_definitions"):
        de.build_evaluation_criterion_definition(
            criterion_ref="criterion_accuracy",
            criterion_revision_ref="criterion_accuracy_revision_1",
            definition_ref="criterion_accuracy_definition_1",
            definition_sha256="b" * 64,
            admissible_evidence_rule_ref="criterion_accuracy_admissible_1",
            admissible_evidence_rule_sha256="c" * 64,
            exclusion_rule_ref="criterion_accuracy_exclusion_1",
            exclusion_rule_sha256="d" * 64,
            response_semantics_ref="criterion_accuracy_response_1",
            response_semantics_sha256="e" * 64,
            abstention_rule_ref="criterion_accuracy_abstention_1",
            abstention_rule_sha256="f" * 64,
            not_observable_rule_ref="criterion_accuracy_not_observable_1",
            not_observable_rule_sha256="1" * 64,
            category_definitions=hostile,
        )



def test_criterion_collection_rejects_subclasses_without_callbacks() -> None:
    """Criterion-set admission rejects foreign list protocols immediately."""
    hostile = _CallbackList([_criterion()])

    with pytest.raises(TypeError, match="criteria"):
        de.build_evaluation_criterion_set_snapshot(
            criterion_set_snapshot_ref="criterion_set_snapshot_1",
            criterion_set_revision_ref="criterion_set_revision_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            rubric_revision_ref="rubric_revision_1",
            intended_use_ref="intended_use_1",
            construct_ref="construct_1",
            population_scope_ref="population_scope_1",
            language_scope_ref="language_scope_1",
            domain_scope_ref="domain_scope_1",
            criteria=hostile,
        )



def test_item_collection_rejects_subclasses_without_callbacks() -> None:
    """Run admission rejects foreign item-container protocols immediately."""
    hostile = _CallbackList([_item()])

    with pytest.raises(TypeError, match="items"):
        de.build_evaluation_item_set_snapshot(
            run_snapshot_ref="run_snapshot_1",
            blueprint_revision_ref="evaluation_blueprint_revision_1",
            items=hostile,
            criterion_set_snapshot=_criterion_set(),
            linking_status=de.LinkingStatus.UNAVAILABLE,
        )
