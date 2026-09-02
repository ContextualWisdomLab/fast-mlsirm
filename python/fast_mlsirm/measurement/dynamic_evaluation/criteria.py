"""Immutable evaluation criteria and criterion-set snapshots."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from ._common import (
    MAX_DYNAMIC_EVALUATION_CATEGORIES,
    MAX_DYNAMIC_EVALUATION_CRITERIA,
    _CATEGORY_TOKEN,
    _CRITERION_SET_TOKEN,
    _CRITERION_TOKEN,
    _error,
    _fingerprint,
    _reference,
    _sha256,
)


@dataclass(frozen=True)
class EvaluationCategoryDefinition:
    """Immutable meaning of one category admitted by a criterion."""

    category_ref: str
    definition_ref: str
    definition_sha256: str
    order_index: int | None
    _admission_token: InitVar[object | None] = None
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _admission_token: object | None) -> None:
        """Seal construction and record the admitted category fingerprint."""
        if _admission_token is not _CATEGORY_TOKEN:
            raise ValueError(
                "EvaluationCategoryDefinition must be created by "
                "build_evaluation_category_definition"
            )
        object.__setattr__(
            self, "_integrity_sha256", _fingerprint(self._payload_unchecked())
        )

    def _payload_unchecked(self) -> dict[str, Any]:
        """Return category identity and meaning without checking integrity."""
        return {
            "category_ref": self.category_ref,
            "definition_ref": self.definition_ref,
            "definition_sha256": self.definition_sha256,
            "order_index": self.order_index,
        }

    def _assert_integrity(self) -> None:
        """Reject object-level mutation after factory admission."""
        try:
            current = _fingerprint(self._payload_unchecked())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _error(
                "category_definition_integrity_mismatch",
                "$.category_definitions",
                "category definition changed after admission",
            ) from exc
        if current != self._integrity_sha256:
            raise _error(
                "category_definition_integrity_mismatch",
                "$.category_definitions",
                "category definition changed after admission",
            )

    @property
    def snapshot_sha256(self) -> str:
        """Return the deterministic immutable category fingerprint."""
        self._assert_integrity()
        return self._integrity_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return the source-text-free category binding."""
        self._assert_integrity()
        return {**self._payload_unchecked(), "snapshot_sha256": self._integrity_sha256}


def build_evaluation_category_definition(
    *,
    category_ref: str,
    definition_ref: str,
    definition_sha256: str,
    order_index: int | None = None,
) -> EvaluationCategoryDefinition:
    """Admit one category only when its meaning is content-addressed."""
    if order_index is not None and type(order_index) is not int:
        raise TypeError("$.order_index must be an integer or None")
    if order_index is not None and order_index < 0:
        raise _error(
            "invalid_category_order",
            "$.order_index",
            "category order must be a non-negative integer when supplied",
        )
    return EvaluationCategoryDefinition(
        category_ref=_reference(category_ref, "$.category_ref"),
        definition_ref=_reference(definition_ref, "$.definition_ref"),
        definition_sha256=_sha256(definition_sha256, "$.definition_sha256"),
        order_index=order_index,
        _admission_token=_CATEGORY_TOKEN,
    )


@dataclass(frozen=True)
class EvaluationCriterionDefinition:
    """Immutable binding to one explicit evaluation criterion and its rules."""

    criterion_ref: str
    criterion_revision_ref: str
    definition_ref: str
    definition_sha256: str
    admissible_evidence_rule_ref: str
    admissible_evidence_rule_sha256: str
    exclusion_rule_ref: str
    exclusion_rule_sha256: str
    response_semantics_ref: str
    response_semantics_sha256: str
    abstention_rule_ref: str
    abstention_rule_sha256: str
    not_observable_rule_ref: str
    not_observable_rule_sha256: str
    category_definitions: tuple[EvaluationCategoryDefinition, ...]
    _admission_token: InitVar[object | None] = None
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _admission_token: object | None) -> None:
        """Seal construction and record the admitted criterion fingerprint."""
        if _admission_token is not _CRITERION_TOKEN:
            raise ValueError(
                "EvaluationCriterionDefinition must be created by "
                "build_evaluation_criterion_definition"
            )
        object.__setattr__(
            self, "_integrity_sha256", _fingerprint(self._payload_unchecked())
        )

    def _payload_unchecked(self) -> dict[str, Any]:
        """Return the criterion payload without recursively checking integrity."""
        return {
            "criterion_ref": self.criterion_ref,
            "criterion_revision_ref": self.criterion_revision_ref,
            "definition_ref": self.definition_ref,
            "definition_sha256": self.definition_sha256,
            "admissible_evidence_rule_ref": self.admissible_evidence_rule_ref,
            "admissible_evidence_rule_sha256": self.admissible_evidence_rule_sha256,
            "exclusion_rule_ref": self.exclusion_rule_ref,
            "exclusion_rule_sha256": self.exclusion_rule_sha256,
            "response_semantics_ref": self.response_semantics_ref,
            "response_semantics_sha256": self.response_semantics_sha256,
            "abstention_rule_ref": self.abstention_rule_ref,
            "abstention_rule_sha256": self.abstention_rule_sha256,
            "not_observable_rule_ref": self.not_observable_rule_ref,
            "not_observable_rule_sha256": self.not_observable_rule_sha256,
            "category_definitions": [
                category.to_dict() for category in self.category_definitions
            ],
        }

    def _assert_integrity(self) -> None:
        """Reject object-level mutation after factory admission."""
        try:
            if any(
                type(category) is not EvaluationCategoryDefinition
                for category in self.category_definitions
            ):
                raise TypeError("criterion contains a foreign category definition")
            for category in self.category_definitions:
                category._assert_integrity()
            current = _fingerprint(self._payload_unchecked())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _error(
                "criterion_definition_integrity_mismatch",
                "$.criteria",
                "criterion definition changed after admission",
            ) from exc
        if current != self._integrity_sha256:
            raise _error(
                "criterion_definition_integrity_mismatch",
                "$.criteria",
                "criterion definition changed after admission",
            )

    @property
    def snapshot_sha256(self) -> str:
        """Return the deterministic immutable criterion fingerprint."""
        self._assert_integrity()
        return self._integrity_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return the source-text-free criterion binding."""
        self._assert_integrity()
        return {**self._payload_unchecked(), "snapshot_sha256": self._integrity_sha256}


def build_evaluation_criterion_definition(
    *,
    criterion_ref: str,
    criterion_revision_ref: str,
    definition_ref: str,
    definition_sha256: str,
    admissible_evidence_rule_ref: str,
    admissible_evidence_rule_sha256: str,
    exclusion_rule_ref: str,
    exclusion_rule_sha256: str,
    response_semantics_ref: str,
    response_semantics_sha256: str,
    abstention_rule_ref: str,
    abstention_rule_sha256: str,
    not_observable_rule_ref: str,
    not_observable_rule_sha256: str,
    category_definitions: (
        tuple[EvaluationCategoryDefinition, ...]
        | list[EvaluationCategoryDefinition]
    ),
) -> EvaluationCriterionDefinition:
    """Admit one criterion only when meaning and evidence rules are explicit."""
    if not isinstance(category_definitions, (tuple, list)) or not category_definitions:
        raise _error(
            "invalid_category_set",
            "$.category_definitions",
            "criterion must define at least one admissible response category",
        )
    if len(category_definitions) > MAX_DYNAMIC_EVALUATION_CATEGORIES:
        raise _error(
            "category_set_budget_exceeded",
            "$.category_definitions",
            "criterion may define at most "
            f"{MAX_DYNAMIC_EVALUATION_CATEGORIES} categories",
        )
    normalized_categories = tuple(category_definitions)
    if any(
        type(category) is not EvaluationCategoryDefinition
        for category in normalized_categories
    ):
        raise TypeError(
            "$.category_definitions must contain exact "
            "EvaluationCategoryDefinition values"
        )
    for category in normalized_categories:
        category._assert_integrity()
    category_refs = [category.category_ref for category in normalized_categories]
    if len(set(category_refs)) != len(category_refs):
        raise _error(
            "duplicate_category_definition",
            "$.category_definitions",
            "category identities must be unique within one criterion",
        )
    order_indexes = [
        category.order_index
        for category in normalized_categories
        if category.order_index is not None
    ]
    if order_indexes and len(order_indexes) != len(normalized_categories):
        raise _error(
            "partial_category_order",
            "$.category_definitions",
            "category ordering must be supplied for every category or none",
        )
    if len(set(order_indexes)) != len(order_indexes):
        raise _error(
            "duplicate_category_order",
            "$.category_definitions",
            "ordered response categories must have unique positions",
        )
    if order_indexes and set(order_indexes) != set(range(len(order_indexes))):
        raise _error(
            "non_contiguous_category_order",
            "$.category_definitions",
            "ordered response categories must use contiguous zero-based positions",
        )
    ordered_categories = (
        tuple(sorted(normalized_categories, key=lambda category: category.order_index))
        if order_indexes
        else tuple(
            sorted(normalized_categories, key=lambda category: category.category_ref)
        )
    )
    return EvaluationCriterionDefinition(
        criterion_ref=_reference(criterion_ref, "$.criterion_ref"),
        criterion_revision_ref=_reference(
            criterion_revision_ref, "$.criterion_revision_ref"
        ),
        definition_ref=_reference(definition_ref, "$.definition_ref"),
        definition_sha256=_sha256(definition_sha256, "$.definition_sha256"),
        admissible_evidence_rule_ref=_reference(
            admissible_evidence_rule_ref, "$.admissible_evidence_rule_ref"
        ),
        admissible_evidence_rule_sha256=_sha256(
            admissible_evidence_rule_sha256,
            "$.admissible_evidence_rule_sha256",
        ),
        exclusion_rule_ref=_reference(exclusion_rule_ref, "$.exclusion_rule_ref"),
        exclusion_rule_sha256=_sha256(
            exclusion_rule_sha256, "$.exclusion_rule_sha256"
        ),
        response_semantics_ref=_reference(
            response_semantics_ref, "$.response_semantics_ref"
        ),
        response_semantics_sha256=_sha256(
            response_semantics_sha256, "$.response_semantics_sha256"
        ),
        abstention_rule_ref=_reference(
            abstention_rule_ref, "$.abstention_rule_ref"
        ),
        abstention_rule_sha256=_sha256(
            abstention_rule_sha256, "$.abstention_rule_sha256"
        ),
        not_observable_rule_ref=_reference(
            not_observable_rule_ref, "$.not_observable_rule_ref"
        ),
        not_observable_rule_sha256=_sha256(
            not_observable_rule_sha256, "$.not_observable_rule_sha256"
        ),
        category_definitions=ordered_categories,
        _admission_token=_CRITERION_TOKEN,
    )


@dataclass(frozen=True)
class EvaluationCriterionSetSnapshot:
    """Immutable set of criteria that makes one evaluation run interpretable."""

    criterion_set_snapshot_ref: str
    criterion_set_revision_ref: str
    blueprint_revision_ref: str
    rubric_revision_ref: str
    intended_use_ref: str
    construct_ref: str
    population_scope_ref: str
    language_scope_ref: str
    domain_scope_ref: str
    criteria: tuple[EvaluationCriterionDefinition, ...]
    _admission_token: InitVar[object | None] = None
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _admission_token: object | None) -> None:
        """Seal construction and record the admitted criterion-set fingerprint."""
        if _admission_token is not _CRITERION_SET_TOKEN:
            raise ValueError(
                "EvaluationCriterionSetSnapshot must be created by "
                "build_evaluation_criterion_set_snapshot"
            )
        object.__setattr__(
            self, "_integrity_sha256", _fingerprint(self._payload_unchecked())
        )

    @property
    def criterion_refs(self) -> tuple[str, ...]:
        """Return the exact criteria available to the bound evaluator."""
        self._assert_integrity()
        return tuple(criterion.criterion_ref for criterion in self.criteria)

    def _payload_unchecked(self) -> dict[str, Any]:
        """Return the criterion-set payload without checking its own fingerprint."""
        return {
            "criterion_set_snapshot_ref": self.criterion_set_snapshot_ref,
            "criterion_set_revision_ref": self.criterion_set_revision_ref,
            "blueprint_revision_ref": self.blueprint_revision_ref,
            "rubric_revision_ref": self.rubric_revision_ref,
            "intended_use_ref": self.intended_use_ref,
            "construct_ref": self.construct_ref,
            "population_scope_ref": self.population_scope_ref,
            "language_scope_ref": self.language_scope_ref,
            "domain_scope_ref": self.domain_scope_ref,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }

    def _assert_integrity(self) -> None:
        """Reject definition or collection mutation after factory admission."""
        try:
            if any(
                type(criterion) is not EvaluationCriterionDefinition
                for criterion in self.criteria
            ):
                raise TypeError("criterion set contains a foreign domain value")
            for criterion in self.criteria:
                criterion._assert_integrity()
            current = _fingerprint(self._payload_unchecked())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _error(
                "criterion_set_integrity_mismatch",
                "$.criterion_set_snapshot",
                "criterion set changed after admission",
            ) from exc
        if current != self._integrity_sha256:
            raise _error(
                "criterion_set_integrity_mismatch",
                "$.criterion_set_snapshot",
                "criterion set changed after admission",
            )

    @property
    def snapshot_sha256(self) -> str:
        """Return the deterministic immutable criterion-set fingerprint."""
        self._assert_integrity()
        return self._integrity_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return the exact criterion snapshot required for evaluation."""
        self._assert_integrity()
        return {**self._payload_unchecked(), "snapshot_sha256": self._integrity_sha256}


def build_evaluation_criterion_set_snapshot(
    *,
    criterion_set_snapshot_ref: str,
    criterion_set_revision_ref: str,
    blueprint_revision_ref: str,
    rubric_revision_ref: str,
    intended_use_ref: str,
    construct_ref: str,
    population_scope_ref: str,
    language_scope_ref: str,
    domain_scope_ref: str,
    criteria: (
        tuple[EvaluationCriterionDefinition, ...]
        | list[EvaluationCriterionDefinition]
    ),
) -> EvaluationCriterionSetSnapshot:
    """Freeze a non-empty criterion set before any item response is evaluated."""
    if not isinstance(criteria, (tuple, list)) or not criteria:
        raise _error(
            "invalid_criterion_set",
            "$.criteria",
            "criterion set must contain at least one explicit criterion",
        )
    if len(criteria) > MAX_DYNAMIC_EVALUATION_CRITERIA:
        raise _error(
            "criterion_set_budget_exceeded",
            "$.criteria",
            "criterion set may contain at most "
            f"{MAX_DYNAMIC_EVALUATION_CRITERIA} criteria",
        )
    normalized = tuple(criteria)
    if any(type(item) is not EvaluationCriterionDefinition for item in normalized):
        raise TypeError(
            "$.criteria must contain exact EvaluationCriterionDefinition values"
        )
    for criterion in normalized:
        criterion._assert_integrity()
    ordered = tuple(sorted(normalized, key=lambda item: item.criterion_ref))
    refs = [criterion.criterion_ref for criterion in ordered]
    if len(set(refs)) != len(refs):
        raise _error(
            "duplicate_criterion_definition",
            "$.criteria",
            "criterion identities must be unique within one set",
        )
    revisions = [criterion.criterion_revision_ref for criterion in ordered]
    if len(set(revisions)) != len(revisions):
        raise _error(
            "duplicate_criterion_revision",
            "$.criteria",
            "criterion revision identities must be unique within one set",
        )
    return EvaluationCriterionSetSnapshot(
        criterion_set_snapshot_ref=_reference(
            criterion_set_snapshot_ref, "$.criterion_set_snapshot_ref"
        ),
        criterion_set_revision_ref=_reference(
            criterion_set_revision_ref, "$.criterion_set_revision_ref"
        ),
        blueprint_revision_ref=_reference(
            blueprint_revision_ref, "$.blueprint_revision_ref"
        ),
        rubric_revision_ref=_reference(rubric_revision_ref, "$.rubric_revision_ref"),
        intended_use_ref=_reference(intended_use_ref, "$.intended_use_ref"),
        construct_ref=_reference(construct_ref, "$.construct_ref"),
        population_scope_ref=_reference(
            population_scope_ref, "$.population_scope_ref"
        ),
        language_scope_ref=_reference(language_scope_ref, "$.language_scope_ref"),
        domain_scope_ref=_reference(domain_scope_ref, "$.domain_scope_ref"),
        criteria=ordered,
        _admission_token=_CRITERION_SET_TOKEN,
    )
