"""Versioned criterion-bound dynamic-evaluation contracts.

The package requires an immutable, content-addressed criterion set before an
item or run can be admitted. Concrete item sets may still be resolved
dynamically, and fixed anchors remain optional for cold-start pilot work.
"""

from ._common import (
    DYNAMIC_EVALUATION_ITEM_CONTRACT_ID,
    DynamicEvaluationContractError,
    DynamicItemOrigin,
    EvaluationItemRole,
    LinkingStatus,
    ReferenceSemantics,
    ReferenceStatus,
    RegenerationStatus,
)
from .criteria import (
    EvaluationCategoryDefinition,
    EvaluationCriterionDefinition,
    EvaluationCriterionSetSnapshot,
    build_evaluation_category_definition,
    build_evaluation_criterion_definition,
    build_evaluation_criterion_set_snapshot,
)
from .items import (
    DynamicEvaluationItemSnapshot,
    EvaluationItemSetSnapshot,
    build_dynamic_evaluation_item,
    build_evaluation_item_set_snapshot,
)

__all__ = [
    "DYNAMIC_EVALUATION_ITEM_CONTRACT_ID",
    "DynamicEvaluationContractError",
    "DynamicEvaluationItemSnapshot",
    "DynamicItemOrigin",
    "EvaluationCategoryDefinition",
    "EvaluationCriterionDefinition",
    "EvaluationCriterionSetSnapshot",
    "EvaluationItemRole",
    "EvaluationItemSetSnapshot",
    "LinkingStatus",
    "ReferenceSemantics",
    "ReferenceStatus",
    "RegenerationStatus",
    "build_dynamic_evaluation_item",
    "build_evaluation_category_definition",
    "build_evaluation_criterion_definition",
    "build_evaluation_criterion_set_snapshot",
    "build_evaluation_item_set_snapshot",
]
