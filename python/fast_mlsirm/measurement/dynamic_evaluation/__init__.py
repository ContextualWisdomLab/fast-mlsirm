"""Versioned criterion-bound dynamic-evaluation contracts.

The package requires an immutable, content-addressed criterion set before an
item or run can be admitted. Concrete item sets may still be resolved
dynamically, and fixed anchors remain optional for cold-start pilot work.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from . import criteria as _criteria
from . import items as _items
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
)
from .items import DynamicEvaluationItemSnapshot, EvaluationItemSetSnapshot

_MISSING = object()


def _exact_collection_guard(
    function: Callable[..., Any], argument_name: str, path: str
) -> Callable[..., Any]:
    """Reject foreign sequence subclasses before they can execute callbacks."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        """Apply callback-free exact-container admission before the factory."""
        value = kwargs.get(argument_name, _MISSING)
        if value is not _MISSING and type(value) not in (tuple, list):
            raise TypeError(f"{path} must be a tuple or list")
        return function(*args, **kwargs)

    return guarded


def _canonical_item_set_guard(function: Callable[..., Any]) -> Callable[..., Any]:
    """Canonicalize mathematical item sets after safe domain-value admission."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        """Sort exact admitted item values without executing caller protocols."""
        value = kwargs.get("items", _MISSING)
        if value is not _MISSING:
            if type(value) not in (tuple, list):
                raise TypeError("$.items must be a tuple or list")
            if all(type(item) is DynamicEvaluationItemSnapshot for item in value):
                for item in value:
                    item._assert_integrity()
                    if type(item.item_instance_ref) is not str:
                        raise TypeError("$.items contains a non-string item identity")
                kwargs = {
                    **kwargs,
                    "items": tuple(sorted(value, key=lambda item: item.item_instance_ref)),
                }
        return function(*args, **kwargs)

    return guarded


build_evaluation_criterion_definition = _exact_collection_guard(
    _criteria.build_evaluation_criterion_definition,
    "category_definitions",
    "$.category_definitions",
)
build_evaluation_criterion_set_snapshot = _exact_collection_guard(
    _criteria.build_evaluation_criterion_set_snapshot,
    "criteria",
    "$.criteria",
)
build_dynamic_evaluation_item = _items.build_dynamic_evaluation_item
build_evaluation_item_set_snapshot = _canonical_item_set_guard(
    _items.build_evaluation_item_set_snapshot
)

# Keep the named submodule entry points aligned with the package Public Binding.
# Importing a submodule still initializes this package first, so every supported
# import path receives the same callback-free collection guard.
_criteria.build_evaluation_criterion_definition = build_evaluation_criterion_definition
_criteria.build_evaluation_criterion_set_snapshot = build_evaluation_criterion_set_snapshot
_items.build_evaluation_item_set_snapshot = build_evaluation_item_set_snapshot

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
