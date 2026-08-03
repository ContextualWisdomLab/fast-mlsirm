"""Public package surface for fast-mlsirm."""

from ._legacy_init import *  # noqa: F403
from ._legacy_init import __all__ as _legacy_all
from .rotation import (
    RotationCriterionInfo as RotationCriterionInfo,
    RotationSolution as RotationSolution,
    available_rotation_criteria as available_rotation_criteria,
    rotate_factor_loadings as rotate_factor_loadings,
    rotation_criterion_value_gradient as rotation_criterion_value_gradient,
)

__all__ = [
    *_legacy_all,
    "RotationCriterionInfo",
    "RotationSolution",
    "available_rotation_criteria",
    "rotate_factor_loadings",
    "rotation_criterion_value_gradient",
]
