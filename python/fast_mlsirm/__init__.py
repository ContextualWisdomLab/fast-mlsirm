"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init

from .bifactor_scoreability import (
    BifactorScoreabilityResult as BifactorScoreabilityResult,
    bifactor_scoreability as bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes as bifactor_scoreability_from_logit_slopes,
)
from .rotation import (
    RotationCriterionInfo as RotationCriterionInfo,
    RotationSolution as RotationSolution,
    available_rotation_criteria as available_rotation_criteria,
    rotate_factor_loadings as rotate_factor_loadings,
    rotation_criterion_value_gradient as rotation_criterion_value_gradient,
)

# Copy only declared legacy exports that are currently defined. This preserves
# the established package surface without leaking helper imports from the
# compatibility module or making import depend on unrelated stale names.
for _public_name in _legacy_init.__all__:
    if hasattr(_legacy_init, _public_name):
        globals()[_public_name] = getattr(_legacy_init, _public_name)


# Resolve distribution metadata at the package boundary on every reload. The
# compatibility module may remain cached, so its copied value is not sufficient
# for reliable source-checkout fallback behavior.
try:
    __version__ = _distribution_version("fast-mlsirm")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = list(_legacy_init.__all__) + [
    "BifactorScoreabilityResult",
    "bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes",
    "RotationCriterionInfo",
    "RotationSolution",
    "available_rotation_criteria",
    "rotate_factor_loadings",
    "rotation_criterion_value_gradient",
]

del _PackageNotFoundError, _distribution_version, _public_name
