"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init

# Copy established package-root attributes that actually exist. Importing with
# ``from ._legacy_init import *`` dereferences every historical ``__all__``
# entry and can make package import depend on an unrelated stale registry item.
for _public_name, _public_value in vars(_legacy_init).items():
    if not _public_name.startswith("_") or _public_name == "__version__":
        globals()[_public_name] = _public_value

# Repair the historical export declared by the legacy registry but omitted from
# its import list, then compose both modular compiled feature surfaces.
from .classification import ClassificationResult as ClassificationResult
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

# Resolve distribution metadata at the package boundary on every reload. The
# legacy module may remain cached, so copying its prior value alone would make
# the documented source-checkout fallback impossible to exercise reliably.
try:
    __version__ = _distribution_version("fast-mlsirm")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    *_legacy_init.__all__,
    "BifactorScoreabilityResult",
    "bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes",
    "RotationCriterionInfo",
    "RotationSolution",
    "available_rotation_criteria",
    "rotate_factor_loadings",
    "rotation_criterion_value_gradient",
]

del _PackageNotFoundError, _distribution_version, _public_name, _public_value
