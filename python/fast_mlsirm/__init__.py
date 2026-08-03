"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from . import _legacy_init as _legacy_init

# Copy the established package-root attributes that actually exist. Importing
# with ``from ._legacy_init import *`` would dereference every historical
# ``__all__`` entry and can break package import when a legacy export is stale;
# this preserves the previous runtime surface without turning that unrelated
# registry issue into a blocker for the modular bifactor API.
for _public_name, _public_value in vars(_legacy_init).items():
    if not _public_name.startswith("_") or _public_name == "__version__":
        globals()[_public_name] = _public_value

# Repair the one historical export declared by the legacy registry but omitted
# from its import list, so ``from fast_mlsirm import *`` remains valid after the
# modular package-surface split.
from .classification import ClassificationResult as ClassificationResult
from .bifactor_scoreability import (
    BifactorScoreabilityResult as BifactorScoreabilityResult,
    bifactor_scoreability as bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes as bifactor_scoreability_from_logit_slopes,
)

__all__ = [
    *_legacy_init.__all__,
    "BifactorScoreabilityResult",
    "bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes",
]

del _public_name, _public_value
