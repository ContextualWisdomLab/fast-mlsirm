"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init

# Copy only declared legacy exports that are currently defined. This avoids
# accidentally publishing helper imports from the compatibility module while
# preserving its established package-root surface.
for _public_name in _legacy_init.__all__:
    if hasattr(_legacy_init, _public_name):
        globals()[_public_name] = getattr(_legacy_init, _public_name)

from .bifactor_scoreability import (
    BifactorScoreabilityResult as BifactorScoreabilityResult,
    bifactor_scoreability as bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes as bifactor_scoreability_from_logit_slopes,
)

# Resolve distribution metadata at the package boundary on every reload. The
# legacy module may remain cached, so copying its prior value alone would make
# the documented source-checkout fallback impossible to exercise reliably.
try:
    __version__ = _distribution_version("fast-mlsirm")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = list(_legacy_init.__all__) + [
    "BifactorScoreabilityResult",
    "bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes",
]

del _PackageNotFoundError, _distribution_version, _public_name
