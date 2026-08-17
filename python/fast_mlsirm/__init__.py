"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init

# Copy only declared legacy exports that are currently defined. This preserves
# the established package surface without leaking helper imports from the
# compatibility module or making import depend on unrelated stale names.
for _public_name in _legacy_init.__all__:
    if hasattr(_legacy_init, _public_name):
        globals()[_public_name] = getattr(_legacy_init, _public_name)

from .bifactor_scoreability import (
    BifactorScoreabilityResult as BifactorScoreabilityResult,
    bifactor_scoreability as bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes as bifactor_scoreability_from_logit_slopes,
)
from .rating_range import (
    RatingRangeEvidence as RatingRangeEvidence,
    paired_rating_range_evidence as paired_rating_range_evidence,
)
from .rotation import (
    RotationCriterionInfo as RotationCriterionInfo,
    RotationSolution as RotationSolution,
    available_rotation_criteria as available_rotation_criteria,
    rotate_factor_loadings as rotate_factor_loadings,
    rotation_criterion_value_gradient as rotation_criterion_value_gradient,
)
from .llm_judge import (
    ContextualOrchestratorJudge as ContextualOrchestratorJudge,
    JudgeCriterion as JudgeCriterion,
    JudgeFormatError as JudgeFormatError,
    LLMJudgeResult as LLMJudgeResult,
    MAX_JUDGE_CATEGORIES as MAX_JUDGE_CATEGORIES,
)
from .irt_contract import (
    IRTItemType as IRTItemType,
    MIN_IRT_ITEMS as MIN_IRT_ITEMS,
    validate_irt_response_matrix as validate_irt_response_matrix,
)

# Bind the rating-range API on the historical validation namespace without
# duplicating implementation or arithmetic ownership.
from . import validation as _validation

_validation.RatingRangeEvidence = RatingRangeEvidence
_validation.paired_rating_range_evidence = paired_rating_range_evidence

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
    "ContextualOrchestratorJudge",
    "JudgeCriterion",
    "JudgeFormatError",
    "LLMJudgeResult",
    "MAX_JUDGE_CATEGORIES",
    "IRTItemType",
    "MIN_IRT_ITEMS",
    "validate_irt_response_matrix",
    "RatingRangeEvidence",
    "paired_rating_range_evidence",
]

del _PackageNotFoundError, _distribution_version, _public_name
