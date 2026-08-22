"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init
from . import reliability as _reliability
from . import scaling as _scaling
from . import serving as _serving
from . import validation as _validation
from ._fleiss_control_safety import install as _install_fleiss_control_safety
from ._icc_control_safety import install as _install_icc_control_safety
from ._scaling_control_safety import install as _install_scaling_control_safety
from ._serving_export_safety import install as _install_serving_export_safety

# Harden historical public adapters before copying legacy exports. These
# wrappers validate and normalize semantic controls only; result arithmetic
# remains in the existing Rust-backed implementations.
_install_icc_control_safety(_reliability)
_install_scaling_control_safety(_scaling)
_install_fleiss_control_safety(_validation)
_install_serving_export_safety(_serving)
_legacy_init.icc = _reliability.icc
_legacy_init.bradley_terry_mm = _scaling.bradley_terry_mm
_legacy_init.fleiss_kappa = _validation.fleiss_kappa
_legacy_init.export_serving_bundle = _serving.export_serving_bundle

del (
    _install_fleiss_control_safety,
    _install_icc_control_safety,
    _install_scaling_control_safety,
    _install_serving_export_safety,
    _reliability,
    _scaling,
    _serving,
)

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
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1 as CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge as ContextualOrchestratorJudge,
    JudgeCriterion as JudgeCriterion,
    JudgeFormatError as JudgeFormatError,
    LLMJudgeResult as LLMJudgeResult,
    MAX_BINARY_THRESHOLD_CALLS as MAX_BINARY_THRESHOLD_CALLS,
    MAX_JUDGE_CATEGORIES as MAX_JUDGE_CATEGORIES,
)
from .judge_calibration import (
    CALIBRATION_VARIANTS as CALIBRATION_VARIANTS,
    CONTAMINATION_STATUSES as CONTAMINATION_STATUSES,
    JudgeCalibrationCase as JudgeCalibrationCase,
    JudgeCalibrationOutcome as JudgeCalibrationOutcome,
    JudgeCalibrationReport as JudgeCalibrationReport,
    build_multiple_choice_calibration_cases as build_multiple_choice_calibration_cases,
    evaluate_paired_calibration as evaluate_paired_calibration,
)
from .irt_contract import (
    IRTItemType as IRTItemType,
    MIN_IRT_ITEMS as MIN_IRT_ITEMS,
    MIN_IRT_PERSONS as MIN_IRT_PERSONS,
    MIN_OBSERVED_PER_ITEM as MIN_OBSERVED_PER_ITEM,
    MIN_ITEM_DISTINCT_VALUES as MIN_ITEM_DISTINCT_VALUES,
    MIN_FACTOR_ANCHOR_ITEMS as MIN_FACTOR_ANCHOR_ITEMS,
    fit_irt_experiment as fit_irt_experiment,
    validate_irt_response_matrix as validate_irt_response_matrix,
    validate_irt_experiment_readiness as validate_irt_experiment_readiness,
)

# Bind the rating-range API on the historical validation namespace without
# duplicating implementation or arithmetic ownership.
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
    "CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1",
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
    "MAX_BINARY_THRESHOLD_CALLS",
    "MAX_JUDGE_CATEGORIES",
    "CALIBRATION_VARIANTS",
    "CONTAMINATION_STATUSES",
    "JudgeCalibrationCase",
    "JudgeCalibrationOutcome",
    "JudgeCalibrationReport",
    "build_multiple_choice_calibration_cases",
    "evaluate_paired_calibration",
    "IRTItemType",
    "fit_irt_experiment",
    "MIN_IRT_ITEMS",
    "validate_irt_response_matrix",
    "MIN_IRT_PERSONS",
    "MIN_OBSERVED_PER_ITEM",
    "MIN_ITEM_DISTINCT_VALUES",
    "MIN_FACTOR_ANCHOR_ITEMS",
    "validate_irt_experiment_readiness",
    "RatingRangeEvidence",
    "paired_rating_range_evidence",
]

del _PackageNotFoundError, _distribution_version, _public_name
