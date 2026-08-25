"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init
from . import exposure as _exposure
from . import inference as _inference
from . import reliability as _reliability
from . import scaling as _scaling
from . import validation as _validation
from ._exposure_array_safety import install as _install_exposure_array_safety
from ._exposure_flexilevel_safety import install as _install_exposure_flexilevel_safety
from ._fleiss_control_safety import install as _install_fleiss_control_safety
from ._fit_public import fit as _public_fit
from ._icc_control_safety import install as _install_icc_control_safety
from ._inference_admission_safety import install as _install_inference_admission_safety
from ._scaling_control_safety import install as _install_scaling_control_safety

# Harden historical public adapters before copying legacy exports. These
# wrappers validate and normalize semantic controls only; result arithmetic
# remains in the existing Rust-backed implementations.
_install_exposure_array_safety(_exposure)
_install_exposure_flexilevel_safety(_exposure)
_install_icc_control_safety(_reliability)
_install_inference_admission_safety(_inference)
_install_scaling_control_safety(_scaling)
_install_fleiss_control_safety(_validation)
_legacy_init.ccat_select = _exposure.ccat_select
_legacy_init.flexilevel_administer = _exposure.flexilevel_administer
_legacy_init.flexilevel_score_distribution = _exposure.flexilevel_score_distribution
_legacy_init.icc = _reliability.icc
_legacy_init.second_order_test = _inference.second_order_test
_legacy_init.vcov_from_hessian = _inference.vcov_from_hessian
_legacy_init.standard_errors_from_vcov = _inference.standard_errors_from_vcov
_legacy_init.bradley_terry_mm = _scaling.bradley_terry_mm
_legacy_init.bratt_mm = _scaling.bratt_mm
_legacy_init.fleiss_kappa = _validation.fleiss_kappa

del (
    _exposure,
    _inference,
    _install_exposure_array_safety,
    _install_exposure_flexilevel_safety,
    _install_fleiss_control_safety,
    _install_icc_control_safety,
    _install_inference_admission_safety,
    _install_scaling_control_safety,
    _reliability,
    _scaling,
)

# Copy only declared legacy exports that are currently defined. This preserves
# the established package surface without leaking helper imports from the
# compatibility module or making import depend on unrelated stale names.
for _public_name in _legacy_init.__all__:
    if hasattr(_legacy_init, _public_name):
        globals()[_public_name] = getattr(_legacy_init, _public_name)

# The legacy compatibility module imports the implementation-level ``fit``
# callable, which carries private reference-backend authority for
# ``fit_reference``. Rebind the package export after the legacy copy so callers
# can never acquire that authority from the public ``fast_mlsirm.fit`` API.
fit = _public_fit

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

del _PackageNotFoundError, _distribution_version, _public_fit, _public_name
