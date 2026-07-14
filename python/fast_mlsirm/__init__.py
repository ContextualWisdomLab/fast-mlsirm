from importlib.metadata import PackageNotFoundError, version

from .config import FitConfig as FitConfig, MLS2PLMConfig as MLS2PLMConfig, PenaltyConfig as PenaltyConfig
from .diagnostics import align_latent_space as align_latent_space, dimensionality_diagnostics as dimensionality_diagnostics, fit_diagnostics as fit_diagnostics, fixed_item_calibration_diagnostics as fixed_item_calibration_diagnostics, predict_proba as predict_proba, recovery_report as recovery_report, response_process_dimensionality_diagnostics as response_process_dimensionality_diagnostics, response_process_fit_diagnostics as response_process_fit_diagnostics
from .fit import fit as fit
from .fitstats import (adjusted_chi2_pairs as adjusted_chi2_pairs,
                       benjamini_hochberg as benjamini_hochberg, chi2_sf as chi2_sf,
                       dif_analysis as dif_analysis,
                       dimensionality_residuals as dimensionality_residuals,
                       infit_outfit as infit_outfit, person_fit as person_fit,
                       person_fit_resampling as person_fit_resampling,
                       residual_item_fit as residual_item_fit,
                       s_x2 as s_x2, select_items as select_items,
                       tcc_drift as tcc_drift,
                       vuong_nonnested as vuong_nonnested)
from .inference import oakes_standard_errors as oakes_standard_errors, observed_information as observed_information, second_order_test as second_order_test, standard_errors_from_vcov as standard_errors_from_vcov, vcov_from_hessian as vcov_from_hessian
from .linking import link_fixed_item_parameters as link_fixed_item_parameters
from .report import render_diagnostics_report as render_diagnostics_report
from .validation import (ValidationVerdict as ValidationVerdict,
                         validate_judge as validate_judge)
from .serving import (bank_information as bank_information,
                      cat_next_item as cat_next_item,
                      export_serving_bundle as export_serving_bundle,
                      plausible_values as plausible_values,
                      load_serving_bundle as load_serving_bundle,
                      score_respondents as score_respondents)
from .preprocessing import irtree_expand as irtree_expand
from .simulation import simulate as simulate
from .test_design import assemble_test_form as assemble_test_form, item_information as item_information, select_cat_item as select_cat_item
from .types import DimensionalityDiagnostics as DimensionalityDiagnostics, FitDiagnostics as FitDiagnostics, FitResult as FitResult, MLSIRMParams as MLSIRMParams, RecoveryReport as RecoveryReport, SimulationData as SimulationData

try:
    __version__ = version("fast-mlsirm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "DimensionalityDiagnostics",
    "FitConfig",
    "FitDiagnostics",
    "FitResult",
    "MLS2PLMConfig",
    "MLSIRMParams",
    "PenaltyConfig",
    "RecoveryReport",
    "SimulationData",
    "align_latent_space",
    "assemble_test_form",
    "dimensionality_diagnostics",
    "ValidationVerdict",
    "benjamini_hochberg",
    "chi2_sf",
    "dif_analysis",
    "dimensionality_residuals",
    "irtree_expand",
    "oakes_standard_errors",
    "validate_judge",
    "vuong_nonnested",
    "adjusted_chi2_pairs",
    "bank_information",
    "cat_next_item",
    "person_fit_resampling",
    "plausible_values",
    "residual_item_fit",
    "tcc_drift",
    "export_serving_bundle",
    "fit",
    "fit_diagnostics",
    "infit_outfit",
    "load_serving_bundle",
    "person_fit",
    "s_x2",
    "score_respondents",
    "select_items",
    "fixed_item_calibration_diagnostics",
    "item_information",
    "link_fixed_item_parameters",
    "observed_information",
    "predict_proba",
    "recovery_report",
    "response_process_dimensionality_diagnostics",
    "response_process_fit_diagnostics",
    "render_diagnostics_report",
    "second_order_test",
    "select_cat_item",
    "simulate",
    "standard_errors_from_vcov",
    "vcov_from_hessian",
]
