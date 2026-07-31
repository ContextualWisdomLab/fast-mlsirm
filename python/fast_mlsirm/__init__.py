from importlib.metadata import PackageNotFoundError, version

from .config import FitConfig as FitConfig, MLS2PLMConfig as MLS2PLMConfig, PenaltyConfig as PenaltyConfig
from .diagnostics import align_latent_space as align_latent_space, dimensionality_diagnostics as dimensionality_diagnostics, fit_diagnostics as fit_diagnostics, fixed_item_calibration_diagnostics as fixed_item_calibration_diagnostics, predict_proba as predict_proba, recovery_report as recovery_report, response_process_dimensionality_diagnostics as response_process_dimensionality_diagnostics, response_process_fit_diagnostics as response_process_fit_diagnostics
from .fit import fit as fit
from .inference import observed_information as observed_information, second_order_test as second_order_test, standard_errors_from_vcov as standard_errors_from_vcov, vcov_from_hessian as vcov_from_hessian
from .linking import link_fixed_item_parameters as link_fixed_item_parameters
from .report import render_diagnostics_report as render_diagnostics_report
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
    "fit",
    "fit_diagnostics",
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
