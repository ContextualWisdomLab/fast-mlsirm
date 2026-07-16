from importlib.metadata import PackageNotFoundError, version

from .config import FitConfig as FitConfig, MLS2PLMConfig as MLS2PLMConfig, PenaltyConfig as PenaltyConfig
from .diagnostics import align_latent_space as align_latent_space, dimensionality_diagnostics as dimensionality_diagnostics, fit_diagnostics as fit_diagnostics, fixed_item_calibration_diagnostics as fixed_item_calibration_diagnostics, predict_proba as predict_proba, recovery_report as recovery_report, response_process_dimensionality_diagnostics as response_process_dimensionality_diagnostics, response_process_fit_diagnostics as response_process_fit_diagnostics
from .fit import fit as fit
from .fitstats import (adjusted_chi2_pairs as adjusted_chi2_pairs,
                       benjamini_hochberg as benjamini_hochberg, chi2_sf as chi2_sf,
                       dif_analysis as dif_analysis,
                       dimensionality_residuals as dimensionality_residuals,
                       empirical_reliability as empirical_reliability,
                       infit_outfit as infit_outfit, person_fit as person_fit,
                       m2 as m2, m2_cmle_rasch as m2_cmle_rasch,
                       m2_multigroup as m2_multigroup,
                       m2_multilevel as m2_multilevel, M2Result as M2Result,
                       person_fit_resampling as person_fit_resampling,
                       residual_item_fit as residual_item_fit,
                       s_x2 as s_x2, select_items as select_items,
                       tcc_drift as tcc_drift,
                       vuong_nonnested as vuong_nonnested)
from .inference import oakes_standard_errors as oakes_standard_errors, observed_information as observed_information, second_order_test as second_order_test, standard_errors_from_vcov as standard_errors_from_vcov, vcov_from_hessian as vcov_from_hessian
from .linking import link_fixed_item_parameters as link_fixed_item_parameters
from .linking import irt_link as irt_link, IrtLinkResult as IrtLinkResult
from .equating import equate_observed_scores as equate_observed_scores, equate_neat as equate_neat, EquateResult as EquateResult, equate_observed_scores_kernel as equate_observed_scores_kernel, loglinear_smooth as loglinear_smooth, equate_neat_linear as equate_neat_linear, equating_standard_errors as equating_standard_errors
from .rt import fit_response_times as fit_response_times, RtFit as RtFit, fit_speed_accuracy as fit_speed_accuracy, rt_person_fit as rt_person_fit
from .cdm import fit_cdm as fit_cdm, CdmFit as CdmFit, fit_gdina as fit_gdina, GdinaFit as GdinaFit, validate_q_matrix as validate_q_matrix, QMatrixValidation as QMatrixValidation, gdina_wald_selection as gdina_wald_selection, WaldModelSelection as WaldModelSelection, fit_ho_cdm as fit_ho_cdm, HoCdmFit as HoCdmFit, fit_ho_gdina as fit_ho_gdina, HoGdinaFit as HoGdinaFit, fit_seq_gdina as fit_seq_gdina, SeqGdinaFit as SeqGdinaFit, fit_seq_gdina_qr as fit_seq_gdina_qr, SeqGdinaQrFit as SeqGdinaQrFit
from .mixture import fit_mixture as fit_mixture, MixtureFit as MixtureFit
from .crm import fit_crm as fit_crm, CrmFit as CrmFit
from . import models as models
from .models import ConfirmatoryModel as ConfirmatoryModel, ExploratoryModel as ExploratoryModel, IrtModel as IrtModel
from .twopl import fit_2pl as fit_2pl, TwoPlFit as TwoPlFit
from .nominal import fit_nominal as fit_nominal, NominalResponseFit as NominalResponseFit
from .grm import fit_grm as fit_grm, GrmFit as GrmFit
from .gpcm import fit_gpcm as fit_gpcm, GpcmFit as GpcmFit
from .rsm import fit_rsm as fit_rsm, RsmFit as RsmFit
from .mixed import fit_mixed_items as fit_mixed_items, MixedFormatFit as MixedFormatFit, MixedItemParameters as MixedItemParameters
from .lltm import fit_lltm as fit_lltm, LltmFit as LltmFit
from .testlet import fit_testlet as fit_testlet, TestletFit as TestletFit
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
from .polytomous import fit_polytomous as fit_polytomous, PolytomousFit as PolytomousFit, score_polytomous as score_polytomous, information_polytomous as information_polytomous, fit_lsirm_polytomous as fit_lsirm_polytomous, PolyLsirmFit as PolyLsirmFit, polytomous_information_criteria as polytomous_information_criteria, item_fit_polytomous as item_fit_polytomous, m2_polytomous as m2_polytomous, local_dependence_polytomous as local_dependence_polytomous, fit_nominal_polytomous as fit_nominal_polytomous, NominalFit as NominalFit, person_fit_polytomous as person_fit_polytomous, cat_simulate_polytomous as cat_simulate_polytomous, dif_polytomous as dif_polytomous, u3_person_fit_polytomous as u3_person_fit_polytomous, u3_cutoff_polytomous as u3_cutoff_polytomous
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
    "M2Result",
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
    "empirical_reliability",
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
    "irt_link",
    "IrtLinkResult",
    "equate_observed_scores",
    "equate_neat",
    "EquateResult",
    "equate_observed_scores_kernel",
    "loglinear_smooth",
    "equate_neat_linear",
    "equating_standard_errors",
    "fit_response_times",
    "RtFit",
    "fit_speed_accuracy",
    "rt_person_fit",
    "fit_cdm",
    "CdmFit",
    "fit_gdina",
    "GdinaFit",
    "validate_q_matrix",
    "QMatrixValidation",
    "gdina_wald_selection",
    "WaldModelSelection",
    "fit_ho_cdm",
    "HoCdmFit",
    "fit_ho_gdina",
    "HoGdinaFit",
    "fit_seq_gdina",
    "SeqGdinaFit",
    "fit_seq_gdina_qr",
    "SeqGdinaQrFit",
    "fit_mixture",
    "MixtureFit",
    "fit_crm",
    "CrmFit",
    "models",
    "ConfirmatoryModel",
    "ExploratoryModel",
    "IrtModel",
    "fit_2pl",
    "TwoPlFit",
    "fit_nominal",
    "NominalResponseFit",
    "fit_grm",
    "GrmFit",
    "fit_gpcm",
    "GpcmFit",
    "fit_rsm",
    "RsmFit",
    "fit_mixed_items",
    "MixedFormatFit",
    "MixedItemParameters",
    "fit_lltm",
    "LltmFit",
    "fit_testlet",
    "TestletFit",
    "export_serving_bundle",
    "fit",
    "fit_polytomous",
    "score_polytomous",
    "information_polytomous",
    "fit_lsirm_polytomous",
    "PolyLsirmFit",
    "polytomous_information_criteria",
    "item_fit_polytomous",
    "m2_polytomous",
    "local_dependence_polytomous",
    "fit_nominal_polytomous",
    "NominalFit",
    "person_fit_polytomous",
    "cat_simulate_polytomous",
    "dif_polytomous",
    "u3_person_fit_polytomous",
    "u3_cutoff_polytomous",
    "PolytomousFit",
    "fit_diagnostics",
    "infit_outfit",
    "m2",
    "m2_cmle_rasch",
    "m2_multigroup",
    "m2_multilevel",
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
