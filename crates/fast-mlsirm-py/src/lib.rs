use std::collections::HashMap;

use mlsirm_core::agreement::{validate_scoring_with_thresholds as core_validate_scoring, ValidationThresholds};
use mlsirm_core::equating::{
    analytic_see as core_analytic_see, bootstrap_see as core_bootstrap_see,
    circle_arc_equate as core_circle_arc_equate,
    circle_arc_middle_anchor as core_circle_arc_middle_anchor,
    composite_linking as core_composite_linking, equate_eg as core_equate_eg,
    equate_eg_ext as core_equate_eg_ext, equate_neat as core_equate_neat,
    equate_neat_linear as core_equate_neat_linear, loglinear_smooth as core_loglinear_smooth,
    nominal_weights_mean_equate as core_nominal_weights_mean_equate, AnchorKind, CircleArcMethod,
    Continuization, EgSmoothOptions, EquateMethod, EquateResult, NeatLinearMethod, NeatMethod,
    SeeResult,
};
use mlsirm_core::fitstats::{
    benjamini_hochberg as core_benjamini_hochberg, chi2_sf as core_chi2_sf,
    cluster_moment_covariance as core_cluster_moment_covariance,
    factorized_multilevel_moments as core_factorized_multilevel_moments,
    factorized_trait_moments as core_factorized_trait_moments,
    infit_outfit as core_infit_outfit, leniency_residuals as core_leniency_residuals,
    m2_cmle_rasch as core_m2_cmle_rasch, m2_rmsea2 as core_m2,
    m2_rmsea2_structured as core_m2_structured, person_fit as core_person_fit,
    poly_local_dependence as core_poly_ld, poly_m2 as core_poly_m2,
    projected_m2 as core_projected_m2,
    projected_m2_workspace_elements as core_projected_m2_workspace_elements,
    s_x2 as core_s_x2, SX2Config, PROJECTED_M2_MAX_WORKSPACE_ELEMENTS,
};
use mlsirm_core::linking::{
    irt_link as core_irt_link, link_fixed_item_parameters as core_link_fixed_item_parameters,
    LinkMethod,
};
use mlsirm_core::inference::{
    finite_difference_hessian as core_finite_difference_hessian,
    second_order_test as core_second_order_test,
    standard_errors_from_vcov as core_standard_errors_from_vcov,
    vcov_from_hessian as core_vcov_from_hessian,
};
use mlsirm_core::jmle_opt::{adam as core_jmle_adam, lbfgs as core_jmle_lbfgs, run_optimizer as core_jmle_run_optimizer};
use mlsirm_core::marginal::{
    fit_marginal_full as core_fit_marginal_full, Anchors, ItemCovariate, MarginalConfig,
    PopulationSpec, XiRuleKind,
};
use mlsirm_core::nodes::XiRule;

use mlsirm_core::cdm::{
    fit_cdm as core_fit_cdm, fit_gdina as core_fit_gdina, fit_ho_cdm as core_fit_ho_cdm,
    fit_ho_gdina as core_fit_ho_gdina, fit_seq_gdina as core_fit_seq_gdina,
    fit_seq_gdina_qr as core_fit_seq_gdina_qr, gdina_wald_selection as core_gdina_wald_selection,
    validate_q_matrix as core_validate_q_matrix, CdmConfig, CdmModel,
};
use mlsirm_core::classification::{
    hanson_brennan as core_hanson_brennan,
    hanson_brennan_from_params as core_hanson_brennan_from_params,
    lee_classification as core_lee_classification,
    livingston_correlation as core_livingston_correlation, livingston_k2 as core_livingston_k2,
    livingston_lewis as core_livingston_lewis, rudner_classification as core_rudner_classification,
    subkoviak_agreement as core_subkoviak_agreement,
    woodruff_sawyer_normal as core_woodruff_sawyer_normal,
    woodruff_sawyer_sb as core_woodruff_sawyer_sb, ClassificationResult, HansonBrennanResult,
    WoodruffSawyerResult,
};
use mlsirm_core::crm::fit_crm as core_fit_crm;
use mlsirm_core::detect::detect_analysis as core_detect_analysis;
use mlsirm_core::detect::dimtest as core_dimtest;
use mlsirm_core::dif::{
    breslow_day_dif as core_breslow_day_dif, delta_plot as core_delta_plot,
    eb_mh_dif as core_eb_mh_dif, gmh_dif as core_gmh_dif, logistic_dif as core_logistic_dif,
    logistic_dif_purified as core_logistic_purified, mantel_haenszel_dif as core_mh_dif,
    mantel_haenszel_dif_purified as core_mh_purified, mantel_smd_dif as core_mantel_smd_dif,
    raju_area as core_raju_area, sibtest as core_sibtest, DeltaThreshold, ExtremeAdjust,
    LogisticDifConfig, LogisticDifRow, MhDifConfig, MhDifRow, PurifyConfig,
    PurifyType as DeltaPurifyType, SibtestConfig,
};
use mlsirm_core::exposure::{
    a_stratified as core_a_stratified, ccat_select as core_ccat_select,
    ci_classify as core_ci_classify, epv_select as core_epv_select,
    flexilevel_administer as core_flexilevel_administer,
    flexilevel_score_distribution as core_flexilevel_score_distribution,
    kl_information as core_kl_information, kl_select as core_kl_select, owen_cat as core_owen_cat,
    owen_update as core_owen_update, pyramidal_administer as core_pyramidal_administer,
    sprt_classify as core_sprt_classify, stradaptive_administer as core_stradaptive_administer,
    sympson_hetter as core_sympson_hetter, two_stage_route as core_two_stage_route,
    two_stage_score as core_two_stage_score, AStratifiedConfig, SympsonHetterConfig,
};
use mlsirm_core::facets::fit_facets as core_fit_facets;
use mlsirm_core::factor::{
    glb_fa_corr as core_glb_fa_corr, glb_fa_data as core_glb_fa_data,
    minres_fa_corr as core_minres_fa_corr, minres_fa_data as core_minres_fa_data,
    omega_total_1f_corr as core_omega_total_1f_corr,
    omega_total_1f_data as core_omega_total_1f_data, velicer_map_corr as core_velicer_map_corr,
    velicer_map_data as core_velicer_map_data, MinresFaResult,
};
use mlsirm_core::fitstats::{
    adjusted_chi2_pairs as core_adjusted_chi2_pairs,
    person_fit_resampling as core_person_fit_resampling,
    residual_item_fit as core_residual_item_fit, tcc_drift as core_tcc_drift,
};
use mlsirm_core::gpcm::{fit_gpcm as core_fit_gpcm, GpcmConfig};
use mlsirm_core::grm::{fit_grm as core_fit_grm, GrmConfig};
use mlsirm_core::gtheory::{
    gtheory_pi as core_gtheory_pi, gtheory_pio as core_gtheory_pio, phi_lambda as core_phi_lambda,
    GTheoryDStudyRow,
};
use mlsirm_core::ksirt::{ksirt as core_ksirt, KsirtKernel};
use mlsirm_core::lltm::{fit_lltm as core_fit_lltm, LltmConfig};
use mlsirm_core::mhrm::{fit_mhrm as core_fit_mhrm, MhrmConfig, MhrmModel};
use mlsirm_core::mixed::{fit_mixed_items as core_fit_mixed_items, MixedItemKind, MixedItemSpec};
use mlsirm_core::mixture::{fit_mixture as core_fit_mixture, MixtureConfig, MixtureModel};
use mlsirm_core::mmle::{fit_mmle_2pl as core_fit_mmle_2pl, MmleConfig};
use mlsirm_core::mokken::{aisp as core_mokken_aisp, coef_h as core_mokken_coef_h};
use mlsirm_core::nominal::{fit_nominal as core_fit_nominal_model, NominalConfig};
use mlsirm_core::parallel::parallel_analysis as core_parallel_analysis;
use mlsirm_core::personfit_np::person_fit_np as core_person_fit_np;
use mlsirm_core::poly::{
    fit_nominal as core_fit_nominal, fit_poly_unidim as core_fit_poly_unidim,
    gpcm_logprobs as core_gpcm_logprobs, grm_logprobs as core_grm_logprobs,
    poly_cat_simulate as core_poly_cat_simulate, poly_dif_sweep as core_poly_dif,
    poly_information_curves as core_poly_information_curves,
    poly_person_fit as core_poly_person_fit, poly_s_x2 as core_poly_s_x2,
    score_poly_eap as core_score_poly_eap, u3_poly_bootstrap_cutoff as core_u3_poly_cutoff,
    u3_poly_person_fit as core_u3_poly_person_fit, PolyModel,
};
use mlsirm_core::poly_marginal::fit_poly_lsirm as core_fit_poly_lsirm;
use mlsirm_core::rasch_cml::{
    andersen_lr_test as core_andersen_lr, fit_rasch_cml as core_fit_rasch_cml,
};
use mlsirm_core::reliability::guttman_lambdas as core_guttman_lambdas;
use mlsirm_core::reliability::tenberge_mu as core_tenberge_mu;
use mlsirm_core::reliability::{
    bhapkar_mh as core_bhapkar_mh, cronbach_alpha as core_cronbach_alpha,
    feldt_alpha_ci as core_feldt_alpha_ci, finn_coefficient as core_finn_coefficient,
    icc as core_icc, kripp_alpha as core_kripp_alpha, maxwell_re as core_maxwell_re,
    mean_pairwise_cor as core_mean_pairwise_cor, mean_pairwise_rho as core_mean_pairwise_rho,
    n_cohen_kappa as core_n_cohen_kappa, rater_bias as core_rater_bias,
    robinson_a as core_robinson_a, separation_reliability as core_separation_reliability,
    stuart_maxwell_mh as core_stuart_maxwell_mh,
};
use mlsirm_core::rsm::fit_rsm as core_fit_rsm;
use mlsirm_core::rt::{
    fit_rt_lognormal as core_fit_rt, rt_person_fit as core_rt_person_fit, RtConfig,
};
use mlsirm_core::rt_joint::{fit_speed_accuracy_covariance as core_fit_sa, SpeedAccuracyConfig};
use mlsirm_core::scoring::{
    bank_information_device as core_bank_information_device,
    cat_ability_eap_device as core_cat_ability_eap_device,
    cat_ability_mle_device as core_cat_ability_mle_device,
    cat_ability_standard_error_device as core_cat_ability_standard_error_device,
    cat_item_information_device as core_cat_item_information_device,
    cat_select_item_device as core_cat_select_item_device,
    cat_next_item_device as core_cat_next_item_device,
    eapsum_tables_device as core_eapsum_tables_device,
    empirical_reliability_device as core_empirical_reliability_device,
    plausible_values_device as core_plausible_values_device,
    score_eap_device as core_score_eap_device, score_eapsum_device as core_score_eapsum_device,
    score_map as core_score_map, score_wle as core_score_wle,
    score_wle_poly as core_score_wle_poly, EapSumTable, ItemBank, PriorSpec,
};
use mlsirm_core::security::gbt as core_gbt;
use mlsirm_core::security::k_index as core_k_index;
use mlsirm_core::security::k_variants as core_k_variants;
use mlsirm_core::security::wollack_omega as core_wollack_omega;
use mlsirm_core::standard_setting::hofstee as core_hofstee;
use mlsirm_core::subscores::subscores as core_subscores;
use mlsirm_core::test_form::assemble_test_form_greedy as core_assemble_test_form_greedy;
use mlsirm_core::testlet::{fit_testlet as core_fit_testlet, TestletConfig, TestletModel};
use mlsirm_core::twopl::{fit_2pl as core_fit_2pl, TwoPlConfig};
use mlsirm_core::utility::{
    selection_utility as core_selection_utility, taylor_russell as core_taylor_russell,
};

fn parse_poly_model(model: &str) -> PyResult<PolyModel> {
    match model.to_lowercase().as_str() {
        "grm" => Ok(PolyModel::Grm),
        "gpcm" => Ok(PolyModel::Gpcm),
        other => Err(PyValueError::new_err(format!(
            "model must be grm or gpcm, got {other}"
        ))),
    }
}

fn poly_responses(y: &[i64], observed: Option<&[bool]>, n_cat: usize) -> PyResult<Vec<usize>> {
    if observed.is_some_and(|o| o.len() != y.len()) {
        return Err(PyValueError::new_err(
            "observed must have the same length as responses",
        ));
    }
    let mut yv = Vec::with_capacity(y.len());
    for (idx, &v) in y.iter().enumerate() {
        if observed.map_or(true, |o| o[idx]) && (v < 0 || v as usize >= n_cat) {
            return Err(PyValueError::new_err(
                "observed responses must be integer categories in 0..n_cat-1",
            ));
        }
        yv.push(if v < 0 { 0 } else { v as usize });
    }
    Ok(yv)
}
use mlsirm_core::{
    neg_loglik_and_grad_device as core_neg_loglik_and_grad_device, Device, ModelConfig, ModelType,
    Params, PenaltyConfig,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y,
    mask,
    factor_id,
    theta,
    alpha,
    b,
    xi,
    zeta,
    tau,
    model,
    eps_distance,
    lambda_theta,
    lambda_xi,
    lambda_zeta,
    lambda_b,
    lambda_alpha,
    lambda_tau,
    mu_alpha,
    mu_tau,
    device = "cpu",
))]
fn neg_loglik_and_grad(
    y: PyReadonlyArray2<'_, f64>,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    factor_id: PyReadonlyArray1<'_, i64>,
    theta: PyReadonlyArray2<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray2<'_, f64>,
    zeta: PyReadonlyArray2<'_, f64>,
    tau: f64,
    model: &str,
    eps_distance: f64,
    lambda_theta: f64,
    lambda_xi: f64,
    lambda_zeta: f64,
    lambda_b: f64,
    lambda_alpha: f64,
    lambda_tau: f64,
    mu_alpha: f64,
    mu_tau: f64,
    device: &str,
) -> PyResult<(f64, HashMap<String, Vec<f64>>, f64)> {
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let y_shape = y.shape();
    let theta_shape = theta.shape();
    let xi_shape = xi.shape();
    let zeta_shape = zeta.shape();
    validate_shapes(
        y_shape,
        factor_id.shape(),
        theta_shape,
        alpha.shape(),
        b.shape(),
        xi_shape,
        zeta_shape,
    )?;

    if let Some(mask_ref) = &mask {
        if mask_ref.shape() != y_shape {
            return Err(PyValueError::new_err("mask shape must match responses"));
        }
    }

    let factors = convert_factor_id(factor_id.as_slice()?, theta_shape[1])?;
    let config = ModelConfig {
        n_persons: y_shape[0],
        n_items: y_shape[1],
        n_dims: theta_shape[1],
        latent_dim: xi_shape[1],
        model_type: parse_model_type(model)?,
        eps_distance,
    };
    if matches!(config.model_type, ModelType::Uls2plm | ModelType::Ulsrm) && config.n_dims != 1 {
        return Err(PyValueError::new_err(format!(
            "{} requires one trait dimension",
            model.to_uppercase()
        )));
    }

    let params = Params {
        theta: theta.as_slice()?.to_vec(),
        alpha: alpha.as_slice()?.to_vec(),
        b: b.as_slice()?.to_vec(),
        xi: xi.as_slice()?.to_vec(),
        zeta: zeta.as_slice()?.to_vec(),
        tau,
    };
    let penalty = PenaltyConfig {
        lambda_theta,
        lambda_xi,
        lambda_zeta,
        lambda_b,
        lambda_alpha,
        lambda_tau,
        mu_alpha,
        mu_tau,
    };

    let y_slice = y.as_slice()?;
    let mask_storage = match mask {
        Some(mask_ref) => Some(mask_ref.as_slice()?.to_vec()),
        None => None,
    };
    let (objective, grad, loglik) = core_neg_loglik_and_grad_device(
        device,
        y_slice,
        mask_storage.as_deref(),
        &factors,
        &params,
        &config,
        &penalty,
    );

    let mut gradients = HashMap::new();
    gradients.insert("theta".to_string(), grad.theta);
    gradients.insert("alpha".to_string(), grad.alpha);
    gradients.insert("b".to_string(), grad.b);
    gradients.insert("xi".to_string(), grad.xi);
    gradients.insert("zeta".to_string(), grad.zeta);
    gradients.insert("tau".to_string(), vec![grad.tau]);
    Ok((objective, gradients, loglik))
}

/// MMLE-EM calibration of a unidimensional 2PL (`mlsirm_core::mmle`).
/// `y` and `observed` are row-major flattened `n_persons * n_items` arrays;
/// cells where `observed` is false are ignored (missing-at-random safe).
#[pyfunction]
fn fit_mmle_2pl(
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool)> {
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| PyValueError::new_err("n_persons * n_items overflows"))?;
    let y_slice = y.as_slice()?;
    let observed_slice = observed.as_slice()?;
    if y_slice.len() != expected || observed_slice.len() != expected {
        return Err(PyValueError::new_err(
            "y and observed must both have length n_persons * n_items",
        ));
    }
    let cfg = MmleConfig {
        max_iter,
        tol,
        ..MmleConfig::default()
    };
    let res = core_fit_mmle_2pl(y_slice, observed_slice, n_persons, n_items, &cfg);
    Ok((res.a, res.b, res.theta, res.loglik_trace, res.converged))
}

/// Marginal-EM fit of a DINA/DINO cognitive diagnosis model (`mlsirm_core::cdm`).
/// `y`/`observed` are row-major `n_persons * n_items`; `q_matrix` is row-major
/// `n_items * n_attributes` with 0/1 entries; `model` is "dina" or "dino". Returns
/// a dict with `slip`, `guess`, `profile_prob` (`2^K`), `map_profile` (bit-encoded,
/// per person), `attr_prob` (`n_persons * n_attributes`), `loglik_trace`, `n_iter`,
/// `converged` and `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, model = "dina", max_iter = 500, tol = 1e-6))]
fn fit_cdm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    model: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let gate = match model {
        "dina" | "DINA" => CdmModel::Dina,
        "dino" | "DINO" => CdmModel::Dino,
        other => {
            return Err(PyValueError::new_err(format!(
                "model must be 'dina' or 'dino'; got {other}"
            )))
        }
    };
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_cdm(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        gate,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("model", model)?;
    out.set_item("slip", res.slip)?;
    out.set_item("guess", res.guess)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Marginal-EM fit of the saturated G-DINA model (`mlsirm_core::cdm::fit_gdina`).
/// `y`/`observed` are row-major `n_persons * n_items`; `q_matrix` is row-major
/// `n_items * n_attributes` with 0/1 entries. Item parameters are ragged (CSR): item
/// `i` owns `item_prob`/`item_delta` slice `[item_off[i]..item_off[i+1])` of width
/// `2^{K_i}`. Returns a dict with `item_off`, `item_prob`, `item_delta`, `k_required`,
/// `profile_prob`, `map_profile`, `attr_prob`, `loglik_trace`, `n_iter`, `converged`,
/// `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, max_iter = 500, tol = 1e-6))]
fn fit_gdina(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_gdina(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_off", res.item_off)?;
    out.set_item("item_prob", res.item_prob)?;
    out.set_item("item_delta", res.item_delta)?;
    out.set_item("k_required", res.k_required)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Shared-Q sequential (continuation-ratio) G-DINA for ordered polytomous responses
/// (Ma & de la Torre, 2016; `mlsirm_core::cdm::fit_seq_gdina`). Every step of an item is a
/// saturated G-DINA over the SAME required attributes (Q row `i`); this is a restriction of
/// the general per-step `q_ik` model — step-distinct attribute requirements are a deferred
/// non-goal, so supply each item's Q-vector as the UNION of its steps' required attributes.
/// `y` holds ordered
/// integer categories `0..=M_i` where observed (`M_i` = max observed category per item);
/// `observed`/`q_matrix` are row-major `n_persons*n_items` / `n_items*n_attributes`.
/// Item parameters are ragged, CLASS-MAJOR CSR: item `i` owns `step_prob` slice
/// `[s_off[i]..s_off[i+1])` (`s_ik(l)` at `s_off[i] + l*M_i + (k-1)`) and `cat_prob`
/// slice `[cat_off[i]..cat_off[i+1])` (`P(X_i=x|l)` at `cat_off[i] + l*(M_i+1) + x`).
/// Returns a dict with `s_off`, `step_prob`, `cat_off`, `cat_prob`, `max_cat`,
/// `k_required`, `profile_prob`, `map_profile`, `attr_prob`, `loglik_trace`, `n_iter`,
/// `converged`, `termination_reason`, `final_loglik_change`,
/// `final_relative_loglik_change`, `stopping_tolerance`, `n_parameters`.
/// Per-step-Q sequential G-DINA (Ma & de la Torre, 2016; `mlsirm_core::cdm::fit_seq_gdina_qr`),
/// the full restricted-Q model where each ordered STEP has its own attribute requirement.
/// `step_q` is row-major `(sum_i n_steps[i]) * n_attributes` (0/1); `n_steps[i] = M_i` (the step
/// count, which must equal item `i`'s maximum observed category), so step `k` of item `i` is row
/// `step_off[i] + (k-1)` with `step_off = cumsum(n_steps)`. Generalizes `fit_seq_gdina` (which is
/// this with every step sharing the item's Q). Step probs are STEP-ROW-major:
/// `step_prob[spo[step_off[i]+(k-1)] + l]` (`l` the reduced class under `q_ik`, width
/// `2^{|q_ik|}`); category probs are union-class-major:
/// `cat_prob[cat_off[i] + uc*(M_i+1) + x]` over the item's union `2^{K^u_i}` classes. Returns a
/// dict with `step_off`, `spo`, `step_prob`, `step_kq`, `cat_off`, `cat_prob`, `max_cat`,
/// `union_k`, `profile_prob`, `map_profile`, `attr_prob`, `loglik_trace`, `n_iter`, `converged`,
/// `termination_reason`, `final_loglik_change`, `final_relative_loglik_change`,
/// `stopping_tolerance`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, step_q, n_steps, n_persons, n_items, n_attributes, max_iter = 500, tol = 1e-6))]
fn fit_seq_gdina_qr(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    step_q: PyReadonlyArray1<'_, i64>,
    n_steps: Vec<usize>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let sq: Vec<u8> = step_q
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("step_q entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_seq_gdina_qr(
        y.as_slice()?,
        observed.as_slice()?,
        &sq,
        &n_steps,
        n_persons,
        n_items,
        n_attributes,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("step_off", res.step_off)?;
    out.set_item("spo", res.spo)?;
    out.set_item("step_prob", res.step_prob)?;
    out.set_item("step_kq", res.step_kq)?;
    out.set_item("cat_off", res.cat_off)?;
    out.set_item("cat_prob", res.cat_prob)?;
    out.set_item("max_cat", res.max_cat)?;
    out.set_item("union_k", res.union_k)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item(
        "final_relative_loglik_change",
        res.final_relative_loglik_change,
    )?;
    out.set_item("stopping_tolerance", res.stopping_tolerance)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, max_iter = 500, tol = 1e-6))]
fn fit_seq_gdina(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_seq_gdina(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("s_off", res.s_off)?;
    out.set_item("step_prob", res.step_prob)?;
    out.set_item("cat_off", res.cat_off)?;
    out.set_item("cat_prob", res.cat_prob)?;
    out.set_item("max_cat", res.max_cat)?;
    out.set_item("k_required", res.k_required)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item(
        "final_relative_loglik_change",
        res.final_relative_loglik_change,
    )?;
    out.set_item("stopping_tolerance", res.stopping_tolerance)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Empirical Q-matrix validation by the PVAF method (de la Torre & Chiu, 2016;
/// `mlsirm_core::cdm::validate_q_matrix`). `y`/`observed` are row-major
/// `n_persons * n_items`; `provisional_q` is row-major `n_items * n_attributes`
/// with 0/1 entries, each item loading at least one attribute. `epsilon` is the
/// PVAF cutoff (0.95 typical). Returns a dict with `suggested_q` (row-major
/// `n_items * n_attributes`), `suggested_pvaf`, `provisional_pvaf`, `flagged`,
/// `n_attributes`, `epsilon`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, provisional_q, n_persons, n_items, n_attributes, epsilon = 0.95, max_iter = 500, tol = 1e-6))]
fn validate_q_matrix(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    provisional_q: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let q: Vec<u8> = provisional_q
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "provisional_q entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_validate_q_matrix(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        epsilon,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    let suggested: Vec<i64> = res.suggested_q.iter().map(|&v| v as i64).collect();
    out.set_item("suggested_q", suggested)?;
    out.set_item("suggested_pvaf", res.suggested_pvaf)?;
    out.set_item("provisional_pvaf", res.provisional_pvaf)?;
    out.set_item("flagged", res.flagged)?;
    out.set_item("n_attributes", res.n_attributes)?;
    out.set_item("epsilon", res.epsilon)?;
    out.set_item("calibration_n_iter", res.calibration_n_iter)?;
    out.set_item("calibration_max_iter", res.calibration_max_iter)?;
    out.set_item(
        "calibration_termination_reason",
        res.calibration_termination_reason,
    )?;
    out.set_item(
        "calibration_final_loglik_change",
        res.calibration_final_loglik_change,
    )?;
    out.set_item("calibration_tol", res.calibration_tol)?;
    Ok(out.into())
}

/// Item-level CDM model selection by the Wald test (de la Torre & Lee, 2013;
/// `mlsirm_core::cdm::gdina_wald_selection`). `y`/`observed` are row-major
/// `n_persons * n_items`; `q_matrix` row-major `n_items * n_attributes` (0/1).
/// Each item's saturated G-DINA is Wald-tested against the reduced DINA, DINO, and
/// A-CDM models; `alpha` is the test level. Returns a dict with `models` (candidate
/// names), `wald_stat`/`wald_df`/`p_value` (row-major `n_items * n_models`),
/// `selected` (per item: model index or -1 for the saturated G-DINA), `alpha`.
/// A nonconverged saturated calibration raises `ValueError`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, alpha = 0.05, max_iter = 500, tol = 1e-6))]
fn gdina_wald_selection(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_gdina_wald_selection(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        alpha,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("models", res.models)?;
    out.set_item("wald_stat", res.wald_stat)?;
    out.set_item("wald_df", res.wald_df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("selected", res.selected)?;
    out.set_item("alpha", res.alpha)?;
    Ok(out.into())
}

/// Higher-order DINA/DINO fit (de la Torre & Douglas, 2004;
/// `mlsirm_core::cdm::fit_ho_cdm`). `y`/`observed` are row-major `n_persons *
/// n_items`; `q_matrix` row-major `n_items * n_attributes` (0/1); `model` is "dina"
/// or "dino". Attribute mastery is structured by a continuous trait
/// `theta ~ N(0,1)`, `P(alpha_k=1|theta)=sigmoid(attr_slope_k*theta+attr_intercept_k)`.
/// Returns a dict with `model`, `slip`, `guess`, `attr_slope` (K), `attr_intercept`
/// (K), `profile_prob` (implied, 2^K), `theta` (N), `map_profile`, `attr_prob`
/// (`N*K`), `loglik_trace`, `n_iter`, `converged`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, model = "dina", max_iter = 500, tol = 1e-6))]
fn fit_ho_cdm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    model: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let gate = match model {
        "dina" | "DINA" => CdmModel::Dina,
        "dino" | "DINO" => CdmModel::Dino,
        other => {
            return Err(PyValueError::new_err(format!(
                "model must be 'dina' or 'dino'; got {other}"
            )))
        }
    };
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_ho_cdm(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        gate,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("model", model)?;
    out.set_item("slip", res.slip)?;
    out.set_item("guess", res.guess)?;
    out.set_item("attr_slope", res.attr_slope)?;
    out.set_item("attr_intercept", res.attr_intercept)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("theta", res.theta)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("max_iter", res.max_iter)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("stopping_tolerance", res.stopping_tolerance)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Higher-order G-DINA fit (de la Torre & Douglas, 2004 x de la Torre, 2011;
/// `mlsirm_core::cdm::fit_ho_gdina`). The saturated G-DINA item model under a
/// higher-order structural attribute prior `theta ~ N(0,1)`. `y`/`observed` are
/// row-major `n_persons * n_items` (0/1); `q_matrix` row-major `n_items *
/// n_attributes` (0/1). Returns a dict with the ragged CSR `item_off`, `item_prob`,
/// `item_delta`, `k_required`; `attr_slope`/`attr_intercept` (K); `profile_prob`
/// (implied, 2^K); `theta`; `map_profile`; `attr_prob` (`N*K`); `loglik_trace`,
/// `n_iter`, `converged`, `termination_reason`, `final_loglik_change`,
/// `final_relative_loglik_change`, `stopping_tolerance`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_matrix, n_persons, n_items, n_attributes, max_iter = 500, tol = 1e-6))]
fn fit_ho_gdina(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_matrix: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let q: Vec<u8> = q_matrix
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("q_matrix entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig {
        max_iter,
        tol,
        ..CdmConfig::default()
    };
    let res = core_fit_ho_gdina(
        y.as_slice()?,
        observed.as_slice()?,
        &q,
        n_persons,
        n_items,
        n_attributes,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_off", res.item_off)?;
    out.set_item("item_prob", res.item_prob)?;
    out.set_item("item_delta", res.item_delta)?;
    out.set_item("k_required", res.k_required)?;
    out.set_item("attr_slope", res.attr_slope)?;
    out.set_item("attr_intercept", res.attr_intercept)?;
    out.set_item("profile_prob", res.profile_prob)?;
    out.set_item("theta", res.theta)?;
    out.set_item("map_profile", res.map_profile)?;
    out.set_item("attr_prob", res.attr_prob)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item(
        "final_relative_loglik_change",
        res.final_relative_loglik_change,
    )?;
    out.set_item("stopping_tolerance", res.stopping_tolerance)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Confirmatory compensatory multidimensional 2PL (Reckase, 2009; Bock, Gibbons, & Muraki,
/// 1988; `mlsirm_core::twopl::fit_2pl`). Each item may load FREELY on several
/// latent dimensions `theta ~ MVN(0, Sigma)` that trade off additively in the logit:
/// `P(X=1) = sigmoid(sum_d L_id a_id theta_d + b_i)`. `loading_pattern` is a row-major
/// `n_items * n_dims` 0/1 pattern; each dimension needs a pure single-loading anchor item
/// (identification; the all-ones pattern is rejected). With `estimate_corr = False` the
/// factors are ORTHOGONAL (`Sigma = I`); with `estimate_corr = True` the inter-factor
/// correlation matrix is estimated (Cholesky node-map + a monotone ECM step). `node_rule` picks
/// the E-step quadrature: `"gh"` (Gauss-Hermite product grid, `n_dims <= 3`) or `"qmc"`/`"mc"`
/// (Halton QMC / Monte-Carlo with `xi_points` prior draws, `n_dims <= 6`; Jank, 2005). `q`
/// applies to `"gh"` only; `xi_points`/`xi_seed` to `"qmc"`/`"mc"` only. Returns a dict
/// with `loading` (row-major `n_items * n_dims`, `0` off-pattern), `intercept`, `theta`
/// (`n_persons * n_dims` EAP), `n_dims`, `corr` (row-major `n_dims * n_dims`, identity when not
/// estimated), `loglik_trace`, `n_iter`, `converged`, `termination_reason`,
/// `final_loglik_change`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, loading_pattern, n_persons, n_items, n_dims, q = 21, estimate_corr = false, max_iter = 500, tol = 1e-6, node_rule = "gh", xi_points = 4000, xi_seed = 0x9E37_79B9_7F4A_7C15))]
fn fit_2pl(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    loading_pattern: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    q: usize,
    estimate_corr: bool,
    max_iter: usize,
    tol: f64,
    node_rule: &str,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let pattern: Vec<u8> = loading_pattern
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "loading_pattern entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    // `node_rule`: "gh" (Gauss-Hermite product grid, D<=3) or "qmc"/"mc" (Halton/Monte-Carlo,
    // D<=6). q applies only to "gh"; xi_points/xi_seed only to the QMC/MC rules.
    let xi_rule = XiRuleKind::parse(node_rule)
        .ok_or_else(|| PyValueError::new_err("node_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let cfg = TwoPlConfig {
        max_iter,
        tol,
        q,
        estimate_corr,
        xi_rule,
        xi_points,
        xi_seed,
        ..TwoPlConfig::default()
    };
    let res = core_fit_2pl(
        y.as_slice()?,
        observed.as_slice()?,
        &pattern,
        n_persons,
        n_items,
        n_dims,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("loading", res.loading)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("theta", res.theta)?;
    out.set_item("n_dims", res.n_dims)?;
    out.set_item("corr", res.corr)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Confirmatory MULTIDIMENSIONAL 2PL by Metropolis-Hastings Robbins-Monro (Cai, 2010;
/// `mlsirm_core::mhrm::fit_mhrm`). A STOCHASTIC-approximation EM that scales confirmatory item factor
/// analysis to a latent dimensionality where the deterministic Gauss-Hermite / QMC E-steps of
/// `fit_2pl` are infeasible: each cycle imputes `theta` by a short persistent random-walk Metropolis
/// chain, then takes one Robbins-Monro stochastic-Newton step on the block-diagonal (per-item)
/// complete-data score/information. Orthogonal factors (`Sigma = I`). `y` is a row-major
/// `n_persons * n_items` binary array; `observed` an optional bool mask (missing dropped MAR).
/// The response family is chosen by `model`: `"2pl"` (binary, DEFAULT) or `"gpcm"` (ordered
/// polytomous generalized partial credit model, Muraki, 1992, with `n_cat` categories `0..n_cat`).
/// For GPCM `base_i = sum_d a_id theta_d` (NO intercept) and `P(Y=k) = softmax_k(k*base_i + step_ik)`
/// with `n_cat - 1` free UNORDERED step intercepts per item, estimated by the SAME MH-RM machinery
/// with the closed-form multinomial Hessian as the Robbins-Monro preconditioner and Louis
/// information. Returns a dict with `loading` (row-major `n_items * n_dims`, `0` off-pattern,
/// reflection-canonicalized), `intercept` (`n_items`; EMPTY for GPCM), `step` (row-major
/// `n_items * (n_cat - 1)` GPCM step intercepts; EMPTY for the 2PL), `n_cat`, `theta`
/// (`n_persons * n_dims` trait EAP), `n_dims`, `corr`, `se_loading`/`se_intercept`/`se_step` (Louis
/// observed-information SEs; empty when `estimate_se = false`), `acceptance_rate`, `n_cycles`,
/// `converged`, `termination_reason`, `final_param_change`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, loading_pattern, n_persons, n_items, n_dims, max_cycles = 2000, burn_in = 200, mh_steps = 5, proposal_sd = 1.0, target_accept = 0.30, tol = 1e-3, seed = 0x9E37_79B9_7F4A_7C15, estimate_se = true, estimate_corr = false, model = "2pl", n_cat = 2))]
fn fit_mhrm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    loading_pattern: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    max_cycles: usize,
    burn_in: usize,
    mh_steps: usize,
    proposal_sd: f64,
    target_accept: f64,
    tol: f64,
    seed: u64,
    estimate_se: bool,
    estimate_corr: bool,
    model: &str,
    n_cat: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let model_kind = match model {
        "2pl" | "2PL" => MhrmModel::TwoPl,
        "gpcm" | "GPCM" => {
            if n_cat < 2 {
                return Err(PyValueError::new_err("n_cat must be >= 2 for the GPCM"));
            }
            MhrmModel::Gpcm { n_cat }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "model must be '2pl' or 'gpcm', got '{other}'"
            )))
        }
    };
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err("y responses must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let pattern: Vec<u8> = loading_pattern
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "loading_pattern entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    let obs_vec: Option<Vec<bool>> = match &observed {
        Some(o) => Some(o.as_slice()?.to_vec()),
        None => None,
    };
    let cfg = MhrmConfig {
        max_cycles,
        burn_in,
        mh_steps,
        proposal_sd,
        target_accept,
        tol,
        seed,
        estimate_se,
        estimate_corr,
        model: model_kind,
        ..MhrmConfig::default()
    };
    let res = core_fit_mhrm(
        &yy,
        obs_vec.as_deref(),
        &pattern,
        n_persons,
        n_items,
        n_dims,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("loading", res.loading)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("step", res.step)?;
    out.set_item("n_cat", res.n_cat)?;
    out.set_item("theta", res.theta)?;
    out.set_item("n_dims", res.n_dims)?;
    out.set_item("corr", res.corr)?;
    out.set_item("se_loading", res.se_loading)?;
    out.set_item("se_intercept", res.se_intercept)?;
    out.set_item("se_step", res.se_step)?;
    out.set_item("acceptance_rate", res.acceptance_rate)?;
    out.set_item("n_cycles", res.n_cycles)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_param_change", res.final_param_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen, Cai, & Bock, 2010;
/// `mlsirm_core::nominal::fit_nominal`). Each item's `n_cat` UNORDERED categories get a
/// free multidimensional discrimination `a_ikd` (free on the confirmatory `loading_pattern`, items x
/// n_dims 0/1) and intercept `c_ik`, with the baseline category `0` pinned to `0`:
/// `P(Y=k|theta) = softmax_k(sum_d a_ikd theta_d + c_ik)`, `theta ~ MVN(0, I)`. Reduces to
/// `fit_nominal` at `n_dims = 1`. `node_rule` picks the E-step quadrature: `"gh"` (`n_dims <= 3`) or
/// `"qmc"`/`"mc"` (Halton/Monte-Carlo, `n_dims <= 6`). `y` is a row-major `n_persons * n_items`
/// integer category array; `observed` an optional bool mask (missing dropped MAR). Returns a dict
/// with `slope` (row-major `n_items * n_cat * n_dims`, baseline/off-pattern `0`), `intercept`
/// (`n_items * n_cat`), `theta` (`n_persons * n_dims` EAP), `n_dims`, `n_cat`, `loglik_trace`,
/// `n_iter`, `converged`, `termination_reason`, `final_loglik_change`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, loading_pattern, n_persons, n_items, n_dims, n_cat, q = 21, max_iter = 500, tol = 1e-6, node_rule = "gh", xi_points = 4000, xi_seed = 0x9E37_79B9_7F4A_7C15))]
fn fit_nominal_model(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    loading_pattern: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    q: usize,
    max_iter: usize,
    tol: f64,
    node_rule: &str,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err("y categories must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let pattern: Vec<u8> = loading_pattern
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "loading_pattern entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    let obs_vec: Option<Vec<bool>> = match &observed {
        Some(o) => Some(o.as_slice()?.to_vec()),
        None => None,
    };
    let xi_rule = XiRuleKind::parse(node_rule)
        .ok_or_else(|| PyValueError::new_err("node_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let cfg = NominalConfig {
        max_iter,
        tol,
        q,
        xi_rule,
        xi_points,
        xi_seed,
        ..NominalConfig::default()
    };
    let res = core_fit_nominal_model(
        &yy,
        obs_vec.as_deref(),
        &pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", res.slope)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("theta", res.theta)?;
    out.set_item("n_dims", res.n_dims)?;
    out.set_item("n_cat", res.n_cat)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995;
/// `mlsirm_core::grm::fit_grm`). Each item's `n_cat` ORDERED categories share a SINGLE
/// multidimensional discrimination `a_i` (free on the confirmatory `loading_pattern`, items x
/// n_dims 0/1) and have `n_cat-1` ORDERED boundary intercepts `beta_i`:
/// `P(Y>=k|theta) = sigmoid(sum_d a_id theta_d + beta_i,{k-1})`, `theta ~ MVN(0, I)`. Reduces to
/// `fit_poly_unidim(GRM)` at `n_dims = 1`. `node_rule` picks the E-step quadrature: `"gh"`
/// (`n_dims <= 3`) or `"qmc"`/`"mc"` (Halton/Monte-Carlo, `n_dims <= 6`). `y` is a row-major
/// `n_persons * n_items` integer-category array; `observed` an optional bool mask (missing dropped
/// MAR). Returns a dict with `slope` (row-major `n_items * n_dims`, `0` off-pattern,
/// reflection-canonicalized), `threshold` (`n_items * (n_cat-1)`, strictly decreasing per item),
/// `theta` (`n_persons * n_dims` EAP), `n_dims`, `n_cat`, `loglik_trace`, `n_iter`, `converged`,
/// `termination_reason`, `final_loglik_change`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, loading_pattern, n_persons, n_items, n_dims, n_cat, q = 21, max_iter = 500, tol = 1e-6, node_rule = "gh", xi_points = 4000, xi_seed = 0x9E37_79B9_7F4A_7C15))]
fn fit_grm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    loading_pattern: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    q: usize,
    max_iter: usize,
    tol: f64,
    node_rule: &str,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err("y categories must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let pattern: Vec<u8> = loading_pattern
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "loading_pattern entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    let obs_vec: Option<Vec<bool>> = match &observed {
        Some(o) => Some(o.as_slice()?.to_vec()),
        None => None,
    };
    let xi_rule = XiRuleKind::parse(node_rule)
        .ok_or_else(|| PyValueError::new_err("node_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let cfg = GrmConfig {
        max_iter,
        tol,
        q,
        xi_rule,
        xi_points,
        xi_seed,
        ..GrmConfig::default()
    };
    let res = core_fit_grm(
        &yy,
        obs_vec.as_deref(),
        &pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", res.slope)?;
    out.set_item("threshold", res.threshold)?;
    out.set_item("theta", res.theta)?;
    out.set_item("n_dims", res.n_dims)?;
    out.set_item("n_cat", res.n_cat)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Confirmatory MULTIDIMENSIONAL generalized partial credit model fit (Muraki, 1992;
/// `mlsirm_core::gpcm::fit_gpcm`). Ordered polytomous categories with a SINGLE discrimination
/// vector per item and INTEGER category scores: `P(Y = k | theta) = softmax_k(k * sum_d a_id
/// theta_d + step_ik)`, `theta ~ MVN(0, I)`. The `n_cat-1` step intercepts are UNORDERED (the
/// softmax is finite for any values). `node_rule` selects the E-step grid: `"gh"` (Gauss-Hermite,
/// `n_dims <= 3`) or `"qmc"`/`"mc"` (Halton/Monte-Carlo, `n_dims <= 6`). `y` is a row-major
/// `n_persons * n_items` integer-category array; `observed` an optional bool mask (missing dropped
/// MAR). Returns a dict with `slope` (row-major `n_items * n_dims`, `0` off-pattern,
/// reflection-canonicalized), `step` (`n_items * (n_cat-1)`, unordered), `theta`
/// (`n_persons * n_dims` EAP), `n_dims`, `n_cat`, `loglik_trace`, `n_iter`, `converged`,
/// `termination_reason`, `final_loglik_change`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, loading_pattern, n_persons, n_items, n_dims, n_cat, q = 21, max_iter = 500, tol = 1e-6, node_rule = "gh", xi_points = 4000, xi_seed = 0x9E37_79B9_7F4A_7C15))]
fn fit_gpcm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    loading_pattern: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    q: usize,
    max_iter: usize,
    tol: f64,
    node_rule: &str,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err("y categories must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let pattern: Vec<u8> = loading_pattern
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "loading_pattern entries must be 0 or 1",
            )),
        })
        .collect::<PyResult<_>>()?;
    let obs_vec: Option<Vec<bool>> = match &observed {
        Some(o) => Some(o.as_slice()?.to_vec()),
        None => None,
    };
    let xi_rule = XiRuleKind::parse(node_rule)
        .ok_or_else(|| PyValueError::new_err("node_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let cfg = GpcmConfig {
        max_iter,
        tol,
        q,
        xi_rule,
        xi_points,
        xi_seed,
        ..GpcmConfig::default()
    };
    let res = core_fit_gpcm(
        &yy,
        obs_vec.as_deref(),
        &pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", res.slope)?;
    out.set_item("step", res.step)?;
    out.set_item("theta", res.theta)?;
    out.set_item("n_dims", res.n_dims)?;
    out.set_item("n_cat", res.n_cat)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (responses, observed, n_persons, n_items, q_theta = 41, max_iter = 500, tol = 1e-6))]
fn fit_crm(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_fit_crm(
        responses.as_slice()?,
        observed.as_slice()?,
        n_persons,
        n_items,
        q_theta,
        max_iter,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", res.slope)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("resid_sd", res.resid_sd)?;
    out.set_item("discrimination", res.discrimination)?;
    out.set_item("difficulty", res.difficulty)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_delta", res.final_delta)?;
    out.set_item("stopping_tolerance", res.stopping_tolerance)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Rating Scale Model fit (Andrich, 1978; `mlsirm_core::rsm::fit_rsm`). `y`/`observed`
/// are row-major `n_persons * n_items` with categories `0..n_cat-1`. Every item has
/// its own location, but the `n_cat-1` category thresholds are shared across items:
/// `ln[P(k)/P(k-1)] = theta - item_location_i - threshold_k`, `theta ~ N(0,1)`.
/// Returns a dict with `item_location` (`n_items`), `thresholds` (`n_cat-1`, centered),
/// `theta` (per-person EAP), `loglik_trace`, `n_iter`, `converged`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, n_persons, n_items, n_cat, q_theta = 41, max_iter = 500, tol = 1e-6))]
fn fit_rsm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            if v >= 0 {
                Ok(v as usize)
            } else {
                Err(PyValueError::new_err(
                    "y must be non-negative category indices",
                ))
            }
        })
        .collect::<PyResult<_>>()?;
    let obs = observed.as_slice()?;
    let res = core_fit_rsm(
        &yy,
        Some(obs),
        n_persons,
        n_items,
        n_cat,
        q_theta,
        max_iter,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_location", res.item_location)?;
    out.set_item("thresholds", res.thresholds)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Many-Facet Rasch Model fit (Linacre, 1989; `mlsirm_core::facets::fit_facets`).
/// `y`/`observed` are row-major `n_persons * n_items * n_raters` (rater fastest)
/// with categories `0..n_cat-1`. Adjacent-category log-odds:
/// `ln[P(k)/P(k-1)] = theta - item_difficulty_i - rater_severity_j - threshold_k`,
/// `theta ~ N(0,1)`; severities and thresholds are centered to sum 0. Returns a
/// dict with `item_difficulty` (`n_items`), `rater_severity` (`n_raters`),
/// `thresholds` (`n_cat-1`), `theta` (per-person EAP), `loglik_trace`, `n_iter`,
/// `converged`, `connected` (design-linking flag), `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, n_persons, n_items, n_raters, n_cat, q_theta = 41, max_iter = 500, tol = 1e-6))]
fn fit_facets(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yy: Vec<usize> = y
        .as_slice()?
        .iter()
        .map(|&v| {
            if v >= 0 {
                Ok(v as usize)
            } else {
                Err(PyValueError::new_err(
                    "y must be non-negative category indices",
                ))
            }
        })
        .collect::<PyResult<_>>()?;
    let obs = observed.as_slice()?;
    let res = core_fit_facets(
        &yy,
        Some(obs),
        n_persons,
        n_items,
        n_raters,
        n_cat,
        q_theta,
        max_iter,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_difficulty", res.item_difficulty)?;
    out.set_item("rater_severity", res.rater_severity)?;
    out.set_item("thresholds", res.thresholds)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("connected", res.connected)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Mokken scalability coefficients (`mlsirm_core::mokken::coef_h`).
/// `x` is a row-major complete `n_persons * n_items` integer score matrix.
/// Returns a dict with `hij`/`zij` (flattened `J*J`, NaN diagonal), `hi`,
/// `zi` (`J`), and scalars `h`, `z`. Sample statistics follow the mokken R
/// package (van der Ark, 2007, https://doi.org/10.18637/jss.v020.i11).
#[pyfunction]
#[pyo3(signature = (x, n_persons, n_items))]
fn mokken_coef_h(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_mokken_coef_h(x.as_slice()?, n_persons, n_items).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("hij", res.hij)?;
    out.set_item("hi", res.hi)?;
    out.set_item("h", res.h)?;
    out.set_item("zij", res.zij)?;
    out.set_item("zi", res.zi)?;
    out.set_item("z", res.z)?;
    Ok(out.into())
}

/// Mokken automated item selection procedure (`mlsirm_core::mokken::aisp`,
/// the "search normal" algorithm of the mokken R package). Returns per-item
/// scale labels: 0 = unscalable, 1, 2, ... in formation order. `c` is the
/// scalability lower bound (rule of thumb 0.3), `alpha` the nominal
/// significance level.
#[pyfunction]
#[pyo3(signature = (x, n_persons, n_items, c = 0.3, alpha = 0.05))]
fn mokken_aisp(
    x: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    c: f64,
    alpha: f64,
) -> PyResult<Vec<u32>> {
    core_mokken_aisp(x.as_slice()?, n_persons, n_items, c, alpha).map_err(PyValueError::new_err)
}

/// Kernel-smoothing nonparametric option characteristic curves
/// (`mlsirm_core::ksirt`; Ramsay, 1991, as cited in Mazza, Punzo, &
/// McGuire, 2014, https://doi.org/10.18637/jss.v058.i06). `x` is a
/// row-major complete `n_persons * n_items` pre-scored response matrix.
/// Returns a dict with `theta` (`N`), `grid` (`Q`), `bandwidth` (`J`), and
/// per-item lists `options`, `occ` (flattened `m_j * Q`, row-major by
/// option), `expected` (`Q`), plus `expected_total` (`Q`).
#[pyfunction]
#[pyo3(signature = (x, n_persons, n_items, kernel = "gaussian", nevalpoints = 51, bandwidth = None))]
fn ksirt_occ(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    kernel: &str,
    nevalpoints: usize,
    bandwidth: Option<Vec<f64>>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let flat = x.as_slice()?;
    if flat.len() != n_persons * n_items {
        return Err(PyValueError::new_err(format!(
            "x has {} entries, expected n_persons * n_items = {}",
            flat.len(),
            n_persons * n_items
        )));
    }
    let kern = match kernel {
        "gaussian" => KsirtKernel::Gaussian,
        "quadratic" => KsirtKernel::Quadratic,
        "uniform" => KsirtKernel::Uniform,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown kernel '{other}' (expected gaussian, quadratic, or uniform)"
            )))
        }
    };
    let rows: Vec<Vec<f64>> = (0..n_persons)
        .map(|i| flat[i * n_items..(i + 1) * n_items].to_vec())
        .collect();
    let res = core_ksirt(&rows, kern, nevalpoints, bandwidth.as_deref())
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta", res.theta)?;
    out.set_item("grid", res.grid)?;
    out.set_item("bandwidth", res.bandwidth)?;
    out.set_item("expected_total", res.expected_total)?;
    let options: Vec<Vec<f64>> = res.items.iter().map(|it| it.options.clone()).collect();
    let occ: Vec<Vec<f64>> = res
        .items
        .iter()
        .map(|it| it.occ.iter().flatten().copied().collect())
        .collect();
    let expected: Vec<Vec<f64>> = res.items.iter().map(|it| it.expected.clone()).collect();
    out.set_item("options", options)?;
    out.set_item("occ", occ)?;
    out.set_item("expected", expected)?;
    Ok(out.into())
}

/// Haberman subscore added-value analysis (`mlsirm_core::subscores`;
/// Haberman, 2008, as cited in Sinharay, 2010,
/// ETS RR-10-16). `x` is a row-major complete `n_persons * n_items`
/// scored response matrix; `groups[j]` in `0..K` assigns item `j` to a
/// subscale. Returns a dict with per-subscale `alpha`, `prmse_s`,
/// `prmse_x`, `prmse_sx`, `tau`, `beta`, `gamma`, `added_value_s`,
/// `added_value_sx`, `alpha_total`, the `(K+1)^2` flattened `corr`, the
/// `K*K` flattened `disattenuated_corr` (NaN diagonal), and the `n*K`
/// flattened estimator matrices `observed`, `subscore_s`, `subscore_x`,
/// `subscore_sx` plus `total` (`n`).
#[pyfunction]
fn subscore_analysis(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    groups: Vec<usize>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let flat = x.as_slice()?;
    // Validate BEFORE allocating rows: unchecked n_persons * n_items can
    // wrap on 64-bit (e.g. 2^63 * 2 == 0, matching an empty array) and then
    // panic with capacity overflow inside the row allocation.
    if n_persons < 3 || n_items < 4 || groups.len() != n_items {
        return Err(PyValueError::new_err(
            "need n_persons >= 3, n_items >= 4, and one group index per item",
        ));
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| PyValueError::new_err("n_persons * n_items overflows"))?;
    if flat.len() != expected {
        return Err(PyValueError::new_err(format!(
            "x has {} entries, expected n_persons * n_items = {expected}",
            flat.len(),
        )));
    }
    let rows: Vec<Vec<f64>> = (0..n_persons)
        .map(|i| flat[i * n_items..(i + 1) * n_items].to_vec())
        .collect();
    let res = core_subscores(&rows, &groups).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha", res.alpha)?;
    out.set_item("alpha_total", res.alpha_total)?;
    out.set_item("prmse_s", res.prmse_s)?;
    out.set_item("prmse_x", res.prmse_x)?;
    out.set_item("prmse_sx", res.prmse_sx)?;
    out.set_item("tau", res.tau)?;
    out.set_item("beta", res.beta)?;
    out.set_item("gamma", res.gamma)?;
    out.set_item("added_value_s", res.added_value_s)?;
    out.set_item("added_value_sx", res.added_value_sx)?;
    out.set_item("total", res.total)?;
    let flatten = |m: Vec<Vec<f64>>| -> Vec<f64> { m.into_iter().flatten().collect() };
    out.set_item("corr", flatten(res.corr))?;
    out.set_item("disattenuated_corr", flatten(res.disattenuated_corr))?;
    out.set_item("observed", flatten(res.observed))?;
    out.set_item("subscore_s", flatten(res.subscore_s))?;
    out.set_item("subscore_x", flatten(res.subscore_x))?;
    out.set_item("subscore_sx", flatten(res.subscore_sx))?;
    Ok(out.into())
}

/// Confirmatory DETECT dimensionality analysis (Zhang & Stout, 1999, as
/// implemented by CRAN sirt's sum-score `scale_score=FALSE` path). `x` is a
/// flattened row-major `n_persons * n_items` binary (0/1, no missing)
/// response matrix; `cluster[j]` is the opaque integer cluster label of item
/// `j`. Returns a dict with `detect`, `assi`, `ratio`, `madcov100`,
/// `mcov100`, `n_pairs`, and the per-pair `pair_i`, `pair_j`, `ccov`.
#[pyfunction]
fn detect_analysis(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    cluster: Vec<i64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let flat = x.as_slice()?;
    // Validate BEFORE any allocation: unchecked n_persons * n_items can wrap
    // on 64-bit and then panic with capacity overflow.
    if n_persons < 2 || n_items < 2 || cluster.len() != n_items {
        return Err(PyValueError::new_err(
            "need n_persons >= 2, n_items >= 2, and one cluster label per item",
        ));
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| PyValueError::new_err("n_persons * n_items overflows"))?;
    if flat.len() != expected {
        return Err(PyValueError::new_err(format!(
            "x has {} entries, expected n_persons * n_items = {expected}",
            flat.len(),
        )));
    }
    let res =
        core_detect_analysis(flat, n_persons, n_items, &cluster).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("detect", res.detect)?;
    out.set_item("assi", res.assi)?;
    out.set_item("ratio", res.ratio)?;
    out.set_item("madcov100", res.madcov100)?;
    out.set_item("mcov100", res.mcov100)?;
    out.set_item("n_pairs", res.n_pairs)?;
    out.set_item("pair_i", res.pair_i)?;
    out.set_item("pair_j", res.pair_j)?;
    out.set_item("ccov", res.ccov)?;
    Ok(out.into())
}

/// Confirmatory Stout-style DIMTEST statistic of essential unidimensionality
/// (`mlsirm_core::detect::dimtest`).
///
/// Formulas transcribed from Nandakumar & Stout's 1992 ERIC technical-report
/// version (ED351383) of "Refinements of Stout's Procedure for Assessing
/// Latent Trait Unidimensionality" (published 1993, *Journal of Educational
/// Statistics, 18*(1), 41-68), which describes Stout (1987, Sec. 4).
/// Kieftenbeld & Nandakumar (2015, PMC5978610) was READ for the original
/// second-AT bias correction vs. later bootstrap DIMTEST distinction.
/// NOT READ: Stout (1987) original Psychometrika article, Stout et al.
/// (2001), Froelich & Habing (2008), and DIM-Pack source code; Stout (1987)
/// is cited only as described by Nandakumar & Stout (1992/1993).
///
/// `x` is a flattened row-major `n_persons * n_items` binary (0/1, no
/// missing) response matrix; `at1`/`at2` are caller-supplied assessment
/// subtest item indices (equal length >= 4, disjoint); PT is the complement.
/// Persons are grouped by raw PT score; groups with fewer than 20 examinees
/// are discarded. Returns a dict with `t`, `t_l`, `t_b`, `p_value`
/// (one-sided upper tail), `groups_used`, `n_discarded`, and
/// `retained_pt_scores`.
#[pyfunction]
fn py_dimtest(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    at1: Vec<usize>,
    at2: Vec<usize>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let flat = x.as_slice()?;
    let res = core_dimtest(flat, n_persons, n_items, &at1, &at2).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("t", res.t)?;
    out.set_item("t_l", res.t_l)?;
    out.set_item("t_b", res.t_b)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("groups_used", res.groups_used)?;
    out.set_item("n_discarded", res.n_discarded)?;
    out.set_item("retained_pt_scores", res.retained_pt_scores)?;
    Ok(out.into())
}

/// Wollack-style omega answer-copying statistic
/// (`mlsirm_core::security::wollack_omega`).
///
/// Formula verified against two READ implementations: the CRAN CopyDetect
/// package R source (`similarity1.r`/`similarity2.r`, computing
/// `(obs - E) / sqrt(V)` with an upper-tail normal p) and the independent
/// `aberrance` package (`compute_OMG` in `detect-ac.R`/`compute.R`).
/// NOT READ: Wollack (1997, *Applied Psychological Measurement, 21*(4),
/// 307-320) original article (access blocked); it is cited only as
/// implemented by those sources. CopyDetect's printed documentation shows
/// the sign flipped (`(E - obs)/sqrt(V)`) but both source files use
/// `(obs - E)/sqrt(V)`; the source convention is implemented here.
///
/// `probs` is a flattened row-major `n_items * n_options` matrix of the
/// COPIER's model-implied option-response probabilities (each row summing
/// to 1); `copier`/`source` are observed option indices. Returns a dict
/// with `observed_matches`, `expected_matches`, `variance`, `omega`, and
/// upper-tail `p_value`.
#[pyfunction]
fn py_wollack_omega(
    py: Python<'_>,
    copier: Vec<usize>,
    source: Vec<usize>,
    probs: PyReadonlyArray1<'_, f64>,
    n_options: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_wollack_omega(&copier, &source, probs.as_slice()?, n_options)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("observed_matches", res.observed_matches)?;
    out.set_item("expected_matches", res.expected_matches)?;
    out.set_item("variance", res.variance)?;
    out.set_item("omega", res.omega)?;
    out.set_item("p_value", res.p_value)?;
    Ok(out.into())
}

/// K-index of matching incorrect answers
/// (`mlsirm_core::security::k_index`), a faithful port of the CRAN
/// CopyDetect package's internal `k()` (READ: `R/similarity1.r`,
/// corroborated by `R/similarity2.r`). NOT READ: Holland (1996, ETS
/// RR-96-07) and Sotaridona & Meijer (2002, *JEM, 39*(2), 115-132); the
/// K-index is cited only as implemented by CopyDetect. The subgroup is
/// every examinee whose number-incorrect equals the copier's — including
/// the copier itself and, when scores match, the source (CopyDetect
/// convention; the paper-style source exclusion is NOT applied).
///
/// `responses` is a flattened row-major `n_persons * n_items` scored 0/1
/// matrix (no missing data). Returns a dict with `wc`, `ws`, `m`,
/// `subgroup`, `emp_agg`, `p`, and the upper-tail `k_index`
/// `P(Bin(ws, p) >= m)`.
#[pyfunction]
fn py_k_index(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    copier: usize,
    source: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_k_index(responses.as_slice()?, n_persons, n_items, copier, source)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("wc", res.wc)?;
    out.set_item("ws", res.ws)?;
    out.set_item("m", res.m)?;
    out.set_item("subgroup", res.subgroup)?;
    out.set_item("emp_agg", res.emp_agg)?;
    out.set_item("p", res.p)?;
    out.set_item("k_index", res.k_index)?;
    Ok(out.into())
}

/// Generalized binomial test (GBT) tail kernel
/// (`mlsirm_core::security::gbt`), a faithful port of the CRAN aberrance
/// package's `compute_GBT` (READ: `src/compute.cpp`), corroborated by
/// CopyDetect's internal `GBT()` (READ: `R/similarity1.r`). NOT READ:
/// van der Linden & Sotaridona (2006, *JEBS, 31*(3), 283-304); GBT is
/// cited only as implemented by those packages. Probability construction
/// is the caller's job (aberrance directional or CopyDetect symmetric
/// recipe). Returns a dict with `observed_matches`, `match_dist` (exact
/// Poisson-binomial pmf, length n_items + 1), and the inclusive upper-tail
/// `p_value` `P(M >= observed_matches)`.
#[pyfunction]
fn py_gbt(
    py: Python<'_>,
    matches: PyReadonlyArray1<'_, f64>,
    match_probs: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_gbt(matches.as_slice()?, match_probs.as_slice()?).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("observed_matches", res.observed_matches)?;
    out.set_item("match_dist", PyArray1::from_slice(py, &res.match_dist))?;
    out.set_item("p_value", res.p_value)?;
    Ok(out.into())
}

/// K1/K2/S1/S2 answer-copying indices
/// (`mlsirm_core::security::k_variants`), a faithful port of the CRAN
/// CopyDetect package's internal `ks12()` (READ: `R/similarity1.r`),
/// specialized to complete scored 0/1 data. NOT READ: Sotaridona & Meijer
/// (2002, *JEM, 39*(2), 115-132) and (2003, *JEM, 40*(1), 53-69); all four
/// indices are cited only as implemented by CopyDetect. Number-incorrect
/// subgroups EXCLUDE the source (opposite of `py_k_index`'s base-`k()`
/// convention). Returns a dict with `wc`, `ws`, `m`, `mm`, `pr`, `pj`
/// (length n_items + 1, NaN at empty subgroups), the clamped/capped
/// predictions `p1`, `p2`, `s1`, `s2`, and the indices `k1`, `k2`,
/// `s1_index`, `s2_index` (small values suggest copying).
#[pyfunction]
fn py_k_variants(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    copier: usize,
    source: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_k_variants(responses.as_slice()?, n_persons, n_items, copier, source)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("wc", res.wc)?;
    out.set_item("ws", res.ws)?;
    out.set_item("m", res.m)?;
    out.set_item("mm", res.mm)?;
    out.set_item("pr", PyArray1::from_slice(py, &res.pr))?;
    out.set_item("pj", PyArray1::from_slice(py, &res.pj))?;
    out.set_item("p1", res.p1)?;
    out.set_item("p2", res.p2)?;
    out.set_item("s1", res.s1)?;
    out.set_item("s2", res.s2)?;
    out.set_item("k1", res.k1)?;
    out.set_item("k2", res.k2)?;
    out.set_item("s1_index", res.s1_index)?;
    out.set_item("s2_index", res.s2_index)?;
    Ok(out.into())
}

/// Hofstee compromise standard-setting cut score
/// (`mlsirm_core::standard_setting::hofstee`), a port of the
/// psychometricsGP R package's `fn_plot_hofstee()` computation (plotting
/// excluded). See the core module header for citation governance and the
/// reduced scope (collinear-overlap and zero-length diagonals rejected).
#[pyfunction]
fn py_hofstee(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    min_cut: f64,
    max_cut: f64,
    min_fail: f64,
    max_fail: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_hofstee(scores.as_slice()?, min_cut, max_cut, min_fail, max_fail)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("cut_score", res.cut_score)?;
    out.set_item("fail_rate", res.fail_rate)?;
    out.set_item("failed", res.failed)?;
    out.set_item(
        "cum_freq_percent",
        PyArray1::from_slice(py, &res.cum_freq_percent),
    )?;
    Ok(out.into())
}

/// Nonparametric person-fit statistics
/// (`mlsirm_core::personfit_np::person_fit_np`), a complete-data port of
/// the CRAN PerFit R package's G, Gnormed, NCI, U3, ZU3, C.Sato, and
/// Cstar (see the core module header for citation governance and the
/// perfect-row / NaN contract). `x` is a row-major complete
/// `n_persons * n_items` 0/1 response matrix.
#[pyfunction]
fn py_person_fit_np(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let flat = x.as_slice()?;
    if flat.len() != n_persons * n_items {
        return Err(PyValueError::new_err(format!(
            "x has {} entries, expected n_persons * n_items = {}",
            flat.len(),
            n_persons * n_items
        )));
    }
    let rows: Vec<Vec<f64>> = (0..n_persons)
        .map(|i| flat[i * n_items..(i + 1) * n_items].to_vec())
        .collect();
    let res = core_person_fit_np(&rows).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("g", PyArray1::from_slice(py, &res.g))?;
    out.set_item("gnormed", PyArray1::from_slice(py, &res.gnormed))?;
    out.set_item("nci", PyArray1::from_slice(py, &res.nci))?;
    out.set_item("u3", PyArray1::from_slice(py, &res.u3))?;
    out.set_item("zu3", PyArray1::from_slice(py, &res.zu3))?;
    out.set_item("c_sato", PyArray1::from_slice(py, &res.c_sato))?;
    out.set_item("cstar", PyArray1::from_slice(py, &res.cstar))?;
    Ok(out.into())
}

/// Angoff Delta plot DIF detection (`mlsirm_core::dif::delta_plot`), a
/// response-type-only port of the deltaPlotR R package (see the core module
/// section header for citation governance and reduced scope). `responses` is
/// a row-major `n_persons * n_items` 0/1 matrix (NaN = missing); `group` has
/// one 0 (reference) / 1 (focal) entry per person. `extreme` is
/// `("constraint", lo, hi)` or `("add", nr_add)`; `threshold` is
/// `("norm", alpha)` or `("fixed", thr)`; `purify` is None or one of
/// "IPP1"/"IPP2"/"IPP3". Item indices in `dif_items` are 0-based.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_delta_plot(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, f64>,
    group: PyReadonlyArray1<'_, u8>,
    n_persons: usize,
    n_items: usize,
    extreme_kind: &str,
    extreme_a: f64,
    extreme_b: f64,
    threshold_kind: &str,
    threshold_value: f64,
    purify: Option<&str>,
    max_iter: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let extreme = match extreme_kind {
        "constraint" => ExtremeAdjust::Constraint {
            lo: extreme_a,
            hi: extreme_b,
        },
        "add" => {
            if !(extreme_a.is_finite() && extreme_a >= 1.0 && extreme_a.fract() == 0.0) {
                return Err(PyValueError::new_err(
                    "nr_add must be a positive integer >= 1",
                ));
            }
            ExtremeAdjust::Add {
                nr_add: extreme_a as usize,
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown extreme adjustment '{other}' (use 'constraint' or 'add')"
            )))
        }
    };
    let threshold = match threshold_kind {
        "norm" => DeltaThreshold::Norm {
            alpha: threshold_value,
        },
        "fixed" => DeltaThreshold::Fixed(threshold_value),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown threshold '{other}' (use 'norm' or 'fixed')"
            )))
        }
    };
    let purify = match purify {
        None => None,
        Some("IPP1") => Some(DeltaPurifyType::Ipp1),
        Some("IPP2") => Some(DeltaPurifyType::Ipp2),
        Some("IPP3") => Some(DeltaPurifyType::Ipp3),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown purification '{other}' (use 'IPP1', 'IPP2', or 'IPP3')"
            )))
        }
    };
    let res = core_delta_plot(
        responses.as_slice()?,
        group.as_slice()?,
        n_persons,
        n_items,
        extreme,
        threshold,
        purify,
        max_iter,
    )
    .map_err(PyValueError::new_err)?;
    let flat2 = |v: &[[f64; 2]]| -> Vec<f64> { v.iter().flat_map(|r| [r[0], r[1]]).collect() };
    let out = pyo3::types::PyDict::new(py);
    out.set_item("props", PyArray1::from_slice(py, &flat2(&res.props)))?;
    out.set_item(
        "adj_props",
        PyArray1::from_slice(py, &flat2(&res.adj_props)),
    )?;
    out.set_item("deltas", PyArray1::from_slice(py, &flat2(&res.deltas)))?;
    let dist_flat: Vec<f64> = res.dist.iter().flatten().copied().collect();
    out.set_item("dist", PyArray1::from_slice(py, &dist_flat))?;
    out.set_item("axis_par", PyArray1::from_slice(py, &flat2(&res.axis_par)))?;
    out.set_item("thresholds", PyArray1::from_slice(py, &res.thresholds))?;
    out.set_item("dif_items", res.dif_items)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

/// Empirical Bayes Mantel-Haenszel DIF (`mlsirm_core::dif::eb_mh_dif`;
/// Zwick & Thayer, 2003, ERIC ED481063 — see the core section header for
/// citation governance and implementation choices). Takes per-item MH D-DIF
/// statistics and standard errors on the ETS delta scale; returns prior
/// estimates, shrinkage weights, posterior means/variances, and the five
/// ETS category probabilities flattened row-major (`n_items * 5`, columns
/// `[C-, B-, A, B+, C+]`).
#[pyfunction]
fn py_eb_mh_dif(
    py: Python<'_>,
    mh: PyReadonlyArray1<'_, f64>,
    se: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_eb_mh_dif(mh.as_slice()?, se.as_slice()?).map_err(PyValueError::new_err)?;
    let probs_flat: Vec<f64> = res.cat_probs.iter().flatten().copied().collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("mu", res.mu)?;
    out.set_item("tau2", res.tau2)?;
    out.set_item("tau2_raw", res.tau2_raw)?;
    out.set_item("weight", PyArray1::from_slice(py, &res.weight))?;
    out.set_item("post_mean", PyArray1::from_slice(py, &res.post_mean))?;
    out.set_item("post_var", PyArray1::from_slice(py, &res.post_var))?;
    out.set_item("cat_probs", PyArray1::from_slice(py, &probs_flat))?;
    Ok(out.into())
}

/// Mantel (1963) polytomous DIF chi-square + standardized mean difference
/// (`mlsirm_core::dif::mantel_smd_dif`; Zwick, Donoghue & Grima, 1993,
/// ERIC ED386493 — see the core section header for citation governance and
/// the documented SMD used-strata renormalization deviation). Takes
/// row-major ordinal integer scores (`n_persons * n_items`) and 0/1 group
/// labels; returns per-item vectors.
#[pyfunction]
fn py_mantel_smd_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, u8>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let rows = core_mantel_smd_dif(y.as_slice()?, group.as_slice()?, n_persons, n_items)
        .map_err(PyValueError::new_err)?;
    let chi2: Vec<f64> = rows.iter().map(|r| r.chi2).collect();
    let p_value: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let smd: Vec<f64> = rows.iter().map(|r| r.smd).collect();
    let used: Vec<f64> = rows.iter().map(|r| r.n_strata_used as f64).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("chi2", PyArray1::from_slice(py, &chi2))?;
    out.set_item("p_value", PyArray1::from_slice(py, &p_value))?;
    out.set_item("smd", PyArray1::from_slice(py, &smd))?;
    out.set_item("n_strata_used", PyArray1::from_slice(py, &used))?;
    Ok(out.into())
}

/// Generalized Mantel-Haenszel nominal DIF statistic
/// (`mlsirm_core::dif::gmh_dif`; Zwick, Donoghue & Grima, 1993, Eq. 10,
/// ERIC ED386493 — see the core section header for citation governance,
/// the reference-group A_k convention, and the effective-category / df
/// contract). Takes row-major non-negative integer category codes
/// (`n_persons * n_items`) and 0/1 group labels; returns per-item vectors.
#[pyfunction]
fn py_gmh_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, u8>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let rows = core_gmh_dif(y.as_slice()?, group.as_slice()?, n_persons, n_items)
        .map_err(PyValueError::new_err)?;
    let chi2: Vec<f64> = rows.iter().map(|r| r.chi2).collect();
    let p_value: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let df: Vec<f64> = rows.iter().map(|r| r.df as f64).collect();
    let used: Vec<f64> = rows.iter().map(|r| r.n_strata_used as f64).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("chi2", PyArray1::from_slice(py, &chi2))?;
    out.set_item("p_value", PyArray1::from_slice(py, &p_value))?;
    out.set_item("df", PyArray1::from_slice(py, &df))?;
    out.set_item("n_strata_used", PyArray1::from_slice(py, &used))?;
    Ok(out.into())
}

/// Breslow-Day (1980, Eq. 4.30) odds-ratio homogeneity test per item
/// (`mlsirm_core::dif::breslow_day_dif`) — the classical NON-UNIFORM DIF
/// companion to `mantel_haenszel_dif`: MH tests a common odds ratio against
/// 1, this tests whether a common odds ratio is tenable at all. Same input
/// conventions as `mantel_haenszel_dif` (row-major 0/1 `y`, 0/1 `group`);
/// the plugged-in common odds ratio is the crate's MH `alpha_mh` (see the
/// core citation-governance header for what was and was not read).
#[pyfunction]
fn py_breslow_day_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, u8>,
    group: PyReadonlyArray1<'_, u8>,
    n_persons: usize,
    n_items: usize,
    exclude_studied_item: bool,
    fdr_q: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let cfg = MhDifConfig {
        exclude_studied_item,
        fdr_q,
    };
    let rows = core_breslow_day_dif(y.as_slice()?, group.as_slice()?, n_persons, n_items, &cfg)
        .map_err(PyValueError::new_err)?;
    let alpha: Vec<f64> = rows.iter().map(|r| r.alpha_mh).collect();
    let chi2: Vec<f64> = rows.iter().map(|r| r.chi2_bd).collect();
    let df: Vec<f64> = rows.iter().map(|r| r.df).collect();
    let p_value: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let used: Vec<f64> = rows.iter().map(|r| r.n_strata_used as f64).collect();
    let flagged: Vec<bool> = rows.iter().map(|r| r.flagged_bh).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha_mh", PyArray1::from_slice(py, &alpha))?;
    out.set_item("chi2", PyArray1::from_slice(py, &chi2))?;
    out.set_item("df", PyArray1::from_slice(py, &df))?;
    out.set_item("p_value", PyArray1::from_slice(py, &p_value))?;
    out.set_item("n_strata_used", PyArray1::from_slice(py, &used))?;
    out.set_item("flagged_bh", flagged)?;
    Ok(out.into())
}

fn classification_result_to_dict(
    py: Python<'_>,
    res: ClassificationResult,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("per_cut_accuracy", res.per_cut_accuracy)?;
    out.set_item("per_cut_consistency", res.per_cut_consistency)?;
    out.set_item("simultaneous_accuracy", res.simultaneous_accuracy)?;
    out.set_item("simultaneous_consistency", res.simultaneous_consistency)?;
    out.set_item("conditional_accuracy", res.conditional_accuracy)?;
    out.set_item("conditional_consistency", res.conditional_consistency)?;
    out.set_item(
        "conditional_simultaneous_accuracy",
        res.conditional_simultaneous_accuracy,
    )?;
    out.set_item(
        "conditional_simultaneous_consistency",
        res.conditional_simultaneous_consistency,
    )?;
    Ok(out.into())
}

/// Rudner (2001, 2005) normal-approximation classification accuracy and
/// consistency (`mlsirm_core::classification`). All slices are per
/// evaluation point except `cutscores` (strictly increasing theta cuts).
#[pyfunction]
fn rudner_classification(
    py: Python<'_>,
    theta: PyReadonlyArray1<'_, f64>,
    sem: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    cutscores: Vec<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_rudner_classification(
        theta.as_slice()?,
        sem.as_slice()?,
        weights.as_slice()?,
        &cutscores,
    )
    .map_err(PyValueError::new_err)?;
    classification_result_to_dict(py, res)
}

/// Lee (2010, as implemented in CRAN cacIRT) summed-score classification
/// accuracy and consistency (`mlsirm_core::classification`). `probs` is a
/// flattened row-major `n_points * n_items` matrix of correct-response
/// probabilities strictly inside (0, 1); `cutscores` are raw-score cuts.
#[pyfunction]
fn lee_classification(
    py: Python<'_>,
    probs: PyReadonlyArray1<'_, f64>,
    n_points: usize,
    n_items: usize,
    weights: PyReadonlyArray1<'_, f64>,
    cutscores: Vec<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_lee_classification(
        probs.as_slice()?,
        n_points,
        n_items,
        weights.as_slice()?,
        &cutscores,
    )
    .map_err(PyValueError::new_err)?;
    classification_result_to_dict(py, res)
}

/// Livingston & Lewis (1995, as implemented in CRAN betafunctions 1.9.0
/// `LL.CA`) classification accuracy and consistency from a single test
/// administration (`mlsirm_core::classification`). `scores` are raw
/// observed scores in `[min_score, max_score]`; `reliability` is in (0, 1)
/// and `cut` is strictly inside `(min_score, max_score)`. Pass = observed
/// score >= cut.
#[pyfunction]
fn livingston_lewis(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    reliability: f64,
    min_score: f64,
    max_score: f64,
    cut: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_livingston_lewis(scores.as_slice()?, reliability, min_score, max_score, cut)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("effective_test_length", res.effective_test_length)?;
    out.set_item("etl_rounded", res.etl_rounded)?;
    out.set_item("lower", res.lower)?;
    out.set_item("upper", res.upper)?;
    out.set_item("alpha", res.alpha)?;
    out.set_item("beta", res.beta)?;
    out.set_item("used_two_parameter", res.used_two_parameter)?;
    out.set_item("p_tp", res.p_tp)?;
    out.set_item("p_fp", res.p_fp)?;
    out.set_item("p_tf", res.p_tf)?;
    out.set_item("p_ff", res.p_ff)?;
    out.set_item("accuracy", res.accuracy)?;
    out.set_item("sensitivity", res.sensitivity)?;
    out.set_item("specificity", res.specificity)?;
    out.set_item("p_ii", res.p_ii)?;
    out.set_item("p_ij", res.p_ij)?;
    out.set_item("p_ji", res.p_ji)?;
    out.set_item("p_jj", res.p_jj)?;
    out.set_item("consistency", res.consistency)?;
    out.set_item("chance_consistency", res.chance_consistency)?;
    out.set_item("kappa", res.kappa)?;
    Ok(out.into())
}

fn hanson_brennan_result_to_dict(
    py: Python<'_>,
    res: &HansonBrennanResult,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("lords_k", res.lords_k)?;
    out.set_item("true_score_moments", res.true_score_moments.to_vec())?;
    out.set_item("lower", res.lower)?;
    out.set_item("upper", res.upper)?;
    out.set_item("alpha", res.alpha)?;
    out.set_item("beta", res.beta)?;
    out.set_item("used_two_parameter", res.used_two_parameter)?;
    out.set_item("p_tp", res.p_tp)?;
    out.set_item("p_fp", res.p_fp)?;
    out.set_item("p_tf", res.p_tf)?;
    out.set_item("p_ff", res.p_ff)?;
    out.set_item("accuracy", res.accuracy)?;
    out.set_item("sensitivity", res.sensitivity)?;
    out.set_item("specificity", res.specificity)?;
    out.set_item("p_ii", res.p_ii)?;
    out.set_item("p_ij", res.p_ij)?;
    out.set_item("p_ji", res.p_ji)?;
    out.set_item("p_jj", res.p_jj)?;
    out.set_item("consistency", res.consistency)?;
    out.set_item("chance_consistency", res.chance_consistency)?;
    out.set_item("kappa", res.kappa)?;
    Ok(out.into())
}

/// Hanson-Brennan (Hanson, 1991, ACT RR 91-5; CRAN betafunctions 1.9.0
/// `HB.CA`) classification accuracy and consistency under the
/// four-parameter beta compound binomial model, from raw number-correct
/// scores (`mlsirm_core::classification`). Pass = observed score >= `cut`
/// (pass-positive; betafunctions labels fail as positive, so its
/// sensitivity is this function's specificity).
#[pyfunction]
fn hanson_brennan(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    n_items: usize,
    reliability: f64,
    cut: usize,
    two_parameter: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_hanson_brennan(scores.as_slice()?, n_items, reliability, cut, two_parameter)
        .map_err(PyValueError::new_err)?;
    hanson_brennan_result_to_dict(py, &res)
}

/// Hanson-Brennan classification indexes from fixed model parameters:
/// Lord's k plus a four-parameter beta true-score distribution
/// (`mlsirm_core::classification`; Hanson, 1991).
#[pyfunction]
fn hanson_brennan_from_params(
    py: Python<'_>,
    n_items: usize,
    lords_k: f64,
    lower: f64,
    upper: f64,
    alpha: f64,
    beta: f64,
    cut: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_hanson_brennan_from_params(n_items, lords_k, lower, upper, alpha, beta, cut)
        .map_err(PyValueError::new_err)?;
    hanson_brennan_result_to_dict(py, &res)
}

/// Subkoviak (1976, ERIC ED120229) single-administration coefficient of
/// agreement for mastery classifications under the simple binomial
/// true-score model (`mlsirm_core::classification`). `alpha = None`
/// derives KR-21 with the population (ddof = 0) variance.
#[pyfunction]
#[pyo3(signature = (scores, n_items, cuts, alpha=None))]
fn subkoviak_agreement(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    n_items: usize,
    cuts: PyReadonlyArray1<'_, f64>,
    alpha: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_subkoviak_agreement(scores.as_slice()?, n_items, cuts.as_slice()?, alpha)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha", res.alpha)?;
    out.set_item("p_hat", PyArray1::from_slice(py, &res.p_hat))?;
    out.set_item("per_person", PyArray1::from_slice(py, &res.per_person))?;
    out.set_item("agreement", res.agreement)?;
    out.set_item("chance_agreement", res.chance_agreement)?;
    out.set_item("kappa", res.kappa)?;
    Ok(out.into())
}

/// Livingston (1972, ERIC ED069624) criterion-referenced reliability `k^2`
/// with Spearman-Brown projections (`mlsirm_core::classification`).
#[pyfunction]
fn livingston_k2(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    cut: f64,
    reliability: f64,
    n_lengths: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_livingston_k2(scores.as_slice()?, cut, reliability, n_lengths.as_slice()?)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("mean", res.mean)?;
    out.set_item("var", res.var)?;
    out.set_item("msd", res.msd)?;
    out.set_item("k2", PyArray1::from_slice(py, &res.k2))?;
    Ok(out.into())
}

/// Livingston (1972, ERIC ED069624) criterion-referenced correlation
/// `k(X, Y)` (`mlsirm_core::classification`).
#[pyfunction]
fn livingston_correlation(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    cut_x: f64,
    cut_y: f64,
) -> PyResult<f64> {
    core_livingston_correlation(x.as_slice()?, y.as_slice()?, cut_x, cut_y)
        .map_err(PyValueError::new_err)
}

fn ws_result_to_dict(
    py: Python<'_>,
    res: WoodruffSawyerResult,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("pass_rate", res.pass_rate)?;
    out.set_item("phi_half", res.phi_half)?;
    out.set_item("theta_half", res.theta_half)?;
    out.set_item("phi", res.phi)?;
    out.set_item("theta", res.theta)?;
    out.set_item("pi00", res.pi00)?;
    out.set_item("pi01", res.pi01)?;
    out.set_item("pi11", res.pi11)?;
    Ok(out.into())
}

/// Woodruff & Sawyer (1988, ERIC ED292877) split-half / Spearman-Brown
/// pass-fail reliability from a 2x2 half-test table
/// (`mlsirm_core::classification`).
#[pyfunction]
fn woodruff_sawyer_sb(
    py: Python<'_>,
    counts: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_woodruff_sawyer_sb(counts.as_slice()?).map_err(PyValueError::new_err)?;
    ws_result_to_dict(py, res)
}

/// Woodruff & Sawyer (1988, ERIC ED292877) bivariate-normal pass-fail
/// reliability from a half-test correlation (`mlsirm_core::classification`).
#[pyfunction]
fn woodruff_sawyer_normal(
    py: Python<'_>,
    mean: f64,
    sd: f64,
    cut: f64,
    r_half: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_woodruff_sawyer_normal(mean, sd, cut, r_half).map_err(PyValueError::new_err)?;
    ws_result_to_dict(py, res)
}

/// One-facet crossed `p x i` generalizability analysis
/// (`mlsirm_core::gtheory`; Huebner & Lucht, 2019, Tables 3-4). `x` is a
/// flattened row-major `n_p x n_i` score matrix; `n_i_prime` lists the
/// proposed D-study item counts.
#[pyfunction]
fn gtheory_pi(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_p: usize,
    n_i: usize,
    n_i_prime: Vec<usize>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_gtheory_pi(x.as_slice()?, n_p, n_i, &n_i_prime).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("df", res.df.to_vec())?;
    out.set_item("ss", res.ss.to_vec())?;
    out.set_item("ms", res.ms.to_vec())?;
    out.set_item("var_raw", res.var_raw.to_vec())?;
    out.set_item("var", res.var.to_vec())?;
    out.set_item("d_study", d_study_rows_to_py(py, &res.d_study)?)?;
    Ok(out.into())
}

/// Brennan-Kane index of dependability `Phi(lambda)` for mastery tests
/// (`mlsirm_core::gtheory`; Kane & Brennan, 1977, ACT TB-28, eq. 33 with
/// a derived unbiased signal estimator — see the core doc comment).
#[pyfunction]
fn phi_lambda(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_p: usize,
    n_i: usize,
    lambda: f64,
    n_i_prime: Vec<usize>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_phi_lambda(x.as_slice()?, n_p, n_i, lambda, &n_i_prime)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("grand_mean", res.grand_mean)?;
    out.set_item("var", res.var.to_vec())?;
    out.set_item("var_xbar", res.var_xbar)?;
    out.set_item("signal", res.signal)?;
    out.set_item("phi", PyArray1::from_slice(py, &res.phi))?;
    Ok(out.into())
}

/// Two-facet crossed `p x i x o` generalizability analysis
/// (`mlsirm_core::gtheory`; Huebner & Lucht, 2019, Tables 5-6). `x` is
/// flattened `x[p*n_i*n_o + i*n_o + o]`; `n_prime` lists proposed
/// `(n_i', n_o')` D-study pairs. Component order in all arrays:
/// (p, i, o, pi, po, io, pio).
#[pyfunction]
fn gtheory_pio(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    n_p: usize,
    n_i: usize,
    n_o: usize,
    n_prime: Vec<(usize, usize)>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_gtheory_pio(x.as_slice()?, n_p, n_i, n_o, &n_prime).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("df", res.df.to_vec())?;
    out.set_item("ss", res.ss.to_vec())?;
    out.set_item("ms", res.ms.to_vec())?;
    out.set_item("var_raw", res.var_raw.to_vec())?;
    out.set_item("var", res.var.to_vec())?;
    out.set_item("d_study", d_study_rows_to_py(py, &res.d_study)?)?;
    Ok(out.into())
}

fn d_study_rows_to_py(
    py: Python<'_>,
    rows: &[GTheoryDStudyRow],
) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    rows.iter()
        .map(|r| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("n_i_prime", r.n_i_prime)?;
            d.set_item("n_o_prime", r.n_o_prime)?;
            d.set_item("rel_error_var", r.rel_error_var)?;
            d.set_item("abs_error_var", r.abs_error_var)?;
            d.set_item("generalizability", r.generalizability)?;
            d.set_item("dependability", r.dependability)?;
            Ok(d.into())
        })
        .collect()
}

fn minres_fa_to_py(py: Python<'_>, res: &MinresFaResult) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("loadings", res.loadings.to_vec())?;
    out.set_item("uniquenesses", res.uniquenesses.to_vec())?;
    out.set_item("communalities", res.communalities.to_vec())?;
    out.set_item("objective", res.objective)?;
    out.set_item("kkt_violation", res.kkt_violation)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

/// Minres (ULS) exploratory factor analysis (`mlsirm_core::factor`;
/// transcription of psych fa.R, fm = "minres"). `corr` is a flattened
/// row-major `p x p` correlation matrix. `loadings` in the result dict is
/// flattened row-major `p x n_factors` (unrotated, column sums >= 0).
#[pyfunction]
fn minres_fa(
    py: Python<'_>,
    corr: PyReadonlyArray1<'_, f64>,
    p: usize,
    n_factors: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_minres_fa_corr(corr.as_slice()?, p, n_factors).map_err(PyValueError::new_err)?;
    minres_fa_to_py(py, &res)
}

/// [`minres_fa`] from raw data (`n x p` flattened row-major; Pearson
/// correlations computed internally, complete data required).
#[pyfunction]
fn minres_fa_from_data(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n: usize,
    p: usize,
    n_factors: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_minres_fa_data(data.as_slice()?, n, p, n_factors).map_err(PyValueError::new_err)?;
    minres_fa_to_py(py, &res)
}

/// McDonald's omega_total for the 1-factor case from a correlation matrix
/// (`mlsirm_core::factor`): `(sum lambda)^2 / ((sum lambda)^2 + sum psi)`
/// on a 1-factor minres fit. Returns the omega plus the embedded fit dict.
#[pyfunction]
fn omega_total_1f(
    py: Python<'_>,
    corr: PyReadonlyArray1<'_, f64>,
    p: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_omega_total_1f_corr(corr.as_slice()?, p).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("omega_total", res.omega_total)?;
    out.set_item("fa", minres_fa_to_py(py, &res.fa)?)?;
    Ok(out.into())
}

/// [`omega_total_1f`] from raw data (`n x p` flattened row-major).
#[pyfunction]
fn omega_total_1f_from_data(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n: usize,
    p: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_omega_total_1f_data(data.as_slice()?, n, p).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("omega_total", res.omega_total)?;
    out.set_item("fa", minres_fa_to_py(py, &res.fa)?)?;
    Ok(out.into())
}

/// Factor-analytic greatest lower bound to reliability (psych glb.fa
/// transcription in `mlsirm_core::factor::glb_fa_corr`; NOT the algebraic
/// glb, which needs an SDP solver). Returns {glb, communalities, nf}.
#[pyfunction]
fn glb_fa(
    py: Python<'_>,
    corr: PyReadonlyArray1<'_, f64>,
    p: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_glb_fa_corr(corr.as_slice()?, p).map_err(PyValueError::new_err)?;
    glb_fa_to_py(py, &res)
}

/// [`glb_fa`] from raw data (`n x p` flattened row-major, complete data).
#[pyfunction]
fn glb_fa_from_data(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n: usize,
    p: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_glb_fa_data(data.as_slice()?, n, p).map_err(PyValueError::new_err)?;
    glb_fa_to_py(py, &res)
}

fn glb_fa_to_py(
    py: Python<'_>,
    res: &mlsirm_core::factor::GlbFaResult,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("glb", res.glb)?;
    out.set_item(
        "communalities",
        numpy::PyArray1::from_slice(py, &res.communalities),
    )?;
    out.set_item("nf", res.nf)?;
    Ok(out.into())
}

/// Velicer's minimum average partial (MAP) test
/// (`mlsirm_core::factor::velicer_map_corr`). `corr` is a flattened
/// row-major `p x p` correlation matrix; rows `m = 0..=max_m`. Returns
/// {f2, f4, retained_f2, retained_f4}; invalid rows (singular partial
/// covariance normalization, e.g. identity R for m >= 1) are NaN and
/// excluded from the retained-count argmin.
///
/// Retained count follows O'Connor's canonical programs (`m` at the
/// minimum, with `m = 0` = the unpartialed baseline); note
/// `fungible::faMAP` prints a 1-based row position (off by one). The
/// fourth-power (revised) criterion uses ELEMENTWISE fourth powers per
/// O'Connor's code; see the core docs for the unresolved
/// `EFA.dimensions` matrix-power conflict.
///
/// References (APA 7th ed.):
///
/// Velicer, W. F. (1976). Determining the number of components from the
///   matrix of partial correlations. *Psychometrika, 41*(3), 321-327.
///   https://doi.org/10.1007/BF02293557 (Not read; formula support is the
///   read O'Connor map.m/map.sps programs and psych VSS.R map().)
/// O'Connor, B. P. (2000). SPSS and SAS programs for determining the
///   number of components using parallel analysis and Velicer's MAP test.
///   *Behavior Research Methods, Instruments, & Computers, 32*(3),
///   396-402. https://doi.org/10.3758/BF03200807 (Programs read; paper
///   not read.)
/// Velicer, W. F., Eaton, C. A., & Fava, J. L. (2000). Construct
///   explication through factor or component analysis. In R. D. Goffin &
///   E. Helmes (Eds.), *Problems and solutions in human assessment*
///   (pp. 41-71). Kluwer. (Not read; fourth-power origin per O'Connor's
///   code comments.)
/// Revelle, W. (2025). *psych: Procedures for psychological,
///   psychometric, and personality research* (Version 2.6.5) [R package].
///   https://CRAN.R-project.org/package=psych (VSS.R map() read.)
#[pyfunction]
fn velicer_map(
    py: Python<'_>,
    corr: PyReadonlyArray1<'_, f64>,
    p: usize,
    max_m: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_velicer_map_corr(corr.as_slice()?, p, max_m).map_err(PyValueError::new_err)?;
    velicer_map_to_py(py, &res)
}

/// [`velicer_map`] from raw data (`n x p` flattened row-major, complete
/// data; Pearson correlations computed internally).
#[pyfunction]
fn velicer_map_from_data(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n: usize,
    p: usize,
    max_m: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_velicer_map_data(data.as_slice()?, n, p, max_m).map_err(PyValueError::new_err)?;
    velicer_map_to_py(py, &res)
}

fn velicer_map_to_py(
    py: Python<'_>,
    res: &mlsirm_core::factor::VelicerMapResult,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("f2", numpy::PyArray1::from_slice(py, &res.f2))?;
    out.set_item("f4", numpy::PyArray1::from_slice(py, &res.f4))?;
    out.set_item("retained_f2", res.retained_f2)?;
    out.set_item("retained_f4", res.retained_f4)?;
    Ok(out.into())
}

/// Brogden-Cronbach-Gleser selection utility with Naylor-Shine selected-group
/// mean (`mlsirm_core::utility::selection_utility`; verified against the CRAN
/// iopsych 0.90.1 utilityBcg/ux source and a scipy oracle). Returns
/// {xc, ux, pux, utility_gain}.
#[pyfunction]
fn selection_utility(
    py: Python<'_>,
    n: f64,
    sdy: f64,
    rxy: f64,
    sr: f64,
    cost_total: f64,
    period: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_selection_utility(n, sdy, rxy, sr, cost_total, period)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("xc", res.xc)?;
    out.set_item("ux", res.ux)?;
    out.set_item("pux", res.pux)?;
    out.set_item("utility_gain", res.utility_gain)?;
    Ok(out.into())
}

/// Taylor-Russell (1939) success ratio under the standard bivariate-normal
/// selection model (`mlsirm_core::utility::taylor_russell`). Returns
/// {success_ratio, base_rate, q_joint}.
#[pyfunction]
fn taylor_russell(py: Python<'_>, rxy: f64, sr: f64, br: f64) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_taylor_russell(rxy, sr, br).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("success_ratio", res.success_ratio)?;
    out.set_item("base_rate", res.base_rate)?;
    out.set_item("q_joint", res.q_joint)?;
    Ok(out.into())
}

/// Sympson-Hetter item-exposure-control calibration for max-info CAT
/// (`mlsirm_core::exposure::sympson_hetter`; algorithm confirmed against
/// Georgiadou, Triantafillou, & Economides, 2007, and Barrada, Olea, &
/// Ponsoda, 2007, Eq. 1-3). Returns {k, exposure, selection, max_exposure,
/// n_iter, converged, history_max_exposure}.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_sympson_hetter(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    r_max: f64,
    test_length: usize,
    n_simulees: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let cfg = SympsonHetterConfig {
        r_max,
        test_length,
        n_simulees,
        max_iter,
        tol,
        seed,
        q_theta,
    };
    let res = core_sympson_hetter(a.as_slice()?, b.as_slice()?, c.as_slice()?, &cfg)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("k", numpy::PyArray1::from_slice(py, &res.k))?;
    out.set_item("exposure", numpy::PyArray1::from_slice(py, &res.exposure))?;
    out.set_item("selection", numpy::PyArray1::from_slice(py, &res.selection))?;
    out.set_item("max_exposure", res.max_exposure)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item(
        "history_max_exposure",
        numpy::PyArray1::from_slice(py, &res.history_max_exposure),
    )?;
    Ok(out.into())
}

/// a-stratified multistage CAT simulation
/// (`mlsirm_core::exposure::a_stratified`).
/// Returns {exposure, max_exposure, stratum, stage_lengths, theta_rmse,
/// theta_bias}.
///
/// Source status: design after Chang & Ying (1999), cited from its
/// abstract only; the b-matching selection rule and ascending-a strata are
/// confirmed from Barrada, Mazuela, & Olea (2006), read in full. Full
/// references and repository-choice labels are in the core rustdoc
/// (`mlsirm_core::exposure::a_stratified`) and the Python docstring
/// (`fast_mlsirm.exposure.a_stratified`).
///
/// References:
/// Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information
/// stratification method for controlling item exposure in computerized
/// adaptive testing. Psicothema, 18(1), 156-159.
/// Chang, H.-H., & Ying, Z. (1999). a-stratified multistage computerized
/// adaptive testing. Applied Psychological Measurement, 23(3), 211-222.
/// https://doi.org/10.1177/01466219922031338
#[pyfunction]
fn py_a_stratified(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    n_strata: usize,
    test_length: usize,
    n_simulees: usize,
    seed: u64,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let cfg = AStratifiedConfig {
        n_strata,
        test_length,
        n_simulees,
        seed,
        q_theta,
    };
    let res = core_a_stratified(a.as_slice()?, b.as_slice()?, c.as_slice()?, &cfg)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("exposure", numpy::PyArray1::from_slice(py, &res.exposure))?;
    out.set_item("max_exposure", res.max_exposure)?;
    out.set_item("stratum", res.stratum)?;
    out.set_item("stage_lengths", res.stage_lengths)?;
    out.set_item("theta_rmse", res.theta_rmse)?;
    out.set_item("theta_bias", res.theta_bias)?;
    Ok(out.into())
}

/// Chang-Ying (1996) Kullback-Leibler information index and CAT item
/// selection (`mlsirm_core::exposure::{kl_information, kl_select}`).
/// `kl_information` returns the UNNORMALIZED area of the pointwise Bernoulli
/// KL divergence over `[theta0 - delta, theta0 + delta]`; `kl_select` uses
/// `delta = r / sqrt(n_administered)` (requires `n_administered >= 1`) and
/// returns {index, selected, delta}. Full sources and the contract are in
/// the core rustdoc.
///
/// References:
/// Chang, H.-H., & Ying, Z. (1996). A global information approach to
/// computerized adaptive testing. Applied Psychological Measurement, 20(3),
/// 213-229. https://doi.org/10.1177/014662169602000303
#[pyfunction]
fn py_kl_information(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    theta0: f64,
    delta: f64,
) -> PyResult<Py<numpy::PyArray1<f64>>> {
    let v = core_kl_information(a.as_slice()?, b.as_slice()?, c.as_slice()?, theta0, delta)
        .map_err(PyValueError::new_err)?;
    Ok(numpy::PyArray1::from_slice(py, &v).into())
}

/// See `py_kl_information`; `administered` is a boolean mask over the pool.
#[pyfunction]
fn py_kl_select(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, bool>,
    theta0: f64,
    n_administered: usize,
    r: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_kl_select(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        administered.as_slice()?,
        theta0,
        n_administered,
        r,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("index", numpy::PyArray1::from_slice(py, &res.index))?;
    out.set_item("selected", res.selected)?;
    out.set_item("delta", res.delta)?;
    Ok(out.into())
}

/// Owen (1975) approximate Bayesian single-item posterior update for the
/// 3PNO model (`mlsirm_core::exposure::owen_update`). Returns the updated
/// `(mu, sig2)` normal-approximation posterior moments. Owen (1975) itself
/// was NOT read (paywalled); the formulas are implemented as reproduced by
/// van der Linden (1998, Appendix Eqs. A.1-A.6) and cross-checked against
/// the R `irt` package `src/est_ability_owen.cpp`. See the core rustdoc for
/// the full citation-governance note.
///
/// References:
/// Owen, R. J. (1975). A Bayesian sequential procedure for quantal response
/// in the context of adaptive mental testing. Journal of the American
/// Statistical Association, 70(350), 351-356.
/// https://doi.org/10.1080/01621459.1975.10479871
/// van der Linden, W. J. (1998). Bayesian item selection criteria for
/// adaptive testing (Research Report 96-01). University of Twente.
#[pyfunction]
fn py_owen_update(
    a: f64,
    b: f64,
    c: f64,
    correct: bool,
    mu: f64,
    sig2: f64,
) -> PyResult<(f64, f64)> {
    core_owen_update(a, b, c, correct, mu, sig2).map_err(PyValueError::new_err)
}

/// Owen (1975) sequential CAT driver (`mlsirm_core::exposure::owen_cat`):
/// b-matching selection (argmin |b_i - mu|, ties to the lowest index),
/// Owen posterior updates, optional posterior-variance stopping. `responses`
/// is a full-pool 0/1 vector consulted for whichever item is selected.
/// Returns {administered, mu_trace, sig2_trace, mu, sig2}.
#[pyfunction]
#[pyo3(signature = (a, b, c, responses, mu0, sig2_0, test_length, sig2_stop=None))]
#[allow(clippy::too_many_arguments)]
fn py_owen_cat(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    responses: PyReadonlyArray1<'_, u8>,
    mu0: f64,
    sig2_0: f64,
    test_length: usize,
    sig2_stop: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_owen_cat(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        responses.as_slice()?,
        mu0,
        sig2_0,
        test_length,
        sig2_stop,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("administered", res.administered)?;
    out.set_item("mu_trace", numpy::PyArray1::from_slice(py, &res.mu_trace))?;
    out.set_item(
        "sig2_trace",
        numpy::PyArray1::from_slice(py, &res.sig2_trace),
    )?;
    out.set_item("mu", res.mu)?;
    out.set_item("sig2", res.sig2)?;
    Ok(out.into())
}

/// Kingsbury & Zara (1989) constrained CAT (CCAT) content-balanced item
/// selection (`mlsirm_core::exposure::ccat_select`). The primary source was
/// NOT read (paywalled; https://doi.org/10.1207/s15324818ame0204_6); the rule
/// is implemented as reproduced by the R catR package (`nextItem.R`,
/// `cbControl`): any eligible group with zero administered items has
/// priority, otherwise the eligible group with the maximal
/// target-minus-empirical-proportion discrepancy wins; within the chosen
/// group the most informative unadministered item (logistic 3PL Fisher
/// information, no D constant) is selected. Ties go to the lowest index
/// (documented deviation from catR's random tie-break). Returns
/// {selected, group, discrepancy, info}.
///
/// References:
/// Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items
/// for computerized adaptive tests. Applied Measurement in Education, 2(4),
/// 359-375. https://doi.org/10.1207/s15324818ame0204_6
/// Magis, D., & Raiche, G. (2012). Random generation of response patterns
/// under computerized adaptive testing with the R package catR. Journal of
/// Statistical Software, 48(8), 1-31. https://doi.org/10.18637/jss.v048.i08
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_ccat_select(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    groups: PyReadonlyArray1<'_, usize>,
    targets: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, bool>,
    theta0: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_ccat_select(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        groups.as_slice()?,
        targets.as_slice()?,
        administered.as_slice()?,
        theta0,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("selected", res.selected)?;
    out.set_item("group", res.group)?;
    out.set_item(
        "discrepancy",
        numpy::PyArray1::from_slice(py, &res.discrepancy),
    )?;
    out.set_item("info", numpy::PyArray1::from_slice(py, &res.info))?;
    Ok(out.into())
}

/// Owen-approximate posterior-predictive expected posterior variance (EPV)
/// item selection (`mlsirm_core::exposure::epv_select`). This is NOT van der
/// Linden's (1998) exact MEPV criterion: the posterior is Owen's (1975) normal
/// approximation N(mu, sig2), the predictive probability is
/// p*_i = c_i + (1 - c_i) * Phi(a_i (mu - b_i) / sqrt(1 + a_i^2 sig2)), and
/// the outcome variances come from `owen_update` rather than exact numerical
/// posteriors. Selects the unadministered item minimizing
/// EPV_i = p*_i sig2_i^+ + (1 - p*_i) sig2_i^-; ties go to the lowest index.
/// Returns {selected, epv, predictive}.
///
/// References:
/// van der Linden, W. J. (1998). Bayesian item selection criteria for
/// adaptive testing. Psychometrika, 63(2), 201-216.
/// https://doi.org/10.1007/BF02294775 (read as ERIC ED424235 research
/// report; the exact-MEPV contract was verified against catR EPV.R and
/// mirtCAT, and this routine deliberately substitutes Owen updates).
/// Owen, R. J. (1975). A Bayesian sequential procedure for quantal response
/// in the context of adaptive mental testing. Journal of the American
/// Statistical Association, 70(350), 351-356. (NOT read; update formulas
/// follow the crate's `owen_update`.)
#[pyfunction]
fn py_epv_select(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, bool>,
    mu: f64,
    sig2: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_epv_select(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        administered.as_slice()?,
        mu,
        sig2,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("selected", res.selected)?;
    out.set_item("epv", numpy::PyArray1::from_slice(py, &res.epv))?;
    out.set_item(
        "predictive",
        numpy::PyArray1::from_slice(py, &res.predictive),
    )?;
    Ok(out.into())
}

/// Single-cut binary-response Wald SPRT classification for CAT
/// (`mlsirm_core::exposure::sprt_classify`). D = 1 logistic 3PL; point
/// hypotheses at `theta_cut -/+ delta`; log Wald boundaries
/// A = ln((1-beta)/alpha), B = ln(beta/(1-alpha)) with inclusive
/// first-crossing decisions ("above"/"below"/"continue"). `llr_trace`
/// entries past `n_used` are offline counterfactual replay values.
///
/// References (APA 7th; see the core module comment for read/not-read
/// source status):
/// Thompson, N. A. (2007). A practitioner's guide for variable-length
/// computerized classification testing. Practical Assessment, Research &
/// Evaluation, 12(1). https://doi.org/10.7275/fq3r-zz60 (READ)
/// Nydick, S. W. (2014). catIrt (R package). (READ: termSPRT.R,
/// logLik.brm.R, p.brm.R)
/// Eggen, T. J. H. M. (1999). Applied Psychological Measurement, 23(3),
/// 249-261. (NOT read; historical citation via Thompson)
/// Reckase, M. D. (1983). A procedure for decision making using tailored
/// testing. (NOT read; historical citation via Thompson)
/// Wald, A. (1947). Sequential analysis. Wiley. (NOT read; boundary forms
/// verified through the READ sources above)
#[pyfunction]
fn py_sprt_classify(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    responses: PyReadonlyArray1<'_, u8>,
    theta_cut: f64,
    delta: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_sprt_classify(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        responses.as_slice()?,
        theta_cut,
        delta,
        alpha,
        beta,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("decision", res.decision)?;
    out.set_item("n_used", res.n_used)?;
    out.set_item("llr", res.llr)?;
    out.set_item("llr_trace", numpy::PyArray1::from_slice(py, &res.llr_trace))?;
    Ok(out.into())
}

/// Single-cut binary-response confidence-interval (ACI) classification for
/// CAT (`mlsirm_core::exposure::ci_classify`). Interim EAP on a fixed
/// 41-point [-4, 4] grid with standard-normal prior; SE is the EAP posterior
/// SD; interval `theta_hat +/- z_crit * se` vs `theta_cut` with STRICT
/// first-crossing decisions ("above"/"below"/"continue"). Trace entries past
/// `n_used` are offline counterfactual replay values.
///
/// References (APA 7th; see the core module comment for read/not-read
/// source status):
/// Nydick, S. W. (2014). catIrt (R package). (READ: termCI.R, eapEst.R,
/// catIrt.Rd at commit c9e979e4812c27d95d367a7f097edfe8e93ac8eb)
/// Kingsbury, G. G., & Weiss, D. J. (1983). In D. J. Weiss (Ed.), New
/// horizons in testing (pp. 257-283). Academic Press. (NOT read;
/// historical origin)
/// Thompson, N. A. (2007). Practical Assessment, Research & Evaluation,
/// 12(1). (NOT read for the CI method section; background only)
/// Eggen, T. J. H. M., & Straetmans, G. J. J. M. (2000). Educational and
/// Psychological Measurement, 60(5), 713-734. (NOT read; historical)
#[pyfunction]
fn py_ci_classify(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    responses: PyReadonlyArray1<'_, u8>,
    theta_cut: f64,
    z_crit: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_ci_classify(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        responses.as_slice()?,
        theta_cut,
        z_crit,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("decision", res.decision)?;
    out.set_item("n_used", res.n_used)?;
    out.set_item(
        "theta_trace",
        numpy::PyArray1::from_slice(py, &res.theta_trace),
    )?;
    out.set_item("se_trace", numpy::PyArray1::from_slice(py, &res.se_trace))?;
    out.set_item(
        "lower_trace",
        numpy::PyArray1::from_slice(py, &res.lower_trace),
    )?;
    out.set_item(
        "upper_trace",
        numpy::PyArray1::from_slice(py, &res.upper_trace),
    )?;
    Ok(out.into())
}

/// Lord self-scoring flexilevel routing + scoring over a full 0/1 response
/// matrix (`mlsirm_core::exposure::flexilevel_administer`): N (odd) items
/// sorted ascending by difficulty, n = (N+1)/2 administered per person
/// starting at the median item (right -> easiest harder, wrong -> hardest
/// easier), number-right scoring with +1/2 for a wrong last answer ("red").
///
/// References (APA 7th; see the core module comment for read/not-read
/// source status):
/// Lord, F. M. (1970). The self-scoring flexilevel test (RB-70-43; ERIC
/// ED042813). Educational Testing Service. (READ)
/// Lord, F. M. (1971). A theoretical study of the measurement effectiveness
/// of flexilevel tests (RB-71-6; ERIC ED051286). Educational Testing
/// Service. (READ)
#[pyfunction]
fn py_flexilevel_administer(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, u8>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_flexilevel_administer(responses.as_slice()?, n_persons, n_items)
        .map_err(PyValueError::new_err)?;
    let items: Vec<u64> = res.items.iter().map(|&c| c as u64).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("n_administered", res.n_administered)?;
    out.set_item("items", numpy::PyArray1::from_slice(py, &items))?;
    out.set_item(
        "number_right",
        numpy::PyArray1::from_slice(py, &res.number_right),
    )?;
    out.set_item("is_red", numpy::PyArray1::from_slice(py, &res.is_red))?;
    out.set_item("score", numpy::PyArray1::from_slice(py, &res.score))?;
    Ok(out.into())
}

/// Exact conditional flexilevel self-score distribution f(x | theta) by
/// Lord's forward recursion
/// (`mlsirm_core::exposure::flexilevel_score_distribution`). `p[c]` is
/// P(correct) on the c-th difficulty-sorted item at the ability of interest;
/// scores lie on the half-integer lattice {1/2, 1, ..., n}.
///
/// References (APA 7th): Lord, F. M. (1971). RB-71-6 (READ; Eqs. 1-2).
#[pyfunction]
fn py_flexilevel_score_distribution(
    py: Python<'_>,
    p: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_flexilevel_score_distribution(p.as_slice()?).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("scores", numpy::PyArray1::from_slice(py, &res.scores))?;
    out.set_item("probs", numpy::PyArray1::from_slice(py, &res.probs))?;
    out.set_item("mean", res.mean)?;
    out.set_item("variance", res.variance)?;
    Ok(out.into())
}

/// Weiss (1973) stratified-adaptive (stradaptive) test administration for a
/// single examinee (`mlsirm_core::exposure::stradaptive_administer`): items
/// grouped into difficulty strata; correct -> next harder stratum, incorrect
/// -> next easier stratum (clamped at the edges); termination when a ceiling
/// stratum (proportion correct <= chance with >= min_items administered) is
/// identified, the pool is exhausted, or max_items is reached. Returns the
/// administration record, ceiling/basal/highest-non-chance strata, Weiss's
/// ten ability scores (NaN when indeterminate), and the consistency index
/// (population variance of the score-9 stratum set; DERIVED, no printed
/// anchor). See the core module comment for READ/NOT-READ source status and
/// DERIVED-rule labels.
///
/// References (APA 7th):
/// Weiss, D. J. (1973). The stratified adaptive computerized ability test
/// (Research Report 73-3; ERIC ED084301). University of Minnesota,
/// Psychometric Methods Program. (READ)
#[pyfunction]
fn py_stradaptive_administer(
    py: Python<'_>,
    stratum: PyReadonlyArray1<'_, u64>,
    difficulty: PyReadonlyArray1<'_, f64>,
    responses: PyReadonlyArray1<'_, u8>,
    entry_stratum: usize,
    chance: f64,
    min_items: usize,
    max_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let stratum_usize: Vec<usize> = stratum
        .as_slice()?
        .iter()
        .map(|&s| {
            usize::try_from(s).map_err(|_| {
                PyValueError::new_err(format!(
                    "stradaptive_administer: stratum {s} does not fit in usize"
                ))
            })
        })
        .collect::<PyResult<_>>()?;
    let res = core_stradaptive_administer(
        &stratum_usize,
        difficulty.as_slice()?,
        responses.as_slice()?,
        entry_stratum,
        chance,
        min_items,
        max_items,
    )
    .map_err(PyValueError::new_err)?;
    let administered: Vec<u64> = res.administered.iter().map(|&c| c as u64).collect();
    let opt = |v: Option<usize>| -> i64 { v.map(|x| x as i64).unwrap_or(-1) };
    let out = pyo3::types::PyDict::new(py);
    out.set_item(
        "administered",
        numpy::PyArray1::from_slice(py, &administered),
    )?;
    out.set_item(
        "responses_taken",
        numpy::PyArray1::from_slice(py, &res.responses_taken),
    )?;
    out.set_item("reason", res.reason)?;
    out.set_item("ceiling", opt(res.ceiling))?;
    out.set_item("basal", opt(res.basal))?;
    out.set_item("hnc", opt(res.hnc))?;
    out.set_item("next_item", opt(res.next_item))?;
    out.set_item("scores", numpy::PyArray1::from_slice(py, &res.scores))?;
    out.set_item("consistency", res.consistency)?;
    Ok(out.into())
}

/// Larkin & Weiss (1974) pyramidal adaptive test administration for a
/// single examinee (`mlsirm_core::exposure::pyramidal_administer`): items in
/// a triangular structure (stage s holds s items, n(n+1)/2 total); routing
/// is up-one/down-one equal offset (correct -> harder neighbour, incorrect
/// -> easier). Returns the routed path and scoring methods 1-6
/// (number-correct, mean b attempted, mean b correct [NaN when 0 correct],
/// final-item b, final difficulty score via the caller-supplied
/// hypothetical stage n+1 [NaN when `b_next` is absent -- M5 unavailable],
/// and the Hansen all-item score as described by Larkin & Weiss). See the
/// core module comment for READ/NOT-READ source status and DERIVED labels.
///
/// References (APA 7th):
/// Larkin, K. C., & Weiss, D. J. (1974). An empirical investigation of
/// computer-administered pyramidal ability testing (Research Report 74-3;
/// ERIC ED096343). University of Minnesota, Psychometric Methods Program.
/// (READ)
#[pyfunction]
#[pyo3(signature = (b, n_stages, u, b_next=None))]
fn py_pyramidal_administer(
    py: Python<'_>,
    b: PyReadonlyArray1<'_, f64>,
    n_stages: usize,
    u: PyReadonlyArray1<'_, u8>,
    b_next: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let bn_slice = match &b_next {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };
    let res = core_pyramidal_administer(b.as_slice()?, n_stages, u.as_slice()?, bn_slice)
        .map_err(PyValueError::new_err)?;
    let path: Vec<u64> = res.path.iter().map(|&c| c as u64).collect();
    let positions: Vec<u64> = res.positions.iter().map(|&c| c as u64).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("path", numpy::PyArray1::from_slice(py, &path))?;
    out.set_item("positions", numpy::PyArray1::from_slice(py, &positions))?;
    out.set_item("number_correct", res.number_correct)?;
    out.set_item("mean_b_attempted", res.mean_b_attempted)?;
    out.set_item("mean_b_correct", res.mean_b_correct)?;
    out.set_item("final_b", res.final_b)?;
    out.set_item("final_difficulty", res.final_difficulty)?;
    out.set_item("all_item_score", res.all_item_score)?;
    Ok(out.into())
}

/// Two-stage adaptive testing, routing step (Betz & Weiss, 1974, Equation 2
/// and the minimum-|difference| routing rule): returns `(theta1, assigned)`
/// where `assigned` is the 0-based measurement-test index closest in mean
/// difficulty to the routing-test ability estimate. See the core module
/// comment for READ/NOT-READ source status and DERIVED labels.
///
/// References (APA 7th):
/// Betz, N. E., & Weiss, D. J. (1974). Simulation studies of two-stage
/// ability testing (Research Report 74-4; ERIC ED103466). University of
/// Minnesota, Psychometric Methods Program. (READ)
#[pyfunction]
fn py_two_stage_route(
    x1: usize,
    m1: usize,
    a1: f64,
    b1: f64,
    b_meas: PyReadonlyArray1<'_, f64>,
    c: f64,
) -> PyResult<(f64, u64)> {
    let (theta1, assigned) = core_two_stage_route(x1, m1, a1, b1, b_meas.as_slice()?, c)
        .map_err(PyValueError::new_err)?;
    Ok((theta1, assigned as u64))
}

/// Two-stage adaptive testing, full scoring (Betz & Weiss, 1974, Equations
/// 2-3): truncated-normal-ogive subtest ability estimates and the
/// item-count-weighted composite. `administered` must equal the index
/// `py_two_stage_route` assigns for the same routing inputs; a mismatch is a
/// ValueError so `x2` is never scored against the wrong measurement test.
///
/// References (APA 7th):
/// Betz, N. E., & Weiss, D. J. (1973). An empirical study of
/// computer-administered two-stage ability testing (Research Report 73-4;
/// ERIC ED084302). University of Minnesota, Psychometric Methods Program.
/// (READ)
/// Betz, N. E., & Weiss, D. J. (1974). Simulation studies of two-stage
/// ability testing (Research Report 74-4; ERIC ED103466). University of
/// Minnesota, Psychometric Methods Program. (READ)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_two_stage_score(
    py: Python<'_>,
    x1: usize,
    m1: usize,
    a1: f64,
    b1: f64,
    x2: usize,
    m2: usize,
    administered: usize,
    a_meas: PyReadonlyArray1<'_, f64>,
    b_meas: PyReadonlyArray1<'_, f64>,
    c: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_two_stage_score(
        x1,
        m1,
        a1,
        b1,
        x2,
        m2,
        administered,
        a_meas.as_slice()?,
        b_meas.as_slice()?,
        c,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta1", res.theta1)?;
    out.set_item("assigned", res.assigned as u64)?;
    out.set_item("theta2", res.theta2)?;
    out.set_item("composite", res.composite)?;
    Ok(out.into())
}

/// Horn's parallel analysis for principal-component retention
/// (`mlsirm_core::parallel`; oracle: CRAN paran 1.5.6, PCA path). `data` is
/// a flattened row-major `n_persons * n_items` matrix; `centile` is 0 for
/// the mean benchmark or 1..=99 for Glorfeld's upper-centile variant.
#[pyfunction]
fn parallel_analysis(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    n_iterations: usize,
    centile: u32,
    seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_parallel_analysis(
        data.as_slice()?,
        n_persons,
        n_items,
        n_iterations,
        centile,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("retained", res.retained)?;
    out.set_item("eigenvalues", res.eigenvalues)?;
    out.set_item("random_eigenvalues", res.random_eigenvalues)?;
    out.set_item("bias", res.bias)?;
    out.set_item("adjusted_eigenvalues", res.adjusted_eigenvalues)?;
    Ok(out.into())
}

/// Guttman (1945) lambda reliability coefficients plus split-half summaries
/// (`mlsirm_core::reliability`; oracle: CRAN psych 2.6.5 `guttman`/
/// `splitHalf`). `data` is a flattened row-major `n_persons * n_items`
/// matrix of complete finite scores. `n_sample_splits` bounds the split
/// enumeration (exhaustive when C(p, floor(p/2)) fits the budget, else
/// LCG-sampled with `seed`).
#[pyfunction]
fn guttman_lambdas(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    n_sample_splits: usize,
    seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_guttman_lambdas(data.as_slice()?, n_persons, n_items, n_sample_splits, seed)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("lambda1", res.lambda1)?;
    out.set_item("lambda2", res.lambda2)?;
    out.set_item("lambda3", res.lambda3)?;
    out.set_item("lambda4", res.lambda4)?;
    out.set_item("lambda5", res.lambda5)?;
    out.set_item("lambda6", res.lambda6)?;
    out.set_item("beta", res.beta)?;
    out.set_item("mean_split", res.mean_split)?;
    out.set_item("n_splits", res.n_splits)?;
    out.set_item("exhaustive", res.exhaustive)?;
    Ok(out.into())
}

/// ten Berge & Zegers (1978) mu0-mu3 reliability lower bounds
/// (`mlsirm_core::reliability`; oracle: CRAN psych 2.6.5 `tenberge.R`).
/// `data` is a flattened row-major `n_persons * n_items` matrix of complete
/// finite scores.
#[pyfunction]
fn tenberge_mu(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_tenberge_mu(data.as_slice()?, n_persons, n_items).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("mu0", res.mu0)?;
    out.set_item("mu1", res.mu1)?;
    out.set_item("mu2", res.mu2)?;
    out.set_item("mu3", res.mu3)?;
    Ok(out.into())
}

/// Cronbach's (1951) coefficient alpha from raw data (covariance form;
/// `mlsirm_core::reliability`). `data` is a flattened row-major
/// `n_persons * n_items` matrix of complete finite scores.
#[pyfunction]
fn cronbach_alpha(
    data: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<f64> {
    core_cronbach_alpha(data.as_slice()?, n_persons, n_items).map_err(PyValueError::new_err)
}

/// Feldt (1965) exact-F confidence interval for coefficient alpha
/// (`mlsirm_core::reliability`; oracle: CRAN psych 2.6.5 `alpha.ci`).
/// `level` is the two-sided confidence level, e.g. 0.95.
#[pyfunction]
fn feldt_alpha_ci(
    py: Python<'_>,
    alpha: f64,
    n_persons: usize,
    n_items: usize,
    level: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        core_feldt_alpha_ci(alpha, n_persons, n_items, level).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha", res.alpha)?;
    out.set_item("lower", res.lower)?;
    out.set_item("upper", res.upper)?;
    out.set_item("r_bar", res.r_bar)?;
    out.set_item("df1", res.df1)?;
    out.set_item("df2", res.df2)?;
    Ok(out.into())
}

/// Intraclass correlation coefficients (Shrout & Fleiss, 1979 taxonomy),
/// transcribed from CRAN irr 0.85 `icc.R` (READ; `mlsirm_core::reliability`).
/// `ratings` is row-major ns x nr; rows with NaN are dropped listwise.
/// Returns a dict with `value`, `subjects`, `raters`, `fvalue`, `df1`,
/// `df2`, `p_value`, `lbound`, `ubound`.
#[pyfunction]
fn icc(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
    model: &str,
    r#type: &str,
    unit: &str,
    r0: f64,
    conf_level: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_icc(
        ratings.as_slice()?,
        ns,
        nr,
        model,
        r#type,
        unit,
        r0,
        conf_level,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    out.set_item("fvalue", res.fvalue)?;
    out.set_item("df1", res.df1)?;
    out.set_item("df2", res.df2)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("lbound", res.lbound)?;
    out.set_item("ubound", res.ubound)?;
    Ok(out.into())
}

/// Krippendorff's alpha (`mlsirm_core::reliability`; transcribed from
/// CRAN irr 0.85 `kripp.alpha.R`, READ). `ratings` is row-major
/// nraters x nsubjects; NaN marks missing. `method` is one of "nominal",
/// "ordinal", "interval", "ratio". Returns a dict with `value`,
/// `subjects`, `raters`, `levels`, `nmatchval`.
#[pyfunction]
fn kripp_alpha(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    nraters: usize,
    nsubjects: usize,
    method: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_kripp_alpha(ratings.as_slice()?, nraters, nsubjects, method)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    out.set_item("levels", res.levels)?;
    out.set_item("nmatchval", res.nmatchval)?;
    Ok(out.into())
}

/// Finn (1970) coefficient of reliability (`mlsirm_core::reliability`;
/// transcribed from CRAN irr 0.85 `finn.R`, READ). `ratings` is row-major
/// ns x nr; rows with NaN are dropped listwise. `s_levels` is the number of
/// discrete scale levels (>= 2); `model` is "oneway" or "twoway". Returns a
/// dict with `value`, `statistic` (+inf for perfect agreement), `df2`,
/// `p_value`, `subjects`, `raters`.
#[pyfunction]
fn finn_coefficient(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
    s_levels: u32,
    model: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_finn_coefficient(ratings.as_slice()?, ns, nr, s_levels, model)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("statistic", res.statistic)?;
    out.set_item("df2", res.df2)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    Ok(out.into())
}

/// Maxwell's RE agreement coefficient for two raters with binary ratings
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `maxwell.R`, READ). `ratings` is row-major ns x 2; rows with NaN are
/// dropped listwise; the distinct-value union across both columns must have
/// at most 2 levels. Returns a dict with `value`, `subjects`, `raters`.
#[pyfunction]
fn maxwell_re(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_maxwell_re(ratings.as_slice()?, ns, nr).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    Ok(out.into())
}

/// Robinson's A coefficient of agreement
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `robinson.R`, READ). `ratings` is row-major ns x nr; rows with NaN are
/// dropped listwise. Degenerate inputs with no subject variance raise
/// ValueError where R silently returns NaN. Returns a dict with `value`,
/// `subjects`, `raters`.
#[pyfunction]
fn robinson_a(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_robinson_a(ratings.as_slice()?, ns, nr).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    Ok(out.into())
}

/// Mean pairwise Pearson correlation of rater columns
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `meancor.R`, READ). `ratings` is row-major ns x nr; rows with NaN are
/// dropped listwise. With `fisher`, perfectly correlated pairs are
/// dropped before the Fisher-z average and a z test is reported;
/// without it `statistic`/`p_value` are NaN. Degenerate inputs raise
/// ValueError where R yields NA/NaN. Returns a dict with `value`,
/// `statistic`, `p_value`, `dropped`, `subjects`, `raters`.
#[pyfunction]
fn mean_pairwise_cor(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
    fisher: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_mean_pairwise_cor(ratings.as_slice()?, ns, nr, fisher)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("statistic", res.statistic)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("dropped", res.dropped)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    Ok(out.into())
}

/// Mean pairwise Spearman rank correlation of rater columns
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `meanrho.R`, READ). `ratings` is row-major ns x nr; rows with NaN
/// are dropped listwise and columns are midrank-transformed before the
/// pairwise Pearson correlations. With `fisher`, perfectly correlated
/// pairs are dropped before the Fisher-z average and a z test is
/// reported; without it `statistic`/`p_value` are NaN. `ties` flags
/// tied values within any column (R appends a warning string).
/// Degenerate inputs raise ValueError where R yields NA/NaN. Returns a
/// dict with `value`, `statistic`, `p_value`, `dropped`, `ties`,
/// `subjects`, `raters`.
#[pyfunction]
fn mean_pairwise_rho(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    ns: usize,
    nr: usize,
    fisher: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_mean_pairwise_rho(ratings.as_slice()?, ns, nr, fisher)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("statistic", res.statistic)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("dropped", res.dropped)?;
    out.set_item("ties", res.ties)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("raters", res.raters)?;
    Ok(out.into())
}

/// Stuart-Maxwell marginal homogeneity chi-square test for a CxC
/// two-rater counts table (`mlsirm_core::reliability`; transcribed
/// from CRAN irr 0.84.1 `stuart.maxwell.R`, READ). `table` is
/// row-major c x c nonnegative integral counts. Categories whose row
/// and column sums are equal are dropped once, simultaneously; the
/// statistic is d' S^-1 d on the remaining K categories with
/// df = K - 1. Degenerate or out-of-domain inputs raise ValueError.
/// Returns a dict with `value`, `df`, `p_value`, `dropped`,
/// `subjects`, `categories`.
#[pyfunction]
fn stuart_maxwell_mh(
    py: Python<'_>,
    table: PyReadonlyArray1<'_, f64>,
    c: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_stuart_maxwell_mh(table.as_slice()?, c).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("dropped", res.dropped)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("categories", res.categories)?;
    Ok(out.into())
}

/// Bhapkar marginal homogeneity chi-square test for a CxC two-rater
/// counts table (`mlsirm_core::reliability`; transcribed from CRAN
/// irr 0.84.1 `bhapkar.r`, READ). `table` is row-major c x c
/// nonnegative integral counts. Unlike `stuart_maxwell_mh` no
/// category is dropped; the statistic is d' W^-1 d with
/// W = S - d d'/n over the first C-1 categories and df = C - 1.
/// Degenerate or out-of-domain inputs raise ValueError. Returns a
/// dict with `value`, `df`, `p_value`, `subjects`, `categories`.
#[pyfunction]
fn bhapkar_mh(
    py: Python<'_>,
    table: PyReadonlyArray1<'_, f64>,
    c: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_bhapkar_mh(table.as_slice()?, c).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("subjects", res.subjects)?;
    out.set_item("categories", res.categories)?;
    Ok(out.into())
}

/// Rater bias chi-square for a CxC two-rater counts table
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `rater.bias.R`, READ). `table` is row-major c x c nonnegative
/// integral counts. value = rbb/(rbb+rbc) (upper-triangle share),
/// statistic = (rbb-rbc)^2/(rbb+rbc), df = 1. Degenerate or
/// out-of-domain inputs raise ValueError. Returns a dict with
/// `value`, `statistic`, `df`, `p_value`, `subjects`.
#[pyfunction]
fn rater_bias(
    py: Python<'_>,
    table: PyReadonlyArray1<'_, f64>,
    c: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_rater_bias(table.as_slice()?, c).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("value", res.value)?;
    out.set_item("statistic", res.statistic)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("subjects", res.subjects)?;
    Ok(out.into())
}

/// Closed-form sample size for testing Cohen's kappa on a 2x2 table
/// (`mlsirm_core::reliability`; transcribed from CRAN irr 0.84.1
/// `N.cohen.kappa.R`, READ). `rate1`/`rate2` are the raters' marginal
/// proportions in (0, 1); `k1`/`k0` the alternative and null kappas.
/// Infeasible or degenerate parameter combinations raise ValueError
/// (stricter than R, which silently produces NaN). Returns a dict with
/// `n`, `q1`, `q0`, `pre_ceil`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn n_cohen_kappa(
    py: Python<'_>,
    rate1: f64,
    rate2: f64,
    k1: f64,
    k0: f64,
    alpha: f64,
    power: f64,
    twosided: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_n_cohen_kappa(rate1, rate2, k1, k0, alpha, power, twosided)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("n", res.n)?;
    out.set_item("q1", res.q1)?;
    out.set_item("q0", res.q0)?;
    out.set_item("pre_ceil", res.pre_ceil)?;
    Ok(out.into())
}

/// Person separation reliability `(SSD - MSE) / SSD`
/// (`mlsirm_core::reliability`; transcribed from CRAN eRm `SepRel.R`).
/// Returns a dict with `sep_rel`, `ssd`, `mse`, `sep_index`.
#[pyfunction]
fn separation_reliability(
    py: Python<'_>,
    measures: PyReadonlyArray1<'_, f64>,
    se: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_separation_reliability(measures.as_slice()?, se.as_slice()?)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("sep_rel", res.sep_rel)?;
    out.set_item("ssd", res.ssd)?;
    out.set_item("mse", res.mse)?;
    out.set_item("sep_index", res.sep_index)?;
    Ok(out.into())
}
/// 1990). `y`/`observed` are row-major `n_persons * n_items`; `model` is "rasch" or
/// "2pl". `n_classes` latent classes each get their own item parameters. Returns a dict
/// with `a`/`b` (class-major `C*J`), `pi` (`C`), `class_posterior` (`N*C`), `map_class`
/// (`N`), `theta` (`N`), `loglik_trace`, `n_iter`, `converged`, `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, n_persons, n_items, n_classes, model = "rasch", n_starts = 1, max_iter = 500, tol = 1e-6, seed = 0x2545F491))]
fn fit_mixture(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    n_classes: usize,
    model: &str,
    n_starts: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let within = match model {
        "rasch" | "Rasch" | "RASCH" => MixtureModel::Rasch,
        "2pl" | "2PL" | "twopl" | "TwoPl" => MixtureModel::TwoPl,
        other => {
            return Err(PyValueError::new_err(format!(
                "model must be 'rasch' or '2pl'; got {other}"
            )))
        }
    };
    let cfg = MixtureConfig {
        max_iter,
        tol,
        n_starts,
        seed,
        ..MixtureConfig::default()
    };
    let res = core_fit_mixture(
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        n_items,
        n_classes,
        within,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("model", model)?;
    out.set_item("n_classes", res.n_classes)?;
    out.set_item("a", res.a)?;
    out.set_item("b", res.b)?;
    out.set_item("pi", res.pi)?;
    out.set_item("class_posterior", res.class_posterior)?;
    out.set_item("map_class", res.map_class)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Marginal-EM fit of the Linear Logistic Test Model (`mlsirm_core::lltm`, Fischer,
/// 1973). `y`/`observed` are row-major `n_persons * n_items`; `q_design` is row-major
/// `n_items * n_basic` (real operation weights). In the crate's additive sign
/// convention, item easiness is `b_i = c + sum_k q_ik eta_k` (Fischer difficulty is
/// `-b_i`). Returns a dict with `eta` (K), `intercept`, `b` (J induced), `theta`
/// (N), `loglik_trace`, `n_iter`, `converged`, `n_parameters`, and (when `compute_lr`)
/// the LR test of LLTM vs Rasch: `loglik_rasch`, `lr_stat`, `lr_df`, `lr_p`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, q_design, n_persons, n_items, n_basic, fit_intercept = true, compute_lr = true, max_iter = 500, tol = 1e-6))]
fn fit_lltm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    q_design: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    n_basic: usize,
    fit_intercept: bool,
    compute_lr: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let cfg = LltmConfig {
        max_iter,
        tol,
        fit_intercept,
        compute_lr,
        ..LltmConfig::default()
    };
    let res = core_fit_lltm(
        y.as_slice()?,
        observed.as_slice()?,
        q_design.as_slice()?,
        n_persons,
        n_items,
        n_basic,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("eta", res.eta)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("b", res.b)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("loglik_rasch", res.loglik_rasch)?;
    out.set_item("lr_stat", res.lr_stat)?;
    out.set_item("lr_df", res.lr_df)?;
    out.set_item("lr_p", res.lr_p)?;
    Ok(out.into())
}

/// Marginal-EM fit of the testlet response model (`mlsirm_core::testlet`, Bradlow,
/// Wainer, & Wang, 1999). `y`/`observed` are row-major `n_persons * n_items`;
/// `testlet_id[i]` is item `i`'s testlet in `0..n_testlets`; `model` is "rasch" or
/// "2pl". Returns a dict with `a`/`b`/`beta` (per item), `sigma2` (per testlet — the
/// local-dependence estimand), `theta`, `loglik_trace`, `n_iter`, `converged`,
/// `n_parameters`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, observed, testlet_id, n_persons, n_items, n_testlets, model = "rasch", max_iter = 500, tol = 1e-6, q_gamma = 21, estimate_sigma = true, init_sigma2 = 0.5))]
fn fit_testlet(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    testlet_id: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_testlets: usize,
    model: &str,
    max_iter: usize,
    tol: f64,
    q_gamma: usize,
    estimate_sigma: bool,
    init_sigma2: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let within = match model {
        "rasch" | "Rasch" | "RASCH" => TestletModel::Rasch,
        "2pl" | "2PL" | "twopl" | "TwoPl" => TestletModel::TwoPl,
        other => {
            return Err(PyValueError::new_err(format!(
                "model must be 'rasch' or '2pl'; got {other}"
            )))
        }
    };
    let tid: Vec<usize> = testlet_id
        .as_slice()?
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyValueError::new_err(
                    "testlet_id entries must be non-negative",
                ))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<PyResult<_>>()?;
    let cfg = TestletConfig {
        max_iter,
        tol,
        q_gamma,
        estimate_sigma,
        init_sigma2,
        ..TestletConfig::default()
    };
    let res = core_fit_testlet(
        y.as_slice()?,
        observed.as_slice()?,
        &tid,
        n_persons,
        n_items,
        n_testlets,
        within,
        &cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("model", model)?;
    out.set_item("a", res.a)?;
    out.set_item("b", res.b)?;
    out.set_item("beta", res.beta)?;
    out.set_item("sigma2", res.sigma2)?;
    out.set_item("theta", res.theta)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("final_loglik_change", res.final_loglik_change)?;
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Marginal (MMLE-EM) calibration of the latent-space model family
/// (`mlsirm_core::marginal`). `pop_kind` is "single", "multigroup" or
/// "multilevel"; `pop_id` carries the per-person group/cluster index (ignored
/// for "single"). Returns a dict of the fitted quantities.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y,
    observed,
    factor_id,
    n_persons,
    n_items,
    n_dims,
    latent_dim,
    model,
    eps_distance,
    pop_kind = "single",
    pop_id = None,
    n_pop = 0,
    q_theta = 21,
    q_xi = 11,
    q_u = 15,
    max_iter = 200,
    tol = 1e-5,
    m_steps = 4,
    lambda_b = 0.25,
    lambda_alpha = 1.0,
    mu_alpha = 0.5,
    lambda_zeta = 1.0,
    lambda_tau = 1.0,
    mu_tau = 0.5,
    device = "cpu",
    xi_rule = "gh",
    xi_points = 256,
    xi_seed = 0,
    anchor_fixed = None,
    anchor_alpha = None,
    anchor_b = None,
    anchor_zeta = None,
    anchor_tau = None,
    zero_inflation = false,
    covariate_w = None,
    covariate_init_delta = 0.0,
))]
fn fit_marginal(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    factor_id: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    latent_dim: usize,
    model: &str,
    eps_distance: f64,
    pop_kind: &str,
    pop_id: Option<PyReadonlyArray1<'_, i64>>,
    n_pop: usize,
    q_theta: usize,
    q_xi: usize,
    q_u: usize,
    max_iter: usize,
    tol: f64,
    m_steps: usize,
    lambda_b: f64,
    lambda_alpha: f64,
    mu_alpha: f64,
    lambda_zeta: f64,
    lambda_tau: f64,
    mu_tau: f64,
    device: &str,
    xi_rule: &str,
    xi_points: usize,
    xi_seed: u64,
    anchor_fixed: Option<PyReadonlyArray1<'_, bool>>,
    anchor_alpha: Option<PyReadonlyArray1<'_, f64>>,
    anchor_b: Option<PyReadonlyArray1<'_, f64>>,
    anchor_zeta: Option<PyReadonlyArray1<'_, f64>>,
    anchor_tau: Option<f64>,
    zero_inflation: bool,
    covariate_w: Option<PyReadonlyArray1<'_, f64>>,
    covariate_init_delta: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: parse_model_type(model)?,
        eps_distance,
    };
    let factors = convert_factor_id(factor_id.as_slice()?, n_dims)?;
    let ids: Option<Vec<usize>> = match &pop_id {
        Some(arr) => Some(
            arr.as_slice()?
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .map_err(|_| PyValueError::new_err("population ids must be >= 0"))
                })
                .collect::<PyResult<Vec<usize>>>()?,
        ),
        None => None,
    };
    let pop = match pop_kind {
        "single" => PopulationSpec::Single,
        "singlefree" => PopulationSpec::SingleFree,
        "multigroup" => PopulationSpec::Multigroup {
            group_id: ids.ok_or_else(|| PyValueError::new_err("multigroup requires pop_id"))?,
            n_groups: n_pop,
        },
        "multilevel" => PopulationSpec::Multilevel {
            cluster_id: ids.ok_or_else(|| PyValueError::new_err("multilevel requires pop_id"))?,
            n_clusters: n_pop,
        },
        _ => {
            return Err(PyValueError::new_err(
                "pop_kind must be one of ['single', 'multigroup', 'multilevel']",
            ))
        }
    };
    let rule = XiRuleKind::parse(xi_rule)
        .ok_or_else(|| PyValueError::new_err("xi_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let mcfg = MarginalConfig {
        q_theta,
        q_xi,
        q_u,
        max_iter,
        tol,
        m_steps,
        xi_rule: rule,
        xi_points,
        xi_seed,
        zero_inflation,
        ..MarginalConfig::default()
    };
    let penalty = PenaltyConfig {
        lambda_b,
        lambda_alpha,
        mu_alpha,
        lambda_zeta,
        lambda_tau,
        mu_tau,
        ..PenaltyConfig::lsirm_prior()
    };
    let anchors: Option<Anchors> = match (&anchor_fixed, &anchor_alpha, &anchor_b, &anchor_zeta) {
        (None, None, None, None) => None,
        (Some(f), Some(a), Some(b_arr), Some(z)) => Some(Anchors {
            fixed: f.as_slice()?.to_vec(),
            alpha: a.as_slice()?.to_vec(),
            b: b_arr.as_slice()?.to_vec(),
            zeta: z.as_slice()?.to_vec(),
            tau: anchor_tau,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "anchors require anchor_fixed, anchor_alpha, anchor_b and anchor_zeta together",
            ))
        }
    };
    let covariate: Option<ItemCovariate> = match &covariate_w {
        Some(w) => Some(ItemCovariate {
            w: w.as_slice()?.to_vec(),
            init_delta: covariate_init_delta,
        }),
        None => None,
    };
    let res = core_fit_marginal_full(
        y.as_slice()?,
        observed.as_slice()?,
        &factors,
        &config,
        &pop,
        &mcfg,
        &penalty,
        device,
        anchors.as_ref(),
        covariate.as_ref(),
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha", res.alpha)?;
    out.set_item("b", res.b)?;
    out.set_item("zeta", res.zeta)?;
    out.set_item("tau", res.tau)?;
    out.set_item("theta_eap", res.theta_eap)?;
    out.set_item("theta_sd", res.theta_sd)?;
    out.set_item("xi_eap", res.xi_eap)?;
    out.set_item("mu", res.mu)?;
    out.set_item("sigma", res.sigma)?;
    out.set_item("sigma_u", res.sigma_u)?;
    out.set_item("u_eap", res.u_eap)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("delta", res.delta)?;
    out.set_item("pi_zero", res.pi_zero)?;
    out.set_item("zero_responsibility", res.zero_responsibility)?;
    if let Some(&ll) = res.loglik_trace.last() {
        let ic = mlsirm_core::fitstats::information_criteria(ll, res.n_parameters, n_persons);
        let icd = pyo3::types::PyDict::new(py);
        icd.set_item("aic", ic.aic)?;
        icd.set_item("bic", ic.bic)?;
        icd.set_item("aicc", ic.aicc)?;
        icd.set_item("sabic", ic.sabic)?;
        icd.set_item("caic", ic.caic)?;
        icd.set_item("n_parameters", ic.n_parameters)?;
        icd.set_item("n", ic.n)?;
        out.set_item("ic", icd)?;
    }
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

fn parse_xi_rule(name: &str, q_xi: usize, xi_points: usize, xi_seed: u64) -> PyResult<XiRule> {
    match XiRuleKind::parse(name) {
        Some(XiRuleKind::GaussHermite) => Ok(XiRule::GaussHermite { q_xi }),
        Some(XiRuleKind::Halton) => Ok(XiRule::Halton {
            n: xi_points,
            shift_seed: xi_seed,
        }),
        Some(XiRuleKind::MonteCarlo) => Ok(XiRule::MonteCarlo {
            n: xi_points,
            seed: xi_seed.max(1),
        }),
        None => Err(PyValueError::new_err(
            "xi_rule must be one of ['gh', 'qmc', 'mc']",
        )),
    }
}

macro_rules! bank_from_args {
    ($alpha:expr, $b:expr, $zeta:expr, $tau:expr, $factor_id:expr, $model:expr,
     $n_dims:expr, $latent_dim:expr, $eps:expr, $factors:ident, $bank:ident) => {
        let $factors = convert_factor_id($factor_id.as_slice()?, $n_dims)?;
        let $bank = ItemBank {
            alpha: $alpha.as_slice()?,
            b: $b.as_slice()?,
            zeta: $zeta.as_slice()?,
            tau: $tau,
            factor_id: &$factors,
            model_type: parse_model_type($model)?,
            n_dims: $n_dims,
            latent_dim: $latent_dim,
            eps_distance: $eps,
        };
    };
}

/// EAP scoring of response vectors against frozen item parameters.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0, device = "auto",
))]
fn score_bank_eap(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    device: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let dev = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err(format!("unknown device: {device}")))?;
    let res = core_score_eap_device(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        q_theta,
        rule,
        dev,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_eap", res.theta_eap)?;
    out.set_item("theta_sd", res.theta_sd)?;
    out.set_item("xi_eap", res.xi_eap)?;
    out.set_item("loglik", res.loglik)?;
    Ok(out.into())
}

/// MAP scoring (posterior Newton) against frozen item parameters.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, max_iter = 100, tol = 1e-6,
))]
fn score_bank_map(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let res = core_score_map(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        max_iter,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_map", res.theta_map)?;
    out.set_item("theta_se", res.theta_se)?;
    out.set_item("xi_map", res.xi_map)?;
    out.set_item("log_posterior", res.log_posterior)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

/// Warm's (1989) weighted-likelihood ability estimates for a unidimensional dichotomous test (Rust
/// compute path). The bias-reduced maximum-likelihood estimator: solves
/// `dlnL/dtheta + J(theta)/(2 I(theta)) = 0` with `J = sum_i P_i' P_i''/(P_i Q_i)` (computed directly;
/// it equals `I'` for the 2PL/Rasch only, and is neither `I'` nor `I'/2` for the 3PL/4PL), yielding a
/// FINITE estimate for the all-correct /
/// all-incorrect patterns where the MLE diverges. `a`/`b`/`c`/`d` are per-item NATURAL-scale parameters
/// (`a` the slope, NOT log-alpha) with `0 <= c_i < d_i <= 1` (2PL: `c=0, d=1`); `y`/`observed` are
/// row-major `n_persons * n_items` (`0/1`; missing items dropped per person). Returns a dict with
/// `theta` (`n_persons`), `se` (`1/sqrt(I)`), and `boundary` (root clamped to `+/- theta_bound`).
///
/// Reference (APA 7th ed.):
///   Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
///     Psychometrika, 54(3), 427-450.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (a, b, c, d, y, observed, n_persons, n_items, theta_bound = 20.0, tol = 1e-8))]
fn score_wle(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    n_items: usize,
    theta_bound: f64,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    if a.as_slice()?.len() != n_items {
        return Err(PyValueError::new_err("a length must equal n_items"));
    }
    let res = core_score_wle(
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        d.as_slice()?,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        theta_bound,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta", res.theta)?;
    out.set_item("se", res.se)?;
    out.set_item("boundary", res.boundary)?;
    Ok(out.into())
}

/// Zumbo (1999) logistic-regression DIF (Rust compute path; Swaminathan & Rogers, 1990). Regresses each
/// item response on the observed matching score, the group, and their interaction in three nested
/// logistic models, separating UNIFORM from NON-UNIFORM (crossing) DIF — the latter is invisible to the
/// Mantel-Haenszel procedure. `y` is a row-major `n_persons * n_items` `0/1` array; `group` is length
/// `n_persons` with `0` = reference, `1` = focal. Returns a dict of per-item arrays: `item`,
/// `chi2_uniform`/`p_uniform` and `chi2_nonuniform`/`p_nonuniform` (1 df each, DESCRIPTIVE and
/// unadjusted), `chi2_total`/`p_total` (2 df, the PRIMARY omnibus test that Benjamini-Hochberg adjusts),
/// `delta_r2` (Nagelkerke `R2(M2) - R2(M0)`), `delta_r2_uniform` (uncalibrated descriptive),
/// `jg_class` (Jodoin & Gierl, 2001 `"A"`/`"B"`/`"C"`, or `"U"` when undefined), `flagged_bh`, and
/// `converged`. A failed fit (separation, rank-deficient design, no convergence) reports NaN statistics,
/// `converged=False`, and is never flagged.
///
/// References (APA 7th ed.):
///   Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size
///     measure with the logistic regression procedure for DIF detection. Applied Measurement in
///     Education, 14(4), 329-349.
///   Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic
///     regression procedures. Journal of Educational Measurement, 27(4), 361-370.
///   Zumbo, B. D. (1999). A handbook on the theory and methods of differential item functioning (DIF).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, group, n_persons, n_items, exclude_studied_item = false, fdr_q = 0.05, max_iter = 50))]
fn logistic_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    exclude_studied_item: bool,
    fdr_q: f64,
    max_iter: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv = binary_u8(y.as_slice()?)?;
    let gv: Vec<u8> = group
        .as_slice()?
        .iter()
        .map(|&g| match g {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "group labels must be 0 (reference) or 1 (focal)",
            )),
        })
        .collect::<PyResult<_>>()?;
    let cfg = LogisticDifConfig {
        exclude_studied_item,
        fdr_q,
        max_iter,
    };
    let rows =
        core_logistic_dif(&yv, &gv, n_persons, n_items, &cfg).map_err(PyValueError::new_err)?;
    Ok(logistic_rows_dict(py, &rows)?.into())
}

/// Per-item arrays for a logistic-regression DIF sweep, shared by the plain and purified entry points.
fn logistic_rows_dict<'py>(
    py: Python<'py>,
    rows: &[LogisticDifRow],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item", rows.iter().map(|r| r.item).collect::<Vec<_>>())?;
    out.set_item(
        "chi2_uniform",
        rows.iter().map(|r| r.chi2_uniform).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "p_uniform",
        rows.iter().map(|r| r.p_uniform).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "chi2_nonuniform",
        rows.iter().map(|r| r.chi2_nonuniform).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "p_nonuniform",
        rows.iter().map(|r| r.p_nonuniform).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "chi2_total",
        rows.iter().map(|r| r.chi2_total).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "p_total",
        rows.iter().map(|r| r.p_total).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "delta_r2",
        rows.iter().map(|r| r.delta_r2).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "delta_r2_uniform",
        rows.iter().map(|r| r.delta_r2_uniform).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "jg_class",
        rows.iter().map(|r| r.jg_class.as_str()).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "flagged_bh",
        rows.iter().map(|r| r.flagged_bh).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "converged",
        rows.iter().map(|r| r.converged).collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// Attach the purification-loop metadata to a per-item row dict.
///
/// The loop's scalar convergence flag is `purify_converged`, NOT `converged`: `logistic_rows_dict`
/// already publishes a per-item `converged` array (did each item's IRLS fit succeed), and
/// `PyDict::set_item` overwrites, so reusing the name silently destroyed a length-`J` array and
/// returned a bare `bool` under a key the caller expects to be indexable. The two flags answer
/// different questions and both are needed, so they get different names on BOTH entry points — the
/// Mantel-Haenszel dict has no `converged` key of its own, but an asymmetric spelling would be its own
/// trap.
fn purify_meta(
    out: &pyo3::Bound<'_, pyo3::types::PyDict>,
    anchor: Vec<bool>,
    n_anchor: usize,
    rounds: usize,
    converged: bool,
    termination_reason: &str,
) -> PyResult<()> {
    debug_assert!(
        !out.contains("purify_converged").unwrap_or(false),
        "purification metadata would overwrite an existing per-item key"
    );
    out.set_item("anchor", anchor)?;
    out.set_item("n_anchor", n_anchor)?;
    out.set_item("rounds", rounds)?;
    out.set_item("purify_converged", converged)?;
    out.set_item("purify_termination_reason", termination_reason)?;
    Ok(())
}

/// Convert an `i64` response slice to `0/1` bytes, rejecting anything else.
fn binary_u8(slice: &[i64]) -> PyResult<Vec<u8>> {
    slice
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("responses must be 0 or 1")),
        })
        .collect()
}

/// Rasch conditional maximum likelihood item difficulties (Rust compute path; Andersen, 1970, 1972).
/// Conditioning each response pattern on its raw score (the sufficient statistic for ability) eliminates
/// the person parameters, so the difficulties are estimated without any ability-distribution assumption
/// and consistently at fixed test length. `y` is a row-major `n_persons * n_items` complete `0/1` array
/// (persons scoring `0` or `n_items` are dropped). Returns a dict with `beta` (sum-zero item
/// difficulties), `se` (from the pseudoinverse of the conditional information), `loglik`, `n_iter`,
/// `converged`, and `n_used`.
///
/// Reference (APA 7th ed.):
///   Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations.
///     Journal of the Royal Statistical Society: Series B, 34(1), 42-54.
#[pyfunction]
#[pyo3(signature = (y, n_persons, n_items, max_iter = 100, tol = 1e-8))]
fn fit_rasch_cml(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv = binary_u8(y.as_slice()?)?;
    let res = core_fit_rasch_cml(&yv, n_persons, n_items, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("beta", res.beta)?;
    out.set_item("se", res.se)?;
    out.set_item("loglik", res.loglik)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("n_used", res.n_used)?;
    Ok(out.into())
}

/// Andersen's (1973) conditional likelihood-ratio test of Rasch fit (Rust compute path). Partitions the
/// persons by `group` (labels `0..n_groups`), fits CML within each group and pooled, and refers
/// `LR = 2[sum_g llc_g - llc_pooled]` to `chi^2((n_groups - 1)(n_items - 1))`; a significant `LR`
/// rejects invariance of the item difficulties across the split. Returns a dict with `lr`, `df`,
/// `p_value`, and `n_used` (per-group retained counts).
///
/// Reference (APA 7th ed.):
///   Andersen, E. B. (1973). A goodness of fit test for the Rasch model. Psychometrika, 38(1), 123-140.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, group, n_groups, n_persons, n_items, max_iter = 100, tol = 1e-8))]
fn andersen_lr_test(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_groups: usize,
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    if n_groups > 256 {
        return Err(PyValueError::new_err("n_groups must be <= 256"));
    }
    let yv = binary_u8(y.as_slice()?)?;
    let gv: Vec<u8> = group
        .as_slice()?
        .iter()
        .map(|&g| {
            if g < 0 || g as usize >= n_groups {
                Err(PyValueError::new_err("group labels must be in 0..n_groups"))
            } else {
                Ok(g as u8)
            }
        })
        .collect::<PyResult<_>>()?;
    let res = core_andersen_lr(&yv, &gv, n_groups, n_persons, n_items, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("lr", res.lr)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("n_used", res.n_used)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

/// Summed-score EAP conversion tables (Lord-Wingersky / Thissen et al. 1995).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim, eps_distance,
    prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11, xi_points = 256,
    xi_seed = 0, device = "auto",
))]
fn eapsum_tables(
    py: Python<'_>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    device: &str,
) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let tables = core_eapsum_tables_device(&bank, &prior, q_theta, rule, device)
        .map_err(PyValueError::new_err)?;
    let mut out = Vec::new();
    for t in tables {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("dim", t.dim)?;
        d.set_item("n_items_dim", t.n_items_dim)?;
        d.set_item("score_prob", t.score_prob)?;
        d.set_item("eap", t.eap)?;
        d.set_item("sd", t.sd)?;
        out.push(d.into());
    }
    Ok(out)
}

/// Apply EAPsum tables to complete dichotomous response vectors in Rust.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, factor_id, n_dims, table_offsets, table_eap, table_sd,
    device = "auto",
))]
fn score_eapsum(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    factor_id: PyReadonlyArray1<'_, i64>,
    n_dims: usize,
    table_offsets: PyReadonlyArray1<'_, i64>,
    table_eap: PyReadonlyArray1<'_, f64>,
    table_sd: PyReadonlyArray1<'_, f64>,
    device: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let factors: Vec<usize> = factor_id
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("factor_id values must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let offsets: Vec<usize> = table_offsets
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("table offsets must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    let eap = table_eap.as_slice()?;
    let sd = table_sd.as_slice()?;
    if offsets.len() != n_dims + 1 || offsets.first() != Some(&0) {
        return Err(PyValueError::new_err(
            "table_offsets must have length n_dims + 1 and start at zero",
        ));
    }
    if offsets.windows(2).any(|pair| pair[1] <= pair[0])
        || offsets.last() != Some(&eap.len())
        || eap.len() != sd.len()
    {
        return Err(PyValueError::new_err(
            "table offsets must be strictly increasing and end at the table value length",
        ));
    }
    let tables: Vec<EapSumTable> = (0..n_dims)
        .map(|dim| {
            let start = offsets[dim];
            let end = offsets[dim + 1];
            EapSumTable {
                dim,
                n_items_dim: end - start - 1,
                score_prob: Vec::new(),
                eap: eap[start..end].to_vec(),
                sd: sd[start..end].to_vec(),
            }
        })
        .collect();
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let result = core_score_eapsum_device(
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &factors,
        n_dims,
        &tables,
        device,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_eap", result.theta_eap)?;
    out.set_item("theta_sd", result.theta_sd)?;
    out.set_item("n_observed", result.n_observed)?;
    Ok(out.into())
}

/// Orlando-Thissen S-X2 with the large-N practical-significance effect size.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0, min_expected = 1.0, fdr_q = 0.05, min_effect = 0.0,
    person_weight = None,
))]
fn s_x2_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    min_expected: f64,
    fdr_q: f64,
    min_effect: f64,
    person_weight: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let cfg = SX2Config {
        q_theta,
        xi_rule: parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?,
        min_expected,
        fdr_q,
        min_effect,
    };
    let weight_storage = match &person_weight {
        Some(w) => Some(w.as_slice()?.to_vec()),
        None => None,
    };
    let res = core_s_x2(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        &cfg,
        weight_storage.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("statistic", res.statistic)?;
    out.set_item("g2_statistic", res.g2_statistic)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("g2_p_value", res.g2_p_value)?;
    out.set_item("rms_residual", res.rms_residual)?;
    out.set_item("flagged_bh", res.flagged_bh)?;
    out.set_item("n_score_groups", res.n_score_groups)?;
    Ok(out.into())
}

/// Per-person observed-vs-expected pass-rate residuals (Rust compute path).
#[pyfunction]
#[pyo3(signature = (y, observed, prob, n_persons))]
fn leniency_residuals_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    prob: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_leniency_residuals(
        y.as_slice()?,
        observed.as_slice()?,
        prob.as_slice()?,
        n_persons,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("residual", res.residual)?;
    out.set_item("observed_mean", res.observed_mean)?;
    out.set_item("expected_mean", res.expected_mean)?;
    out.set_item("n_observed", res.n_observed)?;
    out.set_item("mean", res.mean)?;
    out.set_item("sd", res.sd)?;
    out.set_item("abs_p95", res.abs_p95)?;
    Ok(out.into())
}

/// IRT scale linking (moment / Haebara / Stocking-Lord) for a common-item
/// design. `theta`/`weight` are used by the characteristic-curve methods.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (a_old, b_old, a_new, b_new, theta, weight, method = "stocking_lord"))]
fn irt_link(
    py: Python<'_>,
    a_old: PyReadonlyArray1<'_, f64>,
    b_old: PyReadonlyArray1<'_, f64>,
    a_new: PyReadonlyArray1<'_, f64>,
    b_new: PyReadonlyArray1<'_, f64>,
    theta: PyReadonlyArray1<'_, f64>,
    weight: PyReadonlyArray1<'_, f64>,
    method: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = LinkMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown linking method: {method}")))?;
    let res = core_irt_link(
        a_old.as_slice()?,
        b_old.as_slice()?,
        a_new.as_slice()?,
        b_new.as_slice()?,
        theta.as_slice()?,
        weight.as_slice()?,
        m,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", res.slope)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("criterion", res.criterion)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    out.set_item("termination_reason", res.termination_reason)?;
    out.set_item("max_iter", res.max_iter)?;
    out.set_item("final_objective_span", res.final_objective_span)?;
    out.set_item("objective_tolerance", res.objective_tolerance)?;
    out.set_item("final_parameter_span", res.final_parameter_span)?;
    out.set_item("parameter_tolerance", res.parameter_tolerance)?;
    Ok(out.into())
}

/// Fixed-anchor mean/mean-style parameter linking onto a target metric.
///
/// Returns a dict with linked ``theta`` (list of per-person rows), ``alpha``,
/// ``b``, and affine evidence ``scale`` / ``shift``. Python reconstructs the
/// parameter object and attaches ``anchor_items``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn link_fixed_item_parameters(
    py: Python<'_>,
    source_theta: PyReadonlyArray2<'_, f64>,
    source_alpha: PyReadonlyArray1<'_, f64>,
    source_b: PyReadonlyArray1<'_, f64>,
    target_alpha: PyReadonlyArray1<'_, f64>,
    target_b: PyReadonlyArray1<'_, f64>,
    anchors: PyReadonlyArray1<'_, i64>,
    factors: PyReadonlyArray1<'_, i64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let shape = source_theta.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("source theta must be 2-D"));
    }
    let n_persons = shape[0];
    let n_dims = shape[1];
    let res = core_link_fixed_item_parameters(
        source_theta.as_slice()?,
        n_persons,
        n_dims,
        source_alpha.as_slice()?,
        source_b.as_slice()?,
        target_alpha.as_slice()?,
        target_b.as_slice()?,
        anchors.as_slice()?,
        factors.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let mut theta_rows: Vec<Vec<f64>> = Vec::with_capacity(n_persons);
    for p in 0..n_persons {
        let start = p * n_dims;
        theta_rows.push(res.theta[start..start + n_dims].to_vec());
    }
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta", theta_rows)?;
    out.set_item("alpha", res.alpha)?;
    out.set_item("b", res.b)?;
    out.set_item("scale", res.scale)?;
    out.set_item("shift", res.shift)?;
    Ok(out.into())
}

fn equate_result_dict(py: Python<'_>, res: EquateResult) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("x_scores", res.x_scores)?;
    out.set_item("y_equivalents", res.y_equivalents)?;
    out.set_item("mu_x", res.mu_x)?;
    out.set_item("sigma_x", res.sigma_x)?;
    out.set_item("mu_y", res.mu_y)?;
    out.set_item("sigma_y", res.sigma_y)?;
    out.set_item("mu_eq", res.mu_eq)?;
    out.set_item("sigma_eq", res.sigma_eq)?;
    out.set_item("slope", res.slope)?;
    out.set_item("intercept", res.intercept)?;
    out.set_item("n_x", res.n_x)?;
    out.set_item("n_y", res.n_y)?;
    out.set_item("h_x", res.h_x)?;
    out.set_item("h_y", res.h_y)?;
    Ok(out.into())
}

/// Univariate log-linear presmoothing of a score-frequency distribution (Rust
/// compute path; Holland & Thayer, 2000). `counts` are raw frequencies over
/// scores 0..=k; `degree` moments are preserved. Returns a dict with the smoothed
/// `probs`, `log_lik`, `aic`, `bic`, `moments`, `converged`, `iters`.
#[pyfunction]
#[pyo3(signature = (counts, degree = 6))]
fn loglinear_smooth(
    py: Python<'_>,
    counts: PyReadonlyArray1<'_, f64>,
    degree: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let fit = core_loglinear_smooth(counts.as_slice()?, degree).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("probs", fit.probs)?;
    out.set_item("log_lik", fit.log_lik)?;
    out.set_item("aic", fit.aic)?;
    out.set_item("bic", fit.bic)?;
    out.set_item("moments", fit.moments)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("iters", fit.iters)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("final_gradient_max", fit.final_gradient_max)?;
    out.set_item("gradient_tolerance", fit.gradient_tolerance)?;
    Ok(out.into())
}

/// Equipercentile-family EG equating with optional log-linear presmoothing and a
/// choice of continuization kernel (Rust compute path; Kolen & Brennan, 2014; von
/// Davier et al., 2004). `continuization` is "uniform" (equipercentile) or
/// "gaussian" (kernel). `smooth_degree_x`/`_y` presmooth each form (None = raw);
/// `bandwidth_x`/`_y` fix the Gaussian bandwidth (None = penalty-selected).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x_scores, y_scores, k_x, k_y, continuization = "uniform", smooth_degree_x = None, smooth_degree_y = None, bandwidth_x = None, bandwidth_y = None))]
fn equate_observed_scores_ext(
    py: Python<'_>,
    x_scores: PyReadonlyArray1<'_, f64>,
    y_scores: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    continuization: &str,
    smooth_degree_x: Option<usize>,
    smooth_degree_y: Option<usize>,
    bandwidth_x: Option<f64>,
    bandwidth_y: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let cont = Continuization::parse(continuization).ok_or_else(|| {
        PyValueError::new_err(format!("unknown continuization: {continuization}"))
    })?;
    let res = core_equate_eg_ext(
        x_scores.as_slice()?,
        y_scores.as_slice()?,
        k_x,
        k_y,
        EgSmoothOptions {
            continuization: cont,
            smooth_degree_x,
            smooth_degree_y,
            bandwidth_x,
            bandwidth_y,
        },
    )
    .map_err(PyValueError::new_err)?;
    equate_result_dict(py, res)
}

/// Equivalent-groups observed-score equating of form X onto form Y (Rust compute
/// path; Kolen & Brennan, 2014). `method` is "mean", "linear", or
/// "equipercentile". Returns a dict with the conversion table and moments.
#[pyfunction]
#[pyo3(signature = (x_scores, y_scores, k_x, k_y, method = "equipercentile"))]
fn equate_observed_scores(
    py: Python<'_>,
    x_scores: PyReadonlyArray1<'_, f64>,
    y_scores: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    method: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = EquateMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown equating method: {method}")))?;
    let res = core_equate_eg(x_scores.as_slice()?, y_scores.as_slice()?, k_x, k_y, m)
        .map_err(PyValueError::new_err)?;
    equate_result_dict(py, res)
}

/// NEAT (common-item non-equivalent groups) observed-score equating (Rust compute
/// path; Kolen & Brennan, 2014). Population 1 takes X + anchor V, population 2
/// takes Y + anchor V. `method` is "chained" or "frequency_estimation"; `w1` is
/// the population-1 synthetic weight (FE only).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x_total, x_anchor, y_total, y_anchor, k_x, k_y, k_v, method = "chained", w1 = 0.5))]
fn equate_neat(
    py: Python<'_>,
    x_total: PyReadonlyArray1<'_, f64>,
    x_anchor: PyReadonlyArray1<'_, f64>,
    y_total: PyReadonlyArray1<'_, f64>,
    y_anchor: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    k_v: usize,
    method: &str,
    w1: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = NeatMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown NEAT method: {method}")))?;
    let res = core_equate_neat(
        x_total.as_slice()?,
        x_anchor.as_slice()?,
        y_total.as_slice()?,
        y_anchor.as_slice()?,
        k_x,
        k_y,
        k_v,
        w1,
        m,
    )
    .map_err(PyValueError::new_err)?;
    equate_result_dict(py, res)
}

/// Tucker & Levine linear observed-score NEAT equating (Rust compute path; Kolen
/// & Brennan, 2014). `method` is "tucker" or "levine"; `anchor_kind` is "internal"
/// or "external" (affects Levine only). `w1` is the population-1 synthetic weight.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x_total, x_anchor, y_total, y_anchor, k_x, k_y, method = "tucker", anchor_kind = "internal", w1 = 0.5))]
fn equate_neat_linear(
    py: Python<'_>,
    x_total: PyReadonlyArray1<'_, f64>,
    x_anchor: PyReadonlyArray1<'_, f64>,
    y_total: PyReadonlyArray1<'_, f64>,
    y_anchor: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    method: &str,
    anchor_kind: &str,
    w1: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = NeatLinearMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown linear NEAT method: {method}")))?;
    let ak = AnchorKind::parse(anchor_kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown anchor kind: {anchor_kind}")))?;
    let res = core_equate_neat_linear(
        x_total.as_slice()?,
        x_anchor.as_slice()?,
        y_total.as_slice()?,
        y_anchor.as_slice()?,
        k_x,
        k_y,
        w1,
        m,
        ak,
    )
    .map_err(PyValueError::new_err)?;
    equate_result_dict(py, res)
}

fn see_result_dict(py: Python<'_>, res: SeeResult) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("x_scores", res.x_scores)?;
    out.set_item("y_equivalents", res.y_equivalents)?;
    out.set_item("se", res.se)?;
    out.set_item("ci_lo", res.ci_lo)?;
    out.set_item("ci_hi", res.ci_hi)?;
    out.set_item("n_boot", res.n_boot)?;
    out.set_item("ci_level", res.ci_level)?;
    Ok(out.into())
}

/// Nonparametric bootstrap standard errors of equating for the EG design (Rust
/// compute path; Kolen & Brennan, 2014, ch. 7). Resamples examinees per group
/// independently and re-equates; works for "mean"/"linear"/"equipercentile".
/// Returns a dict with per-score `se`, `ci_lo`, `ci_hi`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x_scores, y_scores, k_x, k_y, method = "equipercentile", n_boot = 1000, ci_level = 0.95, seed = 0))]
fn bootstrap_see(
    py: Python<'_>,
    x_scores: PyReadonlyArray1<'_, f64>,
    y_scores: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    method: &str,
    n_boot: usize,
    ci_level: f64,
    seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = EquateMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown equating method: {method}")))?;
    let res = core_bootstrap_see(
        x_scores.as_slice()?,
        y_scores.as_slice()?,
        k_x,
        k_y,
        m,
        n_boot,
        ci_level,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    see_result_dict(py, res)
}

/// Closed-form delta-method standard errors of equating for the "mean"/"linear"
/// EG methods (Rust compute path; Kolen & Brennan, 2014). Errors on
/// equipercentile (use `bootstrap_see`).
#[pyfunction]
#[pyo3(signature = (x_scores, y_scores, k_x, k_y, method = "linear", ci_level = 0.95))]
fn analytic_see(
    py: Python<'_>,
    x_scores: PyReadonlyArray1<'_, f64>,
    y_scores: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    method: &str,
    ci_level: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = EquateMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown equating method: {method}")))?;
    let res = core_analytic_see(
        x_scores.as_slice()?,
        y_scores.as_slice()?,
        k_x,
        k_y,
        m,
        ci_level,
    )
    .map_err(PyValueError::new_err)?;
    see_result_dict(py, res)
}

/// Circle-arc small-sample observed-score equating (Rust compute path;
/// Livingston & Kim, 2008, ETS RR-08-39). Method "1"/"arc1" fits the arc
/// through the raw points; "2"/"arc2" decomposes into a linear component
/// plus an arc on the transformed points. Scores must lie in [x1, x3]
/// (the source's below-endpoint linear extension is not implemented).
#[pyfunction]
#[pyo3(signature = (scores, low, middle, high, method))]
fn circle_arc_equate(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f64>,
    low: (f64, f64),
    middle: (f64, f64),
    high: (f64, f64),
    method: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = CircleArcMethod::parse(method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown circle-arc method: {method}")))?;
    let res = core_circle_arc_equate(scores.as_slice()?, low, middle, high, m)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("equated", PyArray1::from_slice(py, &res.equated))?;
    d.set_item("xc", res.xc)?;
    d.set_item("yc", res.yc)?;
    d.set_item("r2", res.r2)?;
    d.set_item("collinear", res.collinear)?;
    d.set_item("middle", res.middle)?;
    Ok(d.into())
}

/// Anchor-design middle point for circle-arc equating (Livingston & Kim,
/// 2008, eq. 9): returns (x2, y2) with x2 = m_xa and
/// y2 = m_yb + (s_yb / s_vb) * (m_va - m_vb).
#[pyfunction]
#[pyo3(signature = (m_xa, m_va, m_yb, s_yb, m_vb, s_vb))]
fn circle_arc_middle_anchor(
    m_xa: f64,
    m_va: f64,
    m_yb: f64,
    s_yb: f64,
    m_vb: f64,
    s_vb: f64,
) -> PyResult<(f64, f64)> {
    core_circle_arc_middle_anchor(m_xa, m_va, m_yb, s_yb, m_vb, s_vb).map_err(PyValueError::new_err)
}

/// Nominal weights mean equating for the NEAT design (Rust compute path;
/// Babcock, Albano, & Raymond, 2012, as restated by Albano, 2016, eq. 42).
/// Slope is exactly 1; the intercept is the synthetic-mean difference with
/// nominal-weights gammas k_x/k_v and k_y/k_v.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x_total, x_anchor, y_total, y_anchor, k_x, k_y, k_v, w1 = 0.5))]
fn nominal_weights_mean_equate(
    py: Python<'_>,
    x_total: PyReadonlyArray1<'_, f64>,
    x_anchor: PyReadonlyArray1<'_, f64>,
    y_total: PyReadonlyArray1<'_, f64>,
    y_anchor: PyReadonlyArray1<'_, f64>,
    k_x: usize,
    k_y: usize,
    k_v: usize,
    w1: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_nominal_weights_mean_equate(
        x_total.as_slice()?,
        x_anchor.as_slice()?,
        y_total.as_slice()?,
        y_anchor.as_slice()?,
        k_x,
        k_y,
        k_v,
        w1,
    )
    .map_err(PyValueError::new_err)?;
    equate_result_dict(py, res)
}

/// Composite linking of component conversion tables (Holland & Strawderman,
/// 2011, as cited by Albano, 2016, eqs. 31-32). With `slopes` supplied the
/// symmetric eq.-32 weight adjustment is applied; otherwise weights are
/// normalized raw weights (documented deviation from R's un-normalized path).
#[pyfunction]
#[pyo3(signature = (tables, weights, slopes = None, p = 1.0))]
fn composite_linking(
    py: Python<'_>,
    tables: Vec<PyReadonlyArray1<'_, f64>>,
    weights: PyReadonlyArray1<'_, f64>,
    slopes: Option<PyReadonlyArray1<'_, f64>>,
    p: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let tabs: Vec<Vec<f64>> = tables
        .iter()
        .map(|t| t.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let slope_vec = match &slopes {
        Some(s) => Some(s.as_slice()?.to_vec()),
        None => None,
    };
    let res = core_composite_linking(&tabs, weights.as_slice()?, slope_vec.as_deref(), p)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("composite", PyArray1::from_slice(py, &res.composite))?;
    d.set_item(
        "adjusted_weights",
        PyArray1::from_slice(py, &res.adjusted_weights),
    )?;
    d.set_item("symmetric", res.symmetric)?;
    Ok(d.into())
}

/// Thurstone (1927) Case V paired-comparison scaling, as implemented by
/// psych's `thurstone()` (see `mlsirm_core::scaling`). `choice` is a flat
/// row-major n*n matrix; returns dict with scale, gof, model, residual.
#[pyfunction]
#[pyo3(signature = (choice, n))]
fn thurstone_case_v(
    py: Python<'_>,
    choice: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::thurstone_case_v(choice.as_slice()?, n)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("scale", PyArray1::from_slice(py, &res.scale))?;
    d.set_item("gof", res.gof)?;
    d.set_item("model", PyArray1::from_slice(py, &res.model))?;
    d.set_item("residual", PyArray1::from_slice(py, &res.residual))?;
    Ok(d.into())
}

/// Bradley-Terry maximum-likelihood worths via Hunter's MM algorithm as
/// implemented by choix 0.4.1 (`opt.mm` pairwise path; see
/// `mlsirm_core::scaling::bradley_terry_mm`). `wins` is a flat row-major
/// n*n count matrix (wins[i*n+j] = times i beat j); returns dict with
/// params (centered log-worths), weights (exp scale, sum n), iterations.
#[pyfunction]
#[pyo3(signature = (wins, n, alpha=0.0, max_iter=10000, tol=1e-8))]
fn bradley_terry_mm(
    py: Python<'_>,
    wins: PyReadonlyArray1<'_, f64>,
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::bradley_terry_mm(wins.as_slice()?, n, alpha, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Luce Spectral Ranking (one-shot) log-worths from a dense pairwise
/// win-count matrix, as implemented by choix 0.4.1 (`lsr_pairwise_dense`;
/// see `mlsirm_core::scaling::lsr_pairwise`). Returns dict with params
/// (centered log-worths), weights (stationary distribution, sum n),
/// iterations (always 1).
#[pyfunction]
#[pyo3(signature = (wins, n, alpha=0.0))]
fn lsr_pairwise(
    py: Python<'_>,
    wins: PyReadonlyArray1<'_, f64>,
    n: usize,
    alpha: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::lsr_pairwise(wins.as_slice()?, n, alpha)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Iterative Luce Spectral Ranking (Bradley-Terry MLE) as implemented by
/// choix 0.4.1 (`ilsr_pairwise_dense`; see
/// `mlsirm_core::scaling::ilsr_pairwise`). Same dict layout as
/// `lsr_pairwise`; iterations is the LSR pass count at convergence.
#[pyfunction]
#[pyo3(signature = (wins, n, alpha=0.0, max_iter=100, tol=1e-8))]
fn ilsr_pairwise(
    py: Python<'_>,
    wins: PyReadonlyArray1<'_, f64>,
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::ilsr_pairwise(wins.as_slice()?, n, alpha, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Rank Centrality (win-ratio spectral ranking) as implemented by choix
/// 0.4.1 (`rank_centrality`; see
/// `mlsirm_core::scaling::rank_centrality`). Same dict layout as
/// `lsr_pairwise`; iterations is always 1.
#[pyfunction]
#[pyo3(signature = (wins, n, alpha=0.0))]
fn rank_centrality(
    py: Python<'_>,
    wins: PyReadonlyArray1<'_, f64>,
    n: usize,
    alpha: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::rank_centrality(wins.as_slice()?, n, alpha)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Plackett-Luce Luce-Spectral-Ranking for full/partial rankings (one
/// shot) as implemented by choix 0.4.1 (`lsr_rankings`; see
/// `mlsirm_core::scaling::lsr_rankings`). `rankings` is CSR-flattened
/// item indices (best first per ranking); `starts` are the CSR offsets.
/// Same dict layout as `lsr_pairwise`.
#[pyfunction]
#[pyo3(signature = (rankings, starts, n, alpha=0.0))]
fn lsr_rankings(
    py: Python<'_>,
    rankings: PyReadonlyArray1<'_, u64>,
    starts: PyReadonlyArray1<'_, u64>,
    n: usize,
    alpha: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let rk: Vec<usize> = rankings.as_slice()?.iter().map(|&x| x as usize).collect();
    let st: Vec<usize> = starts.as_slice()?.iter().map(|&x| x as usize).collect();
    let res =
        mlsirm_core::scaling::lsr_rankings(&rk, &st, n, alpha).map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Iterative LSR (I-LSR) for rankings, as implemented by choix 0.4.1
/// (`ilsr_rankings`; see `mlsirm_core::scaling::ilsr_rankings`).
#[pyfunction]
#[pyo3(signature = (rankings, starts, n, alpha=0.0, max_iter=100, tol=1e-8))]
fn ilsr_rankings(
    py: Python<'_>,
    rankings: PyReadonlyArray1<'_, u64>,
    starts: PyReadonlyArray1<'_, u64>,
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let rk: Vec<usize> = rankings.as_slice()?.iter().map(|&x| x as usize).collect();
    let st: Vec<usize> = starts.as_slice()?.iter().map(|&x| x as usize).collect();
    let res = mlsirm_core::scaling::ilsr_rankings(&rk, &st, n, alpha, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Plackett-Luce Luce-Spectral-Ranking for top-1 choice data (one
/// shot) as implemented by choix 0.4.1 (`lsr_top1`; see
/// `mlsirm_core::scaling::lsr_top1`). `winners` has one entry per
/// observation; `losers` is CSR-flattened loser indices with `starts`
/// offsets. Same dict layout as `lsr_pairwise`.
#[pyfunction]
#[pyo3(signature = (winners, losers, starts, n, alpha=0.0))]
fn lsr_top1(
    py: Python<'_>,
    winners: PyReadonlyArray1<'_, u64>,
    losers: PyReadonlyArray1<'_, u64>,
    starts: PyReadonlyArray1<'_, u64>,
    n: usize,
    alpha: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let wn: Vec<usize> = winners.as_slice()?.iter().map(|&x| x as usize).collect();
    let ls: Vec<usize> = losers.as_slice()?.iter().map(|&x| x as usize).collect();
    let st: Vec<usize> = starts.as_slice()?.iter().map(|&x| x as usize).collect();
    let res =
        mlsirm_core::scaling::lsr_top1(&wn, &ls, &st, n, alpha).map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Iterative LSR (I-LSR) for top-1 choice data, as implemented by choix
/// 0.4.1 (`ilsr_top1`; see `mlsirm_core::scaling::ilsr_top1`).
#[pyfunction]
#[pyo3(signature = (winners, losers, starts, n, alpha=0.0, max_iter=100, tol=1e-8))]
fn ilsr_top1(
    py: Python<'_>,
    winners: PyReadonlyArray1<'_, u64>,
    losers: PyReadonlyArray1<'_, u64>,
    starts: PyReadonlyArray1<'_, u64>,
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let wn: Vec<usize> = winners.as_slice()?.iter().map(|&x| x as usize).collect();
    let ls: Vec<usize> = losers.as_slice()?.iter().map(|&x| x as usize).collect();
    let st: Vec<usize> = starts.as_slice()?.iter().map(|&x| x as usize).collect();
    let res = mlsirm_core::scaling::ilsr_top1(&wn, &ls, &st, n, alpha, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("params", PyArray1::from_slice(py, &res.params))?;
    d.set_item("weights", PyArray1::from_slice(py, &res.weights))?;
    d.set_item("iterations", res.iterations as u64)?;
    Ok(d.into())
}

/// Kendall & Babington Smith (1940) circular-triad consistency test for a
/// single judge's paired comparisons, as implemented by eba's `circular()`
/// (see `mlsirm_core::scaling::circular_triads`). `mat` is a flat row-major
/// n*n 0/1 preference matrix; returns dict with T, T_max, T_exp, zeta, chi2,
/// df, p_value, exact.
#[pyfunction]
#[pyo3(signature = (mat, n, alternative="two.sided", correct=true))]
fn circular_triads(
    py: Python<'_>,
    mat: PyReadonlyArray1<'_, f64>,
    n: usize,
    alternative: &str,
    correct: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::circular_triads(mat.as_slice()?, n, alternative, correct)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("t", res.t)?;
    d.set_item("t_max", res.t_max)?;
    d.set_item("t_exp", res.t_exp)?;
    d.set_item("zeta", res.zeta)?;
    d.set_item("chi2", res.chi2)?;
    d.set_item("df", res.df)?;
    d.set_item("p_value", res.p_value)?;
    d.set_item("exact", res.exact)?;
    Ok(d.into())
}

/// Kendall's coefficient of agreement u between m judges, as implemented by
/// eba's `kendall.u()` (see `mlsirm_core::scaling::kendall_u`). `mat` is a
/// flat row-major n*n frequency matrix; returns dict with sigma, u, min_u,
/// chi2 (raw, may be negative), df, p_value.
#[pyfunction]
#[pyo3(signature = (mat, n, correct=true))]
fn kendall_u(
    py: Python<'_>,
    mat: PyReadonlyArray1<'_, f64>,
    n: usize,
    correct: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::kendall_u(mat.as_slice()?, n, correct)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("sigma", res.sigma)?;
    d.set_item("u", res.u)?;
    d.set_item("min_u", res.min_u)?;
    d.set_item("chi2", res.chi2)?;
    d.set_item("df", res.df)?;
    d.set_item("p_value", res.p_value)?;
    Ok(d.into())
}

/// Elo ratings from a game schedule, PlayerRatings `elo()` semantics (see
/// `mlsirm_core::scaling::elo_rating`). Batch-per-period update; returns
/// dict with ratings, games, wins, draws, losses, lag.
#[pyfunction]
#[pyo3(signature = (periods, white, black, score, gamma, n, init, kfac))]
#[allow(clippy::too_many_arguments)]
fn elo_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, u64>,
    black: PyReadonlyArray1<'_, u64>,
    score: PyReadonlyArray1<'_, f64>,
    gamma: PyReadonlyArray1<'_, f64>,
    n: usize,
    init: f64,
    kfac: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let white: Vec<usize> = white.as_slice()?.iter().map(|&v| v as usize).collect();
    let black: Vec<usize> = black.as_slice()?.iter().map(|&v| v as usize).collect();
    let res = mlsirm_core::scaling::elo_rating(
        periods.as_slice()?,
        &white,
        &black,
        score.as_slice()?,
        gamma.as_slice()?,
        n,
        init,
        kfac,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("wins", PyArray1::from_slice(py, &res.wins))?;
    d.set_item("draws", PyArray1::from_slice(py, &res.draws))?;
    d.set_item("losses", PyArray1::from_slice(py, &res.losses))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    Ok(d.into())
}

/// Glicko ratings from a game schedule, PlayerRatings `glicko()` semantics
/// (see `mlsirm_core::scaling::glicko_rating`). Batch-per-period update with
/// deviation inflation; returns dict with ratings, deviations, games, wins,
/// draws, losses, lag.
#[pyfunction]
#[pyo3(signature = (periods, white, black, score, gamma, init_rating, init_dev, cval, rdmax))]
#[allow(clippy::too_many_arguments)]
fn glicko_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, u64>,
    black: PyReadonlyArray1<'_, u64>,
    score: PyReadonlyArray1<'_, f64>,
    gamma: PyReadonlyArray1<'_, f64>,
    init_rating: PyReadonlyArray1<'_, f64>,
    init_dev: PyReadonlyArray1<'_, f64>,
    cval: f64,
    rdmax: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let white: Vec<usize> = white.as_slice()?.iter().map(|&v| v as usize).collect();
    let black: Vec<usize> = black.as_slice()?.iter().map(|&v| v as usize).collect();
    let res = mlsirm_core::scaling::glicko_rating(
        periods.as_slice()?,
        &white,
        &black,
        score.as_slice()?,
        gamma.as_slice()?,
        init_rating.as_slice()?,
        init_dev.as_slice()?,
        cval,
        rdmax,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("deviations", PyArray1::from_slice(py, &res.deviations))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("wins", PyArray1::from_slice(py, &res.wins))?;
    d.set_item("draws", PyArray1::from_slice(py, &res.draws))?;
    d.set_item("losses", PyArray1::from_slice(py, &res.losses))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    Ok(d.into())
}

/// Glicko-2 ratings from a game schedule, PlayerRatings `glicko2()` semantics
/// with Glickman's (2022) Illinois volatility step (see
/// `mlsirm_core::scaling::glicko2_rating`). Returns dict with ratings,
/// deviations, volatilities, games, wins, draws, losses, lag.
#[pyfunction]
#[pyo3(signature = (periods, white, black, score, gamma, init_rating, init_dev, init_vol, tau, rdmax))]
#[allow(clippy::too_many_arguments)]
fn glicko2_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, u64>,
    black: PyReadonlyArray1<'_, u64>,
    score: PyReadonlyArray1<'_, f64>,
    gamma: PyReadonlyArray1<'_, f64>,
    init_rating: PyReadonlyArray1<'_, f64>,
    init_dev: PyReadonlyArray1<'_, f64>,
    init_vol: PyReadonlyArray1<'_, f64>,
    tau: f64,
    rdmax: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    // usize::try_from (not `as`): a u64 id above usize::MAX must fail
    // loudly on 32-bit targets instead of truncating into a valid index.
    let white: Vec<usize> = white
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| {
            PyValueError::new_err("glicko2_rating: player index exceeds platform usize")
        })?;
    let black: Vec<usize> = black
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| {
            PyValueError::new_err("glicko2_rating: player index exceeds platform usize")
        })?;
    let res = mlsirm_core::scaling::glicko2_rating(
        periods.as_slice()?,
        &white,
        &black,
        score.as_slice()?,
        gamma.as_slice()?,
        init_rating.as_slice()?,
        init_dev.as_slice()?,
        init_vol.as_slice()?,
        tau,
        rdmax,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("deviations", PyArray1::from_slice(py, &res.deviations))?;
    d.set_item("volatilities", PyArray1::from_slice(py, &res.volatilities))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("wins", PyArray1::from_slice(py, &res.wins))?;
    d.set_item("draws", PyArray1::from_slice(py, &res.draws))?;
    d.set_item("losses", PyArray1::from_slice(py, &res.losses))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    Ok(d.into())
}

/// Stephenson rating for a game schedule, PlayerRatings `steph()` semantics
/// (see `mlsirm_core::scaling::stephenson_rating`). Returns dict with
/// ratings, deviations, games, wins, draws, losses, lag.
#[pyfunction]
#[pyo3(signature = (periods, white, black, score, gamma, init_rating, init_dev, init_games, init_lag, cval, hval, bval, lambda_, rdmax))]
#[allow(clippy::too_many_arguments)]
fn stephenson_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, u64>,
    black: PyReadonlyArray1<'_, u64>,
    score: PyReadonlyArray1<'_, f64>,
    gamma: PyReadonlyArray1<'_, f64>,
    init_rating: PyReadonlyArray1<'_, f64>,
    init_dev: PyReadonlyArray1<'_, f64>,
    init_games: PyReadonlyArray1<'_, u64>,
    init_lag: PyReadonlyArray1<'_, u64>,
    cval: f64,
    hval: f64,
    bval: f64,
    lambda_: f64,
    rdmax: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    // usize::try_from (not `as`): a u64 id above usize::MAX must fail
    // loudly on 32-bit targets instead of truncating into a valid index.
    let white: Vec<usize> = white
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| {
            PyValueError::new_err("stephenson_rating: player index exceeds platform usize")
        })?;
    let black: Vec<usize> = black
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| {
            PyValueError::new_err("stephenson_rating: player index exceeds platform usize")
        })?;
    let res = mlsirm_core::scaling::stephenson_rating(
        periods.as_slice()?,
        &white,
        &black,
        score.as_slice()?,
        gamma.as_slice()?,
        init_rating.as_slice()?,
        init_dev.as_slice()?,
        init_games.as_slice()?,
        init_lag.as_slice()?,
        cval,
        hval,
        bval,
        lambda_,
        rdmax,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("deviations", PyArray1::from_slice(py, &res.deviations))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("wins", PyArray1::from_slice(py, &res.wins))?;
    d.set_item("draws", PyArray1::from_slice(py, &res.draws))?;
    d.set_item("losses", PyArray1::from_slice(py, &res.losses))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    Ok(d.into())
}

/// Multiplayer Elo rating for nn-player events, PlayerRatings `elom()`
/// semantics (see `mlsirm_core::scaling::elom_rating`). `players` and
/// `scores` are flattened g x nn (row-major); empty seats are player -1
/// with NaN score. `kfac_mode` is "scalar" (uses `kfac_k`) or "kriichi"
/// (uses `kfac_gv`/`kfac_kv`). Returns dict with ratings, games, places
/// (flattened n x nn), lag.
#[pyfunction]
#[pyo3(signature = (periods, players, scores, base, init_ratings, init_games, init_lag, init_places, kfac_mode, kfac_k, kfac_gv, kfac_kv, placing))]
#[allow(clippy::too_many_arguments)]
fn elom_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    players: PyReadonlyArray1<'_, i64>,
    scores: PyReadonlyArray1<'_, f64>,
    base: PyReadonlyArray1<'_, f64>,
    init_ratings: PyReadonlyArray1<'_, f64>,
    init_games: PyReadonlyArray1<'_, u64>,
    init_lag: PyReadonlyArray1<'_, u64>,
    init_places: PyReadonlyArray1<'_, u64>,
    kfac_mode: &str,
    kfac_k: f64,
    kfac_gv: f64,
    kfac_kv: f64,
    placing: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let kfac = match kfac_mode {
        "scalar" => mlsirm_core::scaling::ElomKFactor::Scalar(kfac_k),
        "kriichi" => mlsirm_core::scaling::ElomKFactor::Kriichi {
            gv: kfac_gv,
            kv: kfac_kv,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "elom_rating: kfac_mode {:?} must be \"scalar\" or \"kriichi\"",
                other
            )))
        }
    };
    let res = mlsirm_core::scaling::elom_rating(
        periods.as_slice()?,
        players.as_slice()?,
        scores.as_slice()?,
        base.as_slice()?,
        init_ratings.as_slice()?,
        init_games.as_slice()?,
        init_lag.as_slice()?,
        init_places.as_slice()?,
        kfac,
        placing,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("places", PyArray1::from_slice(py, &res.places))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    Ok(d.into())
}

/// Prediction-quality metrics for binary-outcome forecasts, PlayerRatings
/// `metrics()` semantics (see `mlsirm_core::scaling::metrics_rating`).
/// `pred` is flattened row-major nr x np; the return value is the
/// flattened row-major np x 3 matrix of per-column [bdev, mse, mae].
#[pyfunction]
#[pyo3(signature = (act, pred, nr, np, cap_lo, cap_hi, scale))]
fn metrics_rating(
    py: Python<'_>,
    act: PyReadonlyArray1<'_, f64>,
    pred: PyReadonlyArray1<'_, f64>,
    nr: usize,
    np: usize,
    cap_lo: f64,
    cap_hi: f64,
    scale: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = mlsirm_core::scaling::metrics_rating(
        act.as_slice()?,
        pred.as_slice()?,
        nr,
        np,
        (cap_lo, cap_hi),
        scale,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_slice(py, &out).into())
}

/// FIDE-style Elo rating with the kfide K-factor schedule, PlayerRatings
/// `fide()` semantics (see `mlsirm_core::scaling::fide_rating`). Returns a
/// dict with ratings, games, wins, draws, losses, lag, elite (0/1), and
/// opponent (running mean of post-update opponent ratings).
#[pyfunction]
#[pyo3(signature = (periods, white, black, score, gamma, n, init, kv0, kv1, kv2))]
#[allow(clippy::too_many_arguments)]
fn fide_rating(
    py: Python<'_>,
    periods: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, u64>,
    black: PyReadonlyArray1<'_, u64>,
    score: PyReadonlyArray1<'_, f64>,
    gamma: PyReadonlyArray1<'_, f64>,
    n: usize,
    init: f64,
    kv0: f64,
    kv1: f64,
    kv2: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let white: Vec<usize> = white
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| PyValueError::new_err("fide_rating: player index exceeds platform usize"))?;
    let black: Vec<usize> = black
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| PyValueError::new_err("fide_rating: player index exceeds platform usize"))?;
    let res = mlsirm_core::scaling::fide_rating(
        periods.as_slice()?,
        &white,
        &black,
        score.as_slice()?,
        gamma.as_slice()?,
        n,
        init,
        (kv0, kv1, kv2),
    )
    .map_err(PyValueError::new_err)?;
    let elite: Vec<u64> = res.elite.iter().map(|&v| v as u64).collect();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("ratings", PyArray1::from_slice(py, &res.ratings))?;
    d.set_item("games", PyArray1::from_slice(py, &res.games))?;
    d.set_item("wins", PyArray1::from_slice(py, &res.wins))?;
    d.set_item("draws", PyArray1::from_slice(py, &res.draws))?;
    d.set_item("losses", PyArray1::from_slice(py, &res.losses))?;
    d.set_item("lag", PyArray1::from_slice(py, &res.lag))?;
    d.set_item("elite", PyArray1::from_slice(py, &elite))?;
    d.set_item("opponent", PyArray1::from_slice(py, &res.opponent))?;
    Ok(d.into())
}

/// Predicted game outcomes from fitted ratings, PlayerRatings
/// `predict.rating` two-player branches (see
/// `mlsirm_core::scaling::predict_rating_two`). `white`/`black` are player
/// indices with -1 = unmatched. Pass `deviations` for the
/// Glicko/Glicko-2/Stephenson deviation-shrunk branch.
#[pyfunction]
#[pyo3(signature = (ratings, deviations, games, white, black, gamma, tng, trat_rating, trat_deviation, thresh))]
#[allow(clippy::too_many_arguments)]
fn predict_rating_two(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    deviations: Option<PyReadonlyArray1<'_, f64>>,
    games: PyReadonlyArray1<'_, u64>,
    white: PyReadonlyArray1<'_, i64>,
    black: PyReadonlyArray1<'_, i64>,
    gamma: PyReadonlyArray1<'_, f64>,
    tng: u64,
    trat_rating: Option<f64>,
    trat_deviation: Option<f64>,
    thresh: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let dev_slice = deviations.as_ref().map(|d| d.as_slice()).transpose()?;
    let trat = trat_rating.map(|t1| (t1, trat_deviation.unwrap_or(f64::NAN)));
    let out = mlsirm_core::scaling::predict_rating_two(
        ratings.as_slice()?,
        dev_slice,
        games.as_slice()?,
        white.as_slice()?,
        black.as_slice()?,
        gamma.as_slice()?,
        tng,
        trat,
        thresh,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_slice(py, &out).into())
}

/// Predicted expected scores for multi-player (EloM) events, PlayerRatings
/// `predict.rating` EloM branch (see
/// `mlsirm_core::scaling::predict_rating_multi`). `players` is flattened
/// row-major nr x np with -1 = empty seat; `placing` returns min-tie ranks.
#[pyfunction]
#[pyo3(signature = (ratings, games, players, nr, np, tng, trat, placing))]
#[allow(clippy::too_many_arguments)]
fn predict_rating_multi(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, f64>,
    games: PyReadonlyArray1<'_, u64>,
    players: PyReadonlyArray1<'_, i64>,
    nr: usize,
    np: usize,
    tng: u64,
    trat: Option<f64>,
    placing: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = mlsirm_core::scaling::predict_rating_multi(
        ratings.as_slice()?,
        games.as_slice()?,
        players.as_slice()?,
        nr,
        np,
        tng,
        trat,
        placing,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_slice(py, &out).into())
}

/// Bradley-Terry model with ties (additive alpha0, VGAM `bratt`) fitted
/// by MM (see `mlsirm_core::scaling::bratt_mm`). `wins` and `ties` are
/// flat row-major n*n matrices (ties symmetric); returns dict with alpha
/// (worths, alpha[ref_index] == ref_value), alpha0 (tie parameter),
/// iterations, log_likelihood.
#[pyfunction]
#[pyo3(signature = (wins, ties, n, ref_index=0, ref_value=1.0, max_iter=10000, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn bratt_mm(
    py: Python<'_>,
    wins: PyReadonlyArray1<'_, f64>,
    ties: PyReadonlyArray1<'_, f64>,
    n: usize,
    ref_index: usize,
    ref_value: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::scaling::bratt_mm(
        wins.as_slice()?,
        ties.as_slice()?,
        n,
        ref_index,
        ref_value,
        max_iter,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("alpha", PyArray1::from_slice(py, &res.alpha))?;
    d.set_item("alpha0", res.alpha0)?;
    d.set_item("iterations", res.iterations as u64)?;
    d.set_item("log_likelihood", res.log_likelihood)?;
    Ok(d.into())
}

/// Fleiss' kappa for nominal agreement among nr raters over ns subjects,
/// with the exact (Conger) variant (irr 0.85 `kappam.fleiss`; see
/// `mlsirm_core::agreement::fleiss_kappa`). `ratings` is flat row-major
/// ns*nr of codes 0..k-1; negative = missing (listwise row drop). Returns
/// dict with kappa, subjects_used, z, p_value, category_kappa/z/p
/// (empty arrays and NaN z/p in exact mode).
#[pyfunction]
#[pyo3(signature = (ratings, ns, nr, k, exact=false))]
fn fleiss_kappa(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, i64>,
    ns: usize,
    nr: usize,
    k: usize,
    exact: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::agreement::fleiss_kappa(ratings.as_slice()?, ns, nr, k, exact)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("kappa", res.kappa)?;
    d.set_item("subjects_used", res.subjects_used as u64)?;
    d.set_item("z", res.z)?;
    d.set_item("p_value", res.p_value)?;
    d.set_item(
        "category_kappa",
        PyArray1::from_slice(py, &res.category_kappa),
    )?;
    d.set_item("category_z", PyArray1::from_slice(py, &res.category_z))?;
    d.set_item("category_p", PyArray1::from_slice(py, &res.category_p))?;
    Ok(d.into())
}

/// Light's kappa: mean pairwise unweighted Cohen's kappa with Light's
/// chance-product z test (irr 0.85 `kappam.light` + unweighted `kappa2`;
/// see `mlsirm_core::agreement::light_kappa`). `ratings` is flat row-major
/// ns*nr of integer category codes; negative = missing (listwise row drop).
/// Returns dict with value, subjects_used, raters, kappas, z, p_value.
#[pyfunction]
#[pyo3(signature = (ratings, ns, nr))]
fn light_kappa(
    py: Python<'_>,
    ratings: PyReadonlyArray1<'_, i64>,
    ns: usize,
    nr: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::agreement::light_kappa(ratings.as_slice()?, ns, nr)
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("value", res.value)?;
    d.set_item("subjects_used", res.subjects_used as u64)?;
    d.set_item("raters", res.raters as u64)?;
    d.set_item("kappas", PyArray1::from_slice(py, &res.kappas))?;
    d.set_item("z", res.z)?;
    d.set_item("p_value", res.p_value)?;
    Ok(d.into())
}

/// GPCM/nominal softmax cell log-probabilities at one node (parity surface for
/// the NumPy `category_logprobs` reference).
#[pyfunction]
#[pyo3(signature = (base, scores, intercepts))]
fn gpcm_cell_logprobs(
    base: f64,
    scores: PyReadonlyArray1<'_, f64>,
    intercepts: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    Ok(core_gpcm_logprobs(
        base,
        scores.as_slice()?,
        intercepts.as_slice()?,
    ))
}

/// GRM cumulative-logit cell log-probabilities at one node.
#[pyfunction]
#[pyo3(signature = (base, thresholds))]
fn grm_cell_logprobs(base: f64, thresholds: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(core_grm_logprobs(base, thresholds.as_slice()?))
}

/// Batched public GRM/GPCM category probabilities and expected category scores.
#[pyfunction]
#[pyo3(signature = (theta, slope, cat_params, n_items, n_cat, model))]
fn polytomous_predictions(
    py: Python<'_>,
    theta: PyReadonlyArray1<'_, f64>,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    n_items: usize,
    n_cat: usize,
    model: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let result = mlsirm_core::poly::polytomous_predictions(
        theta.as_slice()?,
        slope.as_slice()?,
        cat_params.as_slice()?,
        n_items,
        n_cat,
        parse_poly_model(model)?,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("probabilities", PyArray1::from_vec(py, result.probabilities))?;
    out.set_item("expected", PyArray1::from_vec(py, result.expected))?;
    Ok(out.into())
}

/// Unidimensional polytomous marginal-EM fit (Rust compute path). `model` is
/// "grm" (default) or "gpcm"; `y` holds integer categories `0..n_cat-1`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, observed = None, model = "grm", q_theta = 21, max_iter = 80, tol = 1e-6))]
fn fit_poly_unidim(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let fit = core_fit_poly_unidim(
        &yv, obs, n_persons, n_items, n_cat, m, q_theta, max_iter, tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", fit.slope)?;
    out.set_item("cat_params", fit.cat_params)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("loglik_trace", fit.loglik_trace)?;
    out.set_item("final_delta", fit.final_delta)?;
    out.set_item("stopping_tolerance", fit.stopping_tolerance)?;
    Ok(out.into())
}

/// Unidimensional nominal categories model fit (Rust compute path). Returns a
/// dict with `scores` and `intercepts` (each `n_items` lists of `n_cat-1` free
/// values, baseline `a_0=c_0=0`), plus `loglik`/`n_iter`.
///
/// References (APA 7th ed.):
///   Bock, R. D. (1972). Estimating item parameters and latent ability when
///     responses are scored in two or more nominal categories. Psychometrika,
///     37(1), 29-51. https://doi.org/10.1007/BF02291411
///   Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item
///     response model. In Handbook of polytomous item response theory models
///     (pp. 43-75). Routledge.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, observed = None, q_theta = 21, max_iter = 200, tol = 1e-6))]
fn fit_nominal(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let fit = core_fit_nominal(&yv, obs, n_persons, n_items, n_cat, q_theta, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("scores", fit.scores)?;
    out.set_item("intercepts", fit.intercepts)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("loglik_trace", fit.loglik_trace)?;
    out.set_item("final_delta", fit.final_delta)?;
    out.set_item("stopping_tolerance", fit.stopping_tolerance)?;
    Ok(out.into())
}

/// Polytomous person-fit l_z / l_z* (Rust compute path). Returns a dict with
/// per-person `lz`, `lz_star`, `theta_eap`, and `flagged` (l_z* < threshold).
///
/// References (APA 7th ed.):
///   Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
///     measurement with polychotomous item response models and standardized
///     indices. British Journal of Mathematical and Statistical Psychology,
///     38(1), 67-86. https://doi.org/10.1111/j.2044-8317.1985.tb00817.x
///   Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
///     statistics with estimated person parameter. Psychometrika, 66(3),
///     331-342. https://doi.org/10.1007/BF02294437
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", q_theta = 21, prior_mean = 0.0, prior_sd = 1.0, flag_threshold = -1.645))]
fn poly_person_fit(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
    prior_mean: f64,
    prior_sd: f64,
    flag_threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let res = core_poly_person_fit(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        q_theta,
        prior_mean,
        prior_sd,
        flag_threshold,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("lz", res.lz)?;
    out.set_item("lz_star", res.lz_star)?;
    out.set_item("theta_eap", res.theta_eap)?;
    out.set_item("flagged", res.flagged)?;
    Ok(out.into())
}

/// Simulate a polytomous computerized adaptive test (Rust compute path). Returns
/// a dict with per-simulee `theta_eap`, `theta_sd` (final CAT SE), and `n_used`.
///
/// References (APA 7th ed.):
///   Dodd, B. G., De Ayala, R. J., & Koch, W. R. (1995). Computerized adaptive
///     testing with polytomous items. Applied Psychological Measurement, 19(1),
///     5-22. https://doi.org/10.1177/014662169501900103
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (true_theta, slope, cat_params, n_items, n_cat, model = "grm", q_theta = 21, se_threshold = 0.3, min_items = 5, max_items = 30, adaptive = true, seed = 0))]
fn poly_cat_simulate(
    py: Python<'_>,
    true_theta: PyReadonlyArray1<'_, f64>,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    n_items: usize,
    n_cat: usize,
    model: &str,
    q_theta: usize,
    se_threshold: f64,
    min_items: usize,
    max_items: usize,
    adaptive: bool,
    seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let res = core_poly_cat_simulate(
        true_theta.as_slice()?,
        slope.as_slice()?,
        cat_params.as_slice()?,
        n_items,
        n_cat,
        m,
        q_theta,
        se_threshold,
        min_items,
        max_items,
        adaptive,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_eap", res.theta_eap)?;
    out.set_item("theta_sd", res.theta_sd)?;
    out.set_item("n_used", res.n_used)?;
    Ok(out.into())
}

/// Warm's (1989) weighted-likelihood ability estimates for a unidimensional POLYTOMOUS test, GRM or
/// GPCM (Rust compute path). The polytomous counterpart of `score_wle`: solves
/// `dlnL/dtheta + J(theta)/(2 I(theta)) = 0` with `I = sum_k P'_k^2 / P_k` and
/// `J = sum_k P'_k P''_k / P_k` summed over the person's observed items — the exact generalization of
/// the dichotomous `sum_i P' P''/(P Q)`, which is its two-category case. `J` is computed DIRECTLY, not
/// as a derivative of `I`.
///
/// Unlike `score_poly_eap` this applies NO prior, so the estimate is not shrunk toward a population
/// mean — the usual requirement when reporting individual scores. It stays FINITE for the all-lowest
/// and all-highest response patterns, where the maximum-likelihood estimate diverges.
///
/// PCM is this GPCM path with `slope = 1`. RSM is NOT supported: its fitted `(delta, shared tau)`
/// parameterization is not convertible through any exposed API.
///
/// The correction is confirmed against the `catR` package's source rather than a primary paper; see
/// the core `score_wle_poly` docs for the full verification status, including the in-repository proof
/// that `J = I'` holds for both shipped families (used only as a test oracle, never as a shortcut).
///
/// `y` is row-major `n_persons * n_items` with categories in `0..n_cat`; `cat_params` is flattened
/// `n_items * (n_cat - 1)`. Returns a dict with `theta`, `se` and `boundary`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", theta_bound = 20.0, tol = 1e-8))]
fn score_wle_poly(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    theta_bound: f64,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let out_scores = core_score_wle_poly(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        theta_bound,
        tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta", out_scores.theta)?;
    out.set_item("se", out_scores.se)?;
    out.set_item("boundary", out_scores.boundary)?;
    Ok(out.into())
}

/// EAP trait scores from polytomous responses given fitted item parameters
/// (Rust compute path). Returns a dict with `theta_eap` and `theta_sd`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", q_theta = 21))]
fn score_poly_eap(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let (eap, sd) = core_score_poly_eap(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        q_theta,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_eap", eap)?;
    out.set_item("theta_sd", sd)?;
    Ok(out.into())
}

/// Polytomous item information curves: flattened `n_theta * n_items` I_i(theta).
#[pyfunction]
#[pyo3(signature = (theta, slope, cat_params, n_items, n_cat, model = "grm"))]
fn poly_information_curves(
    theta: PyReadonlyArray1<'_, f64>,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    n_items: usize,
    n_cat: usize,
    model: &str,
) -> PyResult<Vec<f64>> {
    let m = parse_poly_model(model)?;
    core_poly_information_curves(
        theta.as_slice()?,
        slope.as_slice()?,
        cat_params.as_slice()?,
        n_items,
        n_cat,
        m,
    )
    .map_err(PyValueError::new_err)
}

/// Generalized S-X2 polytomous item fit (Rust compute path). Returns a dict with
/// per-item `statistic`, `df`, `p_value`, and `n_cells` (the retained cell count,
/// the reference df at KNOWN parameters).
///
/// References (APA 7th ed.):
///   Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X² item
///     fit index for polytomous IRT models. Journal of Educational Measurement,
///     45(4), 391-406. https://doi.org/10.1111/j.1745-3984.2008.00070.x
///   Kang, T., & Chen, T. T. (2011). Performance of the generalized S-X² item
///     fit index for the graded response model. Asia Pacific Education Review,
///     12(1), 89-96. https://doi.org/10.1007/s12564-010-9082-4
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", q_theta = 21, min_expected = 1.0))]
fn poly_item_fit_sx2(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
    min_expected: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let res = core_poly_s_x2(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        q_theta,
        min_expected,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("statistic", res.statistic)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("n_cells", res.n_cells)?;
    Ok(out.into())
}

/// Latent-space polytomous LSIRM fit (Rust compute path). Returns a dict of
/// item parameters (`slope`, `cat_params`, `zeta`) and person scores
/// (`theta_eap`, `theta_sd`, `xi_eap`), plus `loglik`/`n_iter`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, latent_dim, observed = None, model = "grm", q_theta = 11, q_xi = 11, max_iter = 60, tol = 1e-5))]
fn fit_poly_lsirm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    latent_dim: usize,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
    q_xi: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let fit = core_fit_poly_lsirm(
        &yv, obs, n_persons, n_items, n_cat, latent_dim, m, q_theta, q_xi, max_iter, tol,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", fit.slope)?;
    out.set_item("cat_params", fit.cat_params)?;
    out.set_item("zeta", fit.zeta)?;
    out.set_item("theta_eap", fit.theta_eap)?;
    out.set_item("theta_sd", fit.theta_sd)?;
    out.set_item("xi_eap", fit.xi_eap)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("n_iter", fit.n_iter)?;
    Ok(out.into())
}

/// Per-item mixed-format marginal MLE (Rust multithreaded CPU path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y,
    n_persons,
    n_items,
    item_models,
    n_categories,
    observed = None,
    latent_dim = 2,
    q_theta = 21,
    q_xi = 7,
    max_iter = 100,
    tol = 1e-5,
    n_threads = 0
))]
fn fit_mixed_items(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    item_models: Vec<String>,
    n_categories: PyReadonlyArray1<'_, i64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    latent_dim: usize,
    q_theta: usize,
    q_xi: usize,
    max_iter: usize,
    tol: f64,
    n_threads: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let raw_y = y.as_slice()?;
    let expected_len = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| PyValueError::new_err("n_persons * n_items overflow"))?;
    if raw_y.len() != expected_len {
        return Err(PyValueError::new_err(
            "y must have length n_persons * n_items",
        ));
    }
    if item_models.len() != n_items {
        return Err(PyValueError::new_err(
            "item_models length must match n_items",
        ));
    }
    let raw_categories = n_categories.as_slice()?;
    if raw_categories.len() != n_items {
        return Err(PyValueError::new_err(
            "n_categories length must match n_items",
        ));
    }
    let yv = raw_y
        .iter()
        .map(|&value| {
            if value < 0 {
                Err(PyValueError::new_err(
                    "responses must be non-negative integer categories",
                ))
            } else {
                Ok(value as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let specs = item_models
        .iter()
        .zip(raw_categories)
        .enumerate()
        .map(|(item, (model, &n_cat))| {
            if n_cat < 2 {
                return Err(PyValueError::new_err(format!(
                    "item {item}: n_categories must be >= 2"
                )));
            }
            let kind = MixedItemKind::parse(model).map_err(PyValueError::new_err)?;
            Ok(MixedItemSpec {
                kind,
                n_categories: n_cat as usize,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mask = observed
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()?;
    let fit = core_fit_mixed_items(
        &yv, mask, n_persons, n_items, &specs, latent_dim, q_theta, q_xi, max_iter, tol, n_threads,
    )
    .map_err(PyValueError::new_err)?;

    let out = pyo3::types::PyDict::new(py);
    let items = pyo3::types::PyList::empty(py);
    for estimate in fit.items {
        let item = pyo3::types::PyDict::new(py);
        item.set_item("model", estimate.kind.as_str())?;
        item.set_item("n_categories", estimate.n_categories)?;
        item.set_item("slope", estimate.slope)?;
        item.set_item("intercepts", estimate.intercepts)?;
        item.set_item("thresholds", estimate.thresholds)?;
        item.set_item("scores", estimate.scores)?;
        item.set_item("location", estimate.location)?;
        item.set_item("lower_asymptote", estimate.lower_asymptote)?;
        item.set_item("upper_asymptote", estimate.upper_asymptote)?;
        item.set_item("zeta", estimate.zeta)?;
        items.append(item)?;
    }
    out.set_item("items", items)?;
    out.set_item("theta_eap", fit.theta_eap)?;
    out.set_item("theta_sd", fit.theta_sd)?;
    out.set_item("xi_eap", fit.xi_eap)?;
    out.set_item("latent_dim", fit.latent_dim)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("loglik_trace", fit.loglik_trace)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("n_threads", fit.n_threads)?;
    Ok(out.into())
}

/// Lognormal response-time model (van der Linden, 2007; Rust compute path).
/// `times` is `n_persons * n_items` row-major raw response times (`> 0` where
/// observed). Returns a dict with item `alpha`/`beta`, `sigma_tau`, per-person
/// `tau_eap`/`tau_sd`, `loglik`, `n_iter`, `converged`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (times, observed, n_persons, n_items, max_iter = 500, tol = 1e-6, var_floor = 1e-4, sigma_floor = 1e-4, fix_sigma_tau = None))]
fn fit_rt_lognormal(
    py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
    var_floor: f64,
    sigma_floor: f64,
    fix_sigma_tau: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let cfg = RtConfig {
        max_iter,
        tol,
        var_floor,
        sigma_floor,
        fix_sigma_tau,
    };
    let fit = core_fit_rt(times.as_slice()?, obs, n_persons, n_items, cfg)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("alpha", fit.alpha)?;
    out.set_item("beta", fit.beta)?;
    out.set_item("mu_tau", fit.mu_tau)?;
    out.set_item("sigma_tau", fit.sigma_tau)?;
    out.set_item("tau_eap", fit.tau_eap)?;
    out.set_item("tau_sd", fit.tau_sd)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("loglik_trace", fit.loglik_trace)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("final_loglik_change", fit.final_loglik_change)?;
    Ok(out.into())
}

/// van der Linden (2007) Level-2 joint speed-accuracy person covariance (two-stage;
/// item params fixed). `responses` (0/1) and `times` (`> 0` where observed) are
/// row-major `n_persons * n_items`; `a`/`b` are the 2PL raw slope/intercept,
/// `alpha`/`beta` the lognormal time discrimination/intensity. Returns a dict with
/// `rho`, `sigma_tau`, `s_theta2`, per-person `theta_eap`/`tau_eap`, `loglik`,
/// `n_iter`, `converged`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (responses, times, observed, a, b, alpha, beta, n_persons, n_items, q = 21, max_iter = 500, tol = 1e-6, fix_sigma_tau = None))]
fn fit_speed_accuracy_covariance(
    py: Python<'_>,
    responses: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    beta: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
    q: usize,
    max_iter: usize,
    tol: f64,
    fix_sigma_tau: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let cfg = SpeedAccuracyConfig {
        q,
        max_iter,
        tol,
        fix_sigma_tau,
        ..Default::default()
    };
    let fit = core_fit_sa(
        responses.as_slice()?,
        times.as_slice()?,
        obs,
        a.as_slice()?,
        b.as_slice()?,
        alpha.as_slice()?,
        beta.as_slice()?,
        n_persons,
        n_items,
        cfg,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("rho", fit.rho)?;
    out.set_item("sigma_tau", fit.sigma_tau)?;
    out.set_item("s_theta2", fit.s_theta2)?;
    out.set_item("theta_eap", fit.theta_eap)?;
    out.set_item("tau_eap", fit.tau_eap)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("loglik_trace", fit.loglik_trace)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("termination_reason", fit.termination_reason)?;
    out.set_item("final_loglik_change", fit.final_loglik_change)?;
    Ok(out.into())
}

/// Response-time person fit (van der Linden & Guo, 2008; Rust compute path).
/// `times` (`> 0` where observed) is row-major `n_persons * n_items`; `alpha`/`beta`
/// come from a fitted lognormal RT model. Returns a dict with per-person `w`
/// (`chi2(n-1)`), `df`, `l_t`, `p_value`, `flagged`, `tau_ml`, and
/// `n_persons*n_items` `z_resid`/`item_flag`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (times, observed, n_persons, n_items, alpha, beta, alpha_level = 0.05, z_fast = 1.645))]
fn rt_person_fit(
    py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    n_persons: usize,
    n_items: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    beta: PyReadonlyArray1<'_, f64>,
    alpha_level: f64,
    z_fast: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let res = core_rt_person_fit(
        times.as_slice()?,
        obs,
        n_persons,
        n_items,
        alpha.as_slice()?,
        beta.as_slice()?,
        alpha_level,
        z_fast,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("w", res.w)?;
    out.set_item("df", res.df)?;
    out.set_item("l_t", res.l_t)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("flagged", res.flagged)?;
    out.set_item("tau_ml", res.tau_ml)?;
    out.set_item("z_resid", res.z_resid)?;
    out.set_item("item_flag", res.item_flag)?;
    Ok(out.into())
}

/// M2 limited-information goodness-of-fit with RMSEA2 (+90% CI) and SRMSR.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0,
))]
fn m2_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let res = core_m2(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        q_theta,
        rule,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("m2", res.m2)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("rmsea2", res.rmsea2)?;
    out.set_item("rmsea2_ci_lower", res.rmsea2_ci_lower)?;
    out.set_item("rmsea2_ci_upper", res.rmsea2_ci_upper)?;
    out.set_item("srmsr", res.srmsr)?;
    out.set_item("null_m2", res.null_m2)?;
    out.set_item("null_df", res.null_df)?;
    out.set_item("cfi", res.cfi)?;
    out.set_item("tli", res.tli)?;
    out.set_item("n_moments", res.n_moments)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("n_complete", res.n_complete)?;
    Ok(out.into())
}

/// Structured single-population M2 with Rust-owned calibration bookkeeping.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0,
    fixed_items = None, estimate_population = false, tau_fixed = false,
))]
fn m2_structured_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    fixed_items: Option<PyReadonlyArray1<'_, bool>>,
    estimate_population: bool,
    tau_fixed: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let fixed = fixed_items
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()?;
    let res = core_m2_structured(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        q_theta,
        rule,
        fixed,
        estimate_population,
        tau_fixed,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("m2", res.m2)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("rmsea2", res.rmsea2)?;
    out.set_item("rmsea2_ci_lower", res.rmsea2_ci_lower)?;
    out.set_item("rmsea2_ci_upper", res.rmsea2_ci_upper)?;
    out.set_item("srmsr", res.srmsr)?;
    out.set_item("null_m2", res.null_m2)?;
    out.set_item("null_df", res.null_df)?;
    out.set_item("cfi", res.cfi)?;
    out.set_item("tli", res.tli)?;
    out.set_item("n_moments", res.n_moments)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("n_complete", res.n_complete)?;
    Ok(out.into())
}



/// Projected M2 quadratic form ownership entrypoint (dense residual / Delta / Xi).
fn decode_m2_item_sets(values: &[i64], offsets: &[i64]) -> PyResult<Vec<Vec<usize>>> {
    if offsets.is_empty() || offsets[0] != 0 {
        return Err(PyValueError::new_err(
            "item-set offsets must start at zero",
        ));
    }
    let final_offset = usize::try_from(*offsets.last().unwrap_or(&-1))
        .map_err(|_| PyValueError::new_err("item-set offsets must be non-negative"))?;
    if final_offset != values.len() {
        return Err(PyValueError::new_err(
            "item-set offsets must end at the item-value length",
        ));
    }
    let mut item_sets = Vec::with_capacity(offsets.len().saturating_sub(1));
    for window in offsets.windows(2) {
        let start = usize::try_from(window[0])
            .map_err(|_| PyValueError::new_err("item-set offsets must be non-negative"))?;
        let end = usize::try_from(window[1])
            .map_err(|_| PyValueError::new_err("item-set offsets must be non-negative"))?;
        if end < start || end > values.len() {
            return Err(PyValueError::new_err("item-set offsets must be monotone"));
        }
        let mut item_set = Vec::with_capacity(end - start);
        for &value in &values[start..end] {
            item_set.push(
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("item-set indices must be non-negative"))?,
            );
        }
        item_sets.push(item_set);
    }
    Ok(item_sets)
}

/// Rust-owned simple-structure M2 moment integration for one population.
#[pyfunction]
#[pyo3(signature = (probs, trait_weights, space_weights, q_theta, factor_id, item_values, item_offsets))]
fn factorized_trait_moments_stat(
    probs: PyReadonlyArray1<'_, f64>,
    trait_weights: PyReadonlyArray1<'_, f64>,
    space_weights: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    factor_id: PyReadonlyArray1<'_, i64>,
    item_values: PyReadonlyArray1<'_, i64>,
    item_offsets: PyReadonlyArray1<'_, i64>,
) -> PyResult<Vec<f64>> {
    let factor_id = factor_id
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("factor_id values must be non-negative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let n_dims = factor_id.iter().copied().max().map_or(Ok(0), |maximum| {
        maximum
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("factor_id dimension overflows"))
    })?;
    let item_sets = decode_m2_item_sets(item_values.as_slice()?, item_offsets.as_slice()?)?;
    core_factorized_trait_moments(
        probs.as_slice()?,
        trait_weights.as_slice()?,
        space_weights.as_slice()?,
        q_theta,
        &factor_id,
        n_dims,
        &item_sets,
    )
    .map_err(PyValueError::new_err)
}

/// Rust-owned shared-cluster M2 moment integration for a multilevel population.
#[pyfunction]
#[pyo3(signature = (probs, cluster_weights, trait_weights, space_weights, q_u, q_theta, factor_id, item_values, item_offsets))]
fn factorized_multilevel_moments_stat(
    probs: PyReadonlyArray1<'_, f64>,
    cluster_weights: PyReadonlyArray1<'_, f64>,
    trait_weights: PyReadonlyArray1<'_, f64>,
    space_weights: PyReadonlyArray1<'_, f64>,
    q_u: usize,
    q_theta: usize,
    factor_id: PyReadonlyArray1<'_, i64>,
    item_values: PyReadonlyArray1<'_, i64>,
    item_offsets: PyReadonlyArray1<'_, i64>,
) -> PyResult<Vec<f64>> {
    let factor_id = factor_id
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("factor_id values must be non-negative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let n_dims = factor_id.iter().copied().max().map_or(Ok(0), |maximum| {
        maximum
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("factor_id dimension overflows"))
    })?;
    let item_sets = decode_m2_item_sets(item_values.as_slice()?, item_offsets.as_slice()?)?;
    core_factorized_multilevel_moments(
        probs.as_slice()?,
        cluster_weights.as_slice()?,
        trait_weights.as_slice()?,
        space_weights.as_slice()?,
        q_u,
        q_theta,
        &factor_id,
        n_dims,
        &item_sets,
    )
    .map_err(PyValueError::new_err)
}

/// Rust-owned cluster-total covariance for multilevel M2 moments.
#[pyfunction]
#[pyo3(signature = (z_rows, model_moments, cluster_id, n_rows, n_moments, n_clusters))]
fn cluster_moment_covariance_stat(
    z_rows: PyReadonlyArray1<'_, f64>,
    model_moments: PyReadonlyArray1<'_, f64>,
    cluster_id: PyReadonlyArray1<'_, i64>,
    n_rows: usize,
    n_moments: usize,
    n_clusters: usize,
) -> PyResult<Vec<f64>> {
    let cluster_id = cluster_id
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("cluster ids must be non-negative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    core_cluster_moment_covariance(
        z_rows.as_slice()?,
        model_moments.as_slice()?,
        &cluster_id,
        n_rows,
        n_moments,
        n_clusters,
    )
    .map_err(PyValueError::new_err)
}

/// Projected M2 quadratic form ownership entrypoint (dense residual / Delta / Xi).
#[pyfunction]
fn projected_m2(
    residual: PyReadonlyArray1<'_, f64>,
    delta: PyReadonlyArray2<'_, f64>,
    xi: PyReadonlyArray2<'_, f64>,
    n: f64,
) -> PyResult<f64> {
    let residual_arr = residual.as_array();
    let delta_arr = delta.as_array();
    let xi_arr = xi.as_array();
    let s = residual_arr.len();
    if delta_arr.shape()[0] != s {
        return Err(PyValueError::new_err(
            "delta rows must match residual length",
        ));
    }
    if xi_arr.shape() != [s, s] {
        return Err(PyValueError::new_err("xi must be residual_len x residual_len"));
    }
    let p = delta_arr.shape()[1];
    let workspace_elements = core_projected_m2_workspace_elements(s, p)
        .map_err(PyValueError::new_err)?;
    if workspace_elements > PROJECTED_M2_MAX_WORKSPACE_ELEMENTS {
        return Err(PyValueError::new_err(
            "projected M2 workspace exceeds supported element budget",
        ));
    }
    if !n.is_finite() {
        return Err(PyValueError::new_err("n must be finite"));
    }
    if residual_arr.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("residual values must be finite"));
    }
    if delta_arr.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("delta values must be finite"));
    }
    if xi_arr.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("xi values must be finite"));
    }
    let residual = residual.as_slice()?;
    let delta_elements = s
        .checked_mul(p)
        .ok_or_else(|| PyValueError::new_err("projected M2 dimensions overflow"))?;
    let xi_elements = s
        .checked_mul(s)
        .ok_or_else(|| PyValueError::new_err("projected M2 dimensions overflow"))?;
    let mut delta_flat = vec![0.0_f64; delta_elements];
    for row in 0..s {
        for col in 0..p {
            delta_flat[row * p + col] = delta_arr[[row, col]];
        }
    }
    let mut xi_flat = vec![0.0_f64; xi_elements];
    for row in 0..s {
        for col in 0..s {
            xi_flat[row * s + col] = xi_arr[[row, col]];
        }
    }
    core_projected_m2(residual, &delta_flat, xi_flat, s, p, n).map_err(PyValueError::new_err)
}

/// Conditional-Rasch M2 (CMLE ownership path).
#[pyfunction]
#[pyo3(signature = (y, observed, n_persons, item_easiness))]
fn m2_cmle_rasch_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    item_easiness: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = core_m2_cmle_rasch(
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        item_easiness.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("m2", res.m2)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("rmsea2", res.rmsea2)?;
    out.set_item("rmsea2_ci_lower", res.rmsea2_ci_lower)?;
    out.set_item("rmsea2_ci_upper", res.rmsea2_ci_upper)?;
    out.set_item("srmsr", res.srmsr)?;
    out.set_item("null_m2", res.null_m2)?;
    out.set_item("null_df", res.null_df)?;
    out.set_item("cfi", res.cfi)?;
    out.set_item("tli", res.tli)?;
    out.set_item("n_moments", res.n_moments)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("n_complete", res.n_complete)?;
    Ok(out.into())
}

/// Polytomous M2 limited-information goodness-of-fit (Rust compute path) for a
/// fitted unidimensional GRM/GPCM. Returns m2, df, p_value, rmsea2 (+90% CI),
/// srmsr, null-model M2/df, CFI/TLIRT, and the bookkeeping counts.
///
/// References (APA 7th ed.):
///   Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in
///     categorical data analysis. Multivariate Behavioral Research, 49(4),
///     305-328. https://doi.org/10.1080/00273171.2014.911075
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", q_theta = 21))]
fn poly_m2(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let res = core_poly_m2(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        q_theta,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("m2", res.m2)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("rmsea2", res.rmsea2)?;
    out.set_item("rmsea2_ci_lower", res.rmsea2_ci_lower)?;
    out.set_item("rmsea2_ci_upper", res.rmsea2_ci_upper)?;
    out.set_item("srmsr", res.srmsr)?;
    out.set_item("null_m2", res.null_m2)?;
    out.set_item("null_df", res.null_df)?;
    out.set_item("cfi", res.cfi)?;
    out.set_item("tli", res.tli)?;
    out.set_item("n_moments", res.n_moments)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("n_complete", res.n_complete)?;
    Ok(out.into())
}

/// Polytomous item-pair local-dependence diagnostics (Rust compute path).
/// Returns a dict of per-pair arrays (`item_i`, `item_j`, `x2`, `g2`, `p_value`,
/// `cramers_v`, `max_abs_std_resid`, `n_pair`) plus the shared `df = (K-1)^2`.
///
/// References (APA 7th ed.):
///   Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs
///     using item response theory. Journal of Educational and Behavioral
///     Statistics, 22(3), 265-289. https://doi.org/10.3102/10769986022003265
///   Liu, Y., & Maydeu-Olivares, A. (2013). Local dependence diagnostics in IRT
///     modeling of binary data. Educational and Psychological Measurement,
///     73(2), 254-274. https://doi.org/10.1177/0013164412453841
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, observed = None, model = "grm", q_theta = 21))]
fn poly_local_dependence(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let res = core_poly_ld(
        &yv,
        obs,
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        q_theta,
    )
    .map_err(PyValueError::new_err)?;
    let item_i: Vec<usize> = res.pairs.iter().map(|&(i, _)| i).collect();
    let item_j: Vec<usize> = res.pairs.iter().map(|&(_, j)| j).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_i", item_i)?;
    out.set_item("item_j", item_j)?;
    out.set_item("x2", res.x2)?;
    out.set_item("g2", res.g2)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("cramers_v", res.cramers_v)?;
    out.set_item("max_abs_std_resid", res.max_abs_std_resid)?;
    out.set_item("n_pair", res.n_pair)?;
    Ok(out.into())
}

/// Likelihood-ratio DIF sweep for polytomous items via two-group marginal EM
/// (Rust compute path). Fits a compact model (all items group-invariant) once,
/// then per studied item an augmented model (that item freed per group);
/// `LR = 2 dloglik ~ chi2((n_groups-1) * n_cat)`. Impact (genuine group ability
/// differences) is absorbed by estimating each group's latent distribution in
/// both models. Returns a dict of per-item arrays (`item`, `lr`, `df`,
/// `p_value`, `flagged_bh`, `effect_size`).
///
/// References (APA 7th ed.):
///   Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential
///     item functioning using the parameters of item response models. In P. W.
///     Holland & H. Wainer (Eds.), Differential item functioning (pp. 67-113).
///     Erlbaum.
///   Woehr, D. J., & Meriac, J. P. (2010). Using polytomous item response theory
///     to examine differential item and test functioning. In N. T. Tippins &
///     S. Adler (Eds.), Technology-enhanced assessment of talent (pp. 199-229).
///     Jossey-Bass.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, group_id, n_groups, n_persons, n_items, n_cat, observed = None,
    model = "gpcm", studied_items = None, q_theta = 21, max_iter = 200, tol = 1e-5, fdr_q = 0.05,
))]
fn poly_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group_id: PyReadonlyArray1<'_, i64>,
    n_groups: usize,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    model: &str,
    studied_items: Option<PyReadonlyArray1<'_, i64>>,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
    fdr_q: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let gid: Vec<usize> = group_id
        .as_slice()?
        .iter()
        .map(|&g| {
            if g < 0 {
                Err(PyValueError::new_err("group_id must be non-negative"))
            } else {
                Ok(g as usize)
            }
        })
        .collect::<PyResult<_>>()?;
    let studied_storage: Option<Vec<usize>> = match &studied_items {
        Some(s) => Some(
            s.as_slice()?
                .iter()
                .map(|&j| {
                    if j < 0 {
                        Err(PyValueError::new_err("studied_items must be non-negative"))
                    } else {
                        Ok(j as usize)
                    }
                })
                .collect::<PyResult<_>>()?,
        ),
        None => None,
    };
    let rows = core_poly_dif(
        &yv,
        obs,
        &gid,
        n_groups,
        n_persons,
        n_items,
        n_cat,
        m,
        studied_storage.as_deref(),
        q_theta,
        max_iter,
        tol,
        fdr_q,
    )
    .map_err(PyValueError::new_err)?;
    let item: Vec<usize> = rows.iter().map(|r| r.item).collect();
    let lr: Vec<f64> = rows.iter().map(|r| r.lr).collect();
    let df: Vec<usize> = rows.iter().map(|r| r.df).collect();
    let p_value: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let flagged: Vec<bool> = rows.iter().map(|r| r.flagged_bh).collect();
    let effect: Vec<f64> = rows.iter().map(|r| r.effect_size).collect();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item", item)?;
    out.set_item("lr", lr)?;
    out.set_item("df", df)?;
    out.set_item("p_value", p_value)?;
    out.set_item("flagged_bh", flagged)?;
    out.set_item("effect_size", effect)?;
    Ok(out.into())
}

/// Mantel-Haenszel differential item functioning (Rust compute path; Holland & Thayer, 1988). The
/// observed-score, calibration-free DIF test: examinees are matched on the number-correct total
/// (studied item included by default; `exclude_studied_item=True` uses the rest score), and per item a
/// common odds ratio `alpha_MH` is estimated across the `2 x 2` (group x response) tables. `y` is a
/// row-major `n_persons * n_items` `0/1` array; `group` is length `n_persons` with `0` = reference and
/// `1` = focal. Returns a dict of per-item arrays: `item`, `alpha_mh`, `chi2_mh`, `p_value` (chi2 df 1),
/// `mh_d_dif` (ETS delta `-2.35 ln(alpha_MH)`, negative = harder for the focal group), `se_d_dif`
/// (Robins-Breslow-Greenland), `std_p_dif` (Dorans & Kulick, 1986, focal minus reference), `ets_class`
/// (ETS `"A"`/`"B"`/`"C"`, or `"U"` when undefined), and `flagged_bh` (Benjamini-Hochberg at `fdr_q`).
/// NaN statistics / `"U"` mean the item had no DIF-informative strata or a degenerate odds ratio.
///
/// References (APA 7th ed.):
///   Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the standardization approach to
///     assessing unexpected differential item performance on the Scholastic Aptitude Test. Journal of
///     Educational Measurement, 23(4), 355-368.
///   Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel
///     procedure. In H. Wainer & H. I. Braun (Eds.), Test validity (pp. 129-145). Erlbaum.
///   Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel-Haenszel variance
///     consistent in both sparse data and large-strata limiting models. Biometrics, 42(2), 311-323.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, group, n_persons, n_items, exclude_studied_item = false, fdr_q = 0.05))]
fn mantel_haenszel_dif(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    exclude_studied_item: bool,
    fdr_q: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv: Vec<u8> = y
        .as_slice()?
        .iter()
        .map(|&v| match v {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err("y responses must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let gv: Vec<u8> = group
        .as_slice()?
        .iter()
        .map(|&g| match g {
            0 => Ok(0u8),
            1 => Ok(1u8),
            _ => Err(PyValueError::new_err(
                "group labels must be 0 (reference) or 1 (focal)",
            )),
        })
        .collect::<PyResult<_>>()?;
    let cfg = MhDifConfig {
        exclude_studied_item,
        fdr_q,
    };
    let rows = core_mh_dif(&yv, &gv, n_persons, n_items, &cfg).map_err(PyValueError::new_err)?;
    Ok(mh_rows_dict(py, &rows)?.into())
}

/// Uniform SIBTEST (Rust compute path; Shealy & Stout, 1993, as implemented in Chalmers, 2012 — the
/// primary text was not consulted, see the core module notes). The third observed-score DIF procedure,
/// and the only one that corrects the MATCHING CRITERION for measurement error: under impact, two
/// examinees from different groups with the same observed score have different expected TRUE scores, so
/// each group's conditional mean is transported from its own Kelley-regressed true score to a common
/// target before being compared. Item purification cannot substitute for this — it fixes which items
/// are in the criterion, not the regression of true score on observed score.
///
/// Each item in turn is the studied subtest; the valid subtest is every OTHER item, always disjoint.
/// Returns per-item arrays `item`, `beta_uni`, `se_beta`, `b_uni`, `p_value`, `alpha_ref`,
/// `alpha_focal`, `n_strata_used`, `flagged_bh`.
///
/// SIGN WARNING: `beta_uni > 0` means harder for the FOCAL group — the OPPOSITE orientation to
/// `mantel_haenszel_dif`'s `mh_d_dif` and `std_p_dif`, which go negative in that same case.
#[pyfunction]
#[pyo3(signature = (y, group, n_persons, n_items, fdr_q = 0.05, j_min = 5))]
fn sibtest(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    fdr_q: f64,
    j_min: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv = binary_u8(y.as_slice()?)?;
    let gv = binary_u8(group.as_slice()?)?;
    let cfg = SibtestConfig { fdr_q, j_min };
    let rows = core_sibtest(&yv, &gv, n_persons, n_items, &cfg).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item", rows.iter().map(|r| r.item).collect::<Vec<_>>())?;
    out.set_item(
        "beta_uni",
        rows.iter().map(|r| r.beta_uni).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "se_beta",
        rows.iter().map(|r| r.se_beta).collect::<Vec<_>>(),
    )?;
    out.set_item("b_uni", rows.iter().map(|r| r.b_uni).collect::<Vec<_>>())?;
    out.set_item(
        "p_value",
        rows.iter().map(|r| r.p_value).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "alpha_ref",
        rows.iter().map(|r| r.alpha_ref).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "alpha_focal",
        rows.iter().map(|r| r.alpha_focal).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n_strata_used",
        rows.iter().map(|r| r.n_strata_used).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "flagged_bh",
        rows.iter().map(|r| r.flagged_bh).collect::<Vec<_>>(),
    )?;
    Ok(out.into())
}

/// Raju (1988/1990) signed/unsigned ICC-area DIF for 2PL (or common-c 3PL)
/// items already linked to a common scale (Rust compute path; formula oracle:
/// difR `RajuZ.R`/`difRaju.R` READ in full — the primary Raju papers were NOT
/// read; both areas and all delta-method partials were re-derived by hand and
/// quadrature/FD-verified in adversarial spec review; see the core module).
///
/// Signed: `h = b_foc - b_ref` (positive = harder for focal). Unsigned:
/// `h = |H|`, the total area between the ICCs. `z = H / se(H)` from unscaled
/// quantities; for common-c 3PL, `h` and `se` are reported scaled by `(1-c)`.
/// Returns arrays `h`, `se`, `z`, `p_value` plus `dif_items` and `signed`.
#[pyfunction]
#[pyo3(signature = (a_ref, b_ref, se_a_ref, se_b_ref, cov_ab_ref, a_foc, b_foc, se_a_foc, se_b_foc, cov_ab_foc, guess = None, signed = false, alpha = 0.05))]
#[allow(clippy::too_many_arguments)]
fn raju_area(
    py: Python<'_>,
    a_ref: PyReadonlyArray1<'_, f64>,
    b_ref: PyReadonlyArray1<'_, f64>,
    se_a_ref: PyReadonlyArray1<'_, f64>,
    se_b_ref: PyReadonlyArray1<'_, f64>,
    cov_ab_ref: PyReadonlyArray1<'_, f64>,
    a_foc: PyReadonlyArray1<'_, f64>,
    b_foc: PyReadonlyArray1<'_, f64>,
    se_a_foc: PyReadonlyArray1<'_, f64>,
    se_b_foc: PyReadonlyArray1<'_, f64>,
    cov_ab_foc: PyReadonlyArray1<'_, f64>,
    guess: Option<PyReadonlyArray1<'_, f64>>,
    signed: bool,
    alpha: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let guess_vec = match &guess {
        Some(g) => Some(g.as_slice()?.to_vec()),
        None => None,
    };
    let res = core_raju_area(
        a_ref.as_slice()?,
        b_ref.as_slice()?,
        se_a_ref.as_slice()?,
        se_b_ref.as_slice()?,
        cov_ab_ref.as_slice()?,
        a_foc.as_slice()?,
        b_foc.as_slice()?,
        se_a_foc.as_slice()?,
        se_b_foc.as_slice()?,
        cov_ab_foc.as_slice()?,
        guess_vec.as_deref(),
        signed,
        alpha,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("h", numpy::PyArray1::from_slice(py, &res.h))?;
    out.set_item("se", numpy::PyArray1::from_slice(py, &res.se))?;
    out.set_item("z", numpy::PyArray1::from_slice(py, &res.z))?;
    out.set_item("p_value", numpy::PyArray1::from_slice(py, &res.p_value))?;
    out.set_item("dif_items", res.dif_items)?;
    out.set_item("signed", res.signed)?;
    Ok(out.into())
}

/// Per-item arrays for a Mantel-Haenszel sweep, shared by the plain and purified entry points.
fn mh_rows_dict<'py>(
    py: Python<'py>,
    rows: &[MhDifRow],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item", rows.iter().map(|r| r.item).collect::<Vec<_>>())?;
    out.set_item(
        "alpha_mh",
        rows.iter().map(|r| r.alpha_mh).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "chi2_mh",
        rows.iter().map(|r| r.chi2_mh).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "p_value",
        rows.iter().map(|r| r.p_value).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "mh_d_dif",
        rows.iter().map(|r| r.mh_d_dif).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "se_d_dif",
        rows.iter().map(|r| r.se_d_dif).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "std_p_dif",
        rows.iter().map(|r| r.std_p_dif).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "ets_class",
        rows.iter()
            .map(|r| r.ets_class.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "flagged_bh",
        rows.iter().map(|r| r.flagged_bh).collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// Mantel-Haenszel DIF with an ITERATIVELY PURIFIED matching criterion (Rust compute path; Candell &
/// Drasgow, 1988; Clauser, Mazor & Hambleton, 1993). The criterion is rebuilt from the currently
/// unflagged (anchor) items and the sweep re-run until the flagged set stabilises or `max_rounds` is
/// reached, which reduces the contamination the raw number-correct total suffers when it contains the
/// very items under test. Returns the same per-item arrays as `mantel_haenszel_dif` plus `anchor`
/// (bool per item), `n_anchor`, `rounds`, `purify_converged` (scalar), and
/// `purify_termination_reason`.
///
/// IMPORTANT: the anchor is selected from the same data that is then tested against it, so the returned
/// p-values are conditional on a data-dependent selection. They are NOT guaranteed super-uniform under
/// the null and Benjamini-Hochberg does NOT control the FDR at `fdr_q` for a purified sweep — treat
/// `flagged_bh` here as a screening device. Purification reduces rather than removes contamination and
/// can fail when DIF is unbalanced in direction (Wang & Su, 2004); Mantel-Haenszel is also blind to
/// crossing DIF, so a purely non-uniform item stays in the anchor and keeps contaminating it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, group, n_persons, n_items, exclude_studied_item = false, fdr_q = 0.05, max_rounds = 3, min_anchor_items = 4))]
fn mantel_haenszel_dif_purified(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    exclude_studied_item: bool,
    fdr_q: f64,
    max_rounds: usize,
    min_anchor_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv = binary_u8(y.as_slice()?)?;
    let gv = binary_u8(group.as_slice()?)?;
    let cfg = MhDifConfig {
        exclude_studied_item,
        fdr_q,
    };
    let purify = PurifyConfig {
        max_rounds,
        min_anchor_items,
    };
    let res = core_mh_purified(&yv, &gv, n_persons, n_items, &cfg, &purify)
        .map_err(PyValueError::new_err)?;
    let out = mh_rows_dict(py, &res.rows)?;
    purify_meta(
        &out,
        res.anchor,
        res.n_anchor,
        res.rounds,
        res.converged,
        res.termination_reason,
    )?;
    Ok(out.into())
}

/// Zumbo logistic-regression DIF with an ITERATIVELY PURIFIED matching criterion (Rust compute path).
/// Same purification loop as `mantel_haenszel_dif_purified`, with the flag taken from `jg_class` (the
/// 2-df omnibus test). Returns the `logistic_dif` per-item arrays — including its PER-ITEM `converged`
/// array — plus `anchor`, `n_anchor`, `rounds`, scalar `purify_converged`, and
/// `purify_termination_reason`. The same caveat
/// applies: the anchor is data-selected, so the p-values carry no FDR guarantee and the flags are a
/// screening device.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, group, n_persons, n_items, exclude_studied_item = false, fdr_q = 0.05, max_iter = 50, max_rounds = 3, min_anchor_items = 4))]
fn logistic_dif_purified(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    group: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    exclude_studied_item: bool,
    fdr_q: f64,
    max_iter: usize,
    max_rounds: usize,
    min_anchor_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let yv = binary_u8(y.as_slice()?)?;
    let gv = binary_u8(group.as_slice()?)?;
    let cfg = LogisticDifConfig {
        exclude_studied_item,
        fdr_q,
        max_iter,
    };
    let purify = PurifyConfig {
        max_rounds,
        min_anchor_items,
    };
    let res = core_logistic_purified(&yv, &gv, n_persons, n_items, &cfg, &purify)
        .map_err(PyValueError::new_err)?;
    let out = logistic_rows_dict(py, &res.rows)?;
    purify_meta(
        &out,
        res.anchor,
        res.n_anchor,
        res.rounds,
        res.converged,
        res.termination_reason,
    )?;
    Ok(out.into())
}

/// Nonparametric polytomous person-fit U3poly (Rust compute path). Generalizes
/// van der Flier's U3 to ordered polytomous items via sample item-step response
/// functions; no fitted IRT model. Returns a dict of per-person arrays
/// (`u3poly` in [0,1], `total_score`, `flagged`); NaN where undefined. `cutoff`
/// (see `u3_bootstrap_cutoff`) flags `u3poly >= cutoff`.
///
/// References (APA 7th ed.):
///   Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous
///     item scores. Applied Psychological Measurement, 32(3), 224-247.
///     https://doi.org/10.1177/0146621607302479
#[pyfunction]
#[pyo3(signature = (y, n_persons, n_items, n_cat, observed = None, cutoff = None))]
fn u3_person_fit(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    observed: Option<PyReadonlyArray1<'_, bool>>,
    cutoff: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let yv = poly_responses(y.as_slice()?, obs, n_cat)?;
    let res = core_u3_poly_person_fit(&yv, obs, n_persons, n_items, n_cat, cutoff)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("u3poly", res.u3poly)?;
    out.set_item("total_score", res.total_score)?;
    out.set_item("flagged", res.flagged)?;
    Ok(out.into())
}

/// Simulated (1-alpha) critical value for `u3_person_fit` via a parametric
/// bootstrap from a fitted GRM/GPCM at theta ~ N(0,1) (Rust compute path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n_persons, n_items, n_cat, slope, cat_params, model = "gpcm", alpha = 0.05, n_rep = 200, seed = 0))]
fn u3_bootstrap_cutoff(
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    model: &str,
    alpha: f64,
    n_rep: usize,
    seed: u64,
) -> PyResult<f64> {
    let m = parse_poly_model(model)?;
    core_u3_poly_cutoff(
        n_persons,
        n_items,
        n_cat,
        slope.as_slice()?,
        cat_params.as_slice()?,
        m,
        alpha,
        n_rep,
        seed,
    )
    .map_err(PyValueError::new_err)
}

/// l_z / Snijders l_z* person fit at EAP estimates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, theta, xi, prior_mean = None, flag_threshold = -1.645,
))]
fn person_fit_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    theta: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray1<'_, f64>,
    prior_mean: Option<PyReadonlyArray1<'_, f64>>,
    flag_threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let pm_storage = match &prior_mean {
        Some(v) => v.as_slice()?.to_vec(),
        None => Vec::new(),
    };
    let res = core_person_fit(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        theta.as_slice()?,
        xi.as_slice()?,
        &pm_storage,
        flag_threshold,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("lz", res.lz)?;
    out.set_item("lz_star", res.lz_star)?;
    out.set_item("flagged", res.flagged)?;
    Ok(out.into())
}

/// Per-item infit/outfit mean squares at EAP estimates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, theta, xi,
))]
fn infit_outfit_stat(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    theta: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let res = core_infit_outfit(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        theta.as_slice()?,
        xi.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("infit", res.infit)?;
    out.set_item("outfit", res.outfit)?;
    Ok(out.into())
}

/// Machine-scoring validation gates (Williamson, Xi & Breyer 2012).
#[pyfunction]
#[pyo3(signature = (auto, human, k, human_a = None, human_b = None, subgroup = None, qwk_min = 0.70, pearson_r_min = 0.70, degradation_max = 0.10, overall_smd_max = 0.15, subgroup_smd_max = 0.10, min_subgroup_n = 2))]
#[allow(clippy::too_many_arguments)]
fn validate_scoring(
    py: Python<'_>,
    auto: PyReadonlyArray1<'_, u32>,
    human: PyReadonlyArray1<'_, u32>,
    k: usize,
    human_a: Option<PyReadonlyArray1<'_, u32>>,
    human_b: Option<PyReadonlyArray1<'_, u32>>,
    subgroup: Option<PyReadonlyArray1<'_, u32>>,
    qwk_min: f64,
    pearson_r_min: f64,
    degradation_max: f64,
    overall_smd_max: f64,
    subgroup_smd_max: f64,
    min_subgroup_n: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let hh_storage = match (&human_a, &human_b) {
        (Some(a), Some(b)) => Some((a.as_slice()?.to_vec(), b.as_slice()?.to_vec())),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "human_a and human_b must be provided together",
            ))
        }
    };
    let sg_storage = match &subgroup {
        Some(g) => Some(g.as_slice()?.to_vec()),
        None => None,
    };
    let thr = ValidationThresholds {
        qwk_min,
        pearson_r_min,
        degradation_max,
        overall_smd_max,
        subgroup_smd_max,
        min_subgroup_n,
    };
    let verdict = core_validate_scoring(
        auto.as_slice()?,
        human.as_slice()?,
        k,
        hh_storage
            .as_ref()
            .map(|(a, b)| (a.as_slice(), b.as_slice())),
        sg_storage.as_deref(),
        thr,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    let gates = pyo3::types::PyList::empty(py);
    for g in &verdict.gates {
        let gd = pyo3::types::PyDict::new(py);
        gd.set_item("name", g.name)?;
        gd.set_item("value", g.value)?;
        gd.set_item("threshold", g.threshold)?;
        gd.set_item("pass", g.pass)?;
        gates.append(gd)?;
    }
    out.set_item("gates", gates)?;
    out.set_item("exact_agreement", verdict.exact_agreement)?;
    out.set_item("adjacent_agreement", verdict.adjacent_agreement)?;
    out.set_item("pass", verdict.pass)?;
    Ok(out.into())
}

/// Vuong non-nested model comparison from casewise log-likelihoods
/// (Schneider et al. 2019).
#[pyfunction]
#[pyo3(signature = (loglik_a, loglik_b, k_a, k_b, bic_correction = true))]
fn vuong_nonnested(
    py: Python<'_>,
    loglik_a: PyReadonlyArray1<'_, f64>,
    loglik_b: PyReadonlyArray1<'_, f64>,
    k_a: usize,
    k_b: usize,
    bic_correction: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res = mlsirm_core::fitstats::vuong_nonnested(
        loglik_a.as_slice()?,
        loglik_b.as_slice()?,
        k_a,
        k_b,
        bic_correction,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("z", res.z)?;
    out.set_item("p_two_sided", res.p_two_sided)?;
    out.set_item("omega", res.omega)?;
    out.set_item("mean_diff", res.mean_diff)?;
    Ok(out.into())
}

/// Yen Q3 residual correlations and a descriptive mean absolute residual
/// cross-product. The legacy `gddm` key remains as an explicitly documented
/// compatibility alias; it is not the published Levy-Svetina GDDM.
#[pyfunction]
fn dimensionality_residuals(
    py: Python<'_>,
    resid: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_items: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let res =
        mlsirm_core::fitstats::dimensionality_residuals(resid.as_slice()?, n_persons, n_items)
            .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("q3", res.q3)?;
    out.set_item("q3_max_abs", res.q3_max_abs)?;
    out.set_item("q3_mean_abs", res.q3_mean_abs)?;
    out.set_item(
        "mean_abs_residual_cross_product",
        res.mean_abs_residual_cross_product,
    )?;
    out.set_item("gddm", res.gddm)?;
    Ok(out.into())
}

/// Oakes-identity observed-information SEs for a fitted marginal model
/// (Pritikin 2017). Population parameters are conditioned on, not
/// differentiated.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, factor_id, n_persons, n_items, n_dims, latent_dim, model,
    eps_distance, alpha, b, zeta, tau, pop_kind = "single", pop_id = None,
    n_pop = 0, mu = None, sigma = None, sigma_u = 0.0, q_theta = 21, q_xi = 11,
    q_u = 15, xi_rule = "gh", xi_points = 256, xi_seed = 0, lambda_b = 0.25,
    lambda_alpha = 1.0, mu_alpha = 0.5, lambda_zeta = 1.0, lambda_tau = 1.0,
    mu_tau = 0.5, h = 1e-5,
))]
fn oakes_standard_errors(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    factor_id: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    latent_dim: usize,
    model: &str,
    eps_distance: f64,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    pop_kind: &str,
    pop_id: Option<PyReadonlyArray1<'_, i64>>,
    n_pop: usize,
    mu: Option<PyReadonlyArray1<'_, f64>>,
    sigma: Option<PyReadonlyArray1<'_, f64>>,
    sigma_u: f64,
    q_theta: usize,
    q_xi: usize,
    q_u: usize,
    xi_rule: &str,
    xi_points: usize,
    xi_seed: u64,
    lambda_b: f64,
    lambda_alpha: f64,
    mu_alpha: f64,
    lambda_zeta: f64,
    lambda_tau: f64,
    mu_tau: f64,
    h: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: parse_model_type(model)?,
        eps_distance,
    };
    let factors = convert_factor_id(factor_id.as_slice()?, n_dims)?;
    let ids: Option<Vec<usize>> = match &pop_id {
        Some(arr) => Some(
            arr.as_slice()?
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .map_err(|_| PyValueError::new_err("population ids must be >= 0"))
                })
                .collect::<PyResult<Vec<usize>>>()?,
        ),
        None => None,
    };
    let pop = match pop_kind {
        "single" => PopulationSpec::Single,
        "singlefree" => PopulationSpec::SingleFree,
        "multigroup" => PopulationSpec::Multigroup {
            group_id: ids.ok_or_else(|| PyValueError::new_err("multigroup requires pop_id"))?,
            n_groups: n_pop,
        },
        "multilevel" => PopulationSpec::Multilevel {
            cluster_id: ids.ok_or_else(|| PyValueError::new_err("multilevel requires pop_id"))?,
            n_clusters: n_pop,
        },
        _ => {
            return Err(PyValueError::new_err(
                "pop_kind must be one of ['single', 'singlefree', 'multigroup', 'multilevel']",
            ))
        }
    };
    let rule = XiRuleKind::parse(xi_rule)
        .ok_or_else(|| PyValueError::new_err("xi_rule must be one of ['gh', 'qmc', 'mc']"))?;
    let mcfg = MarginalConfig {
        q_theta,
        q_xi,
        q_u,
        xi_rule: rule,
        xi_points,
        xi_seed,
        ..MarginalConfig::default()
    };
    let penalty = PenaltyConfig {
        lambda_b,
        lambda_alpha,
        mu_alpha,
        lambda_zeta,
        lambda_tau,
        mu_tau,
        ..PenaltyConfig::lsirm_prior()
    };
    let mu_v = match &mu {
        Some(v) => v.as_slice()?.to_vec(),
        None => Vec::new(),
    };
    let sigma_v = match &sigma {
        Some(v) => v.as_slice()?.to_vec(),
        None => Vec::new(),
    };
    let res = mlsirm_core::oakes::observed_information_oakes(
        y.as_slice()?,
        observed.as_slice()?,
        &factors,
        &config,
        &pop,
        &mcfg,
        &penalty,
        alpha.as_slice()?,
        b.as_slice()?,
        zeta.as_slice()?,
        tau,
        &mu_v,
        &sigma_v,
        sigma_u,
        h,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("labels", res.labels)?;
    out.set_item("se", res.se)?;
    out.set_item("information", res.information)?;
    Ok(out.into())
}

/// Rust-owned CAT MLE ability estimation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    xi_mean, administered, responses, alpha, b, zeta, tau, factor_id, model,
    n_dims, latent_dim, eps_distance, start = None, max_iter = 50, tol = 1e-6,
    bound = 6.0, device = "auto",
))]
fn cat_ability_mle(
    xi_mean: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, i64>,
    responses: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    start: Option<PyReadonlyArray1<'_, f64>>,
    max_iter: usize,
    tol: f64,
    bound: f64,
    device: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<bool>)> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let administered = convert_item_indices(administered.as_slice()?)?;
    let start = start
        .as_ref()
        .map(|values| values.as_slice().map(|slice| slice.to_vec()))
        .transpose()?;
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let result = core_cat_ability_mle_device(
        &bank,
        xi_mean.as_slice()?,
        &administered,
        responses.as_slice()?,
        start.as_deref(),
        max_iter,
        tol,
        bound,
        device,
    )
    .map_err(PyValueError::new_err)?;
    Ok((result.theta, result.se, result.finite))
}

/// Rust-owned CAT EAP ability estimation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    xi_mean, administered, responses, alpha, b, zeta, tau, factor_id, model,
    n_dims, latent_dim, eps_distance, prior_mean, prior_sd, n_quad = 41,
    quad_range = 6.0, device = "auto",
))]
fn cat_ability_eap(
    xi_mean: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, i64>,
    responses: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    n_quad: usize,
    quad_range: f64,
    device: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<bool>)> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let administered = convert_item_indices(administered.as_slice()?)?;
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let result = core_cat_ability_eap_device(
        &bank,
        xi_mean.as_slice()?,
        &administered,
        responses.as_slice()?,
        &prior,
        n_quad,
        quad_range,
        device,
    )
    .map_err(PyValueError::new_err)?;
    Ok((result.theta, result.se, result.finite))
}

/// Rust-owned CAT asymptotic standard-error reduction.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    xi_mean, theta, administered, alpha, b, zeta, tau, factor_id, model,
    n_dims, latent_dim, eps_distance, device = "auto",
))]
fn cat_ability_standard_error(
    xi_mean: PyReadonlyArray1<'_, f64>,
    theta: PyReadonlyArray1<'_, f64>,
    administered: Option<PyReadonlyArray1<'_, i64>>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    device: &str,
) -> PyResult<Vec<f64>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let administered = match &administered {
        Some(values) => Some(convert_item_indices(values.as_slice()?)?),
        None => None,
    };
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    core_cat_ability_standard_error_device(
        &bank,
        xi_mean.as_slice()?,
        theta.as_slice()?,
        administered.as_deref(),
        device,
    )
    .map_err(PyValueError::new_err)
}

/// Dichotomous per-item Fisher information at one theta for CAT/test design.
///
/// Delegates to the same frozen-bank information kernel as ability SE and bank
/// information (van der Linden & Pashley, 2010 2PL form under c=0, d=1).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    theta, xi_mean, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, device = "auto",
))]
fn cat_item_information(
    theta: PyReadonlyArray1<'_, f64>,
    xi_mean: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    device: &str,
) -> PyResult<Vec<f64>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    core_cat_item_information_device(
        &bank,
        theta.as_slice()?,
        xi_mean.as_slice()?,
        device,
    )
    .map_err(PyValueError::new_err)
}

/// Maximum-Fisher-information next-item selection with administered exclusion.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    theta, xi_mean, administered, alpha, b, zeta, tau, factor_id, model, n_dims,
    latent_dim, eps_distance, device = "auto",
))]
fn cat_select_item(
    theta: PyReadonlyArray1<'_, f64>,
    xi_mean: PyReadonlyArray1<'_, f64>,
    administered: Option<PyReadonlyArray1<'_, i64>>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    device: &str,
) -> PyResult<usize> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let administered = match &administered {
        Some(values) => Some(convert_item_indices(values.as_slice()?)?),
        None => None,
    };
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    core_cat_select_item_device(
        &bank,
        theta.as_slice()?,
        xi_mean.as_slice()?,
        administered.as_deref(),
        device,
    )
    .map_err(PyValueError::new_err)
}

/// Greedy maximum-information fixed-form assembly with content constraints.
///
/// Python validates public shapes and marshals maps; ordering, exclusion, and
/// content-feasibility decisions are owned by the Rust numeric core.
#[pyfunction]
#[pyo3(signature = (
    information,
    length,
    content = None,
    min_per_content = None,
    max_per_content = None,
    exclude = None,
))]
fn assemble_test_form_greedy(
    information: PyReadonlyArray1<'_, f64>,
    length: usize,
    content: Option<Vec<String>>,
    min_per_content: Option<HashMap<String, i64>>,
    max_per_content: Option<HashMap<String, i64>>,
    exclude: Option<Vec<i64>>,
) -> PyResult<Vec<i64>> {
    let min_map = min_per_content.unwrap_or_default();
    let max_map = max_per_content.unwrap_or_default();
    let exclude_idx = exclude.unwrap_or_default();
    let content_ref = content.as_deref();
    core_assemble_test_form_greedy(
        information.as_slice()?,
        length,
        content_ref,
        &min_map,
        &max_map,
        &exclude_idx,
    )
    .map_err(PyValueError::new_err)
}

/// Adam optimizer for packed JMLE parameters (Kingma & Ba, 2015).
///
/// `objective(x) -> (obj, grad, loglik)` is evaluated from Python; moment
/// updates and convergence control run in Rust.
#[pyfunction]
#[pyo3(signature = (x0, objective, learning_rate, max_iter, tolerance))]
fn jmle_optimize_adam<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<'_, f64>,
    objective: Bound<'py, PyAny>,
    learning_rate: f64,
    max_iter: usize,
    tolerance: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, String)> {
    let x0_slice = x0.as_slice()?.to_vec();
    let mut obj_cb = |x: &[f64]| -> Result<(f64, Vec<f64>, f64), String> {
        let arr = PyArray1::from_slice(py, x);
        let out = objective
            .call1((arr,))
            .map_err(|e| format!("jmle objective failed: {e}"))?;
        let (obj, grad, loglik): (f64, Vec<f64>, f64) = out
            .extract()
            .map_err(|e| format!("jmle objective must return (float, sequence[float], float): {e}"))?;
        Ok((obj, grad, loglik))
    };
    let (x, obj_t, ll_t, status) =
        core_jmle_adam(&x0_slice, &mut obj_cb, learning_rate, max_iter, tolerance)
            .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_slice(py, &x),
        PyArray1::from_slice(py, &obj_t),
        PyArray1::from_slice(py, &ll_t),
        status,
    ))
}

/// L-BFGS optimizer for packed JMLE parameters (Liu & Nocedal, 1989).
#[pyfunction]
#[pyo3(signature = (x0, objective, max_iter, tolerance, history))]
fn jmle_optimize_lbfgs<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<'_, f64>,
    objective: Bound<'py, PyAny>,
    max_iter: usize,
    tolerance: f64,
    history: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, String)> {
    let x0_slice = x0.as_slice()?.to_vec();
    let mut obj_cb = |x: &[f64]| -> Result<(f64, Vec<f64>, f64), String> {
        let arr = PyArray1::from_slice(py, x);
        let out = objective
            .call1((arr,))
            .map_err(|e| format!("jmle objective failed: {e}"))?;
        let (obj, grad, loglik): (f64, Vec<f64>, f64) = out
            .extract()
            .map_err(|e| format!("jmle objective must return (float, sequence[float], float): {e}"))?;
        Ok((obj, grad, loglik))
    };
    let (x, obj_t, ll_t, status) =
        core_jmle_lbfgs(&x0_slice, &mut obj_cb, max_iter, tolerance, history)
            .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_slice(py, &x),
        PyArray1::from_slice(py, &obj_t),
        PyArray1::from_slice(py, &ll_t),
        status,
    ))
}

/// Sequence Adam and/or L-BFGS for public JMLE (`adam` / `lbfgs` / `adam_lbfgs`).
#[pyfunction]
#[pyo3(signature = (x0, objective, optimizer, max_iter, learning_rate, tolerance, lbfgs_history))]
fn jmle_optimize<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<'_, f64>,
    objective: Bound<'py, PyAny>,
    optimizer: &str,
    max_iter: usize,
    learning_rate: f64,
    tolerance: f64,
    lbfgs_history: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, String, usize)> {
    let x0_slice = x0.as_slice()?.to_vec();
    let mut obj_cb = |x: &[f64]| -> Result<(f64, Vec<f64>, f64), String> {
        let arr = PyArray1::from_slice(py, x);
        let out = objective
            .call1((arr,))
            .map_err(|e| format!("jmle objective failed: {e}"))?;
        let (obj, grad, loglik): (f64, Vec<f64>, f64) = out
            .extract()
            .map_err(|e| format!("jmle objective must return (float, sequence[float], float): {e}"))?;
        Ok((obj, grad, loglik))
    };
    let (x, obj_t, ll_t, status, n_iter) = core_jmle_run_optimizer(
        &x0_slice,
        &mut obj_cb,
        optimizer,
        max_iter,
        learning_rate,
        tolerance,
        lbfgs_history,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_slice(py, &x),
        PyArray1::from_slice(py, &obj_t),
        PyArray1::from_slice(py, &ll_t),
        status,
        n_iter,
    ))
}

/// Item/test information at supplied (theta, xi) points (Magis 2013 4PL
/// formula, c=0/d=1 logistic case; Lord test-information tradition).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    theta, xi, n_points, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, device = "auto",
))]
fn bank_information(
    py: Python<'_>,
    theta: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray1<'_, f64>,
    n_points: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    device: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let (item_info, test_info) =
        core_bank_information_device(&bank, theta.as_slice()?, xi.as_slice()?, n_points, device)
            .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_info", item_info)?;
    out.set_item("test_info", test_info)?;
    Ok(out.into())
}

/// One adaptive-EAP CAT step. Bock and Mislevy (1982) support EAP scoring;
/// Wang et al. (2010) support multidimensional CAT with information selection.
/// Largest-posterior-SD dimension targeting is a repository policy.
///
/// # References
///
/// Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
/// a microcomputer environment. *Applied Psychological Measurement, 6*(4),
/// 431–444. <https://doi.org/10.1177/014662168200600405>
///
/// Wang, C.-S., Kuo, C.-L., & Chao, C.-Y. (2010). A multidimensional
/// computerized adaptive testing system for enhancing the Chinese as second
/// language proficiency test. In N. E. Mastorakis, V. Mladenov, Z. Bojkovic,
/// & S. Kartalopoulos (Eds.), *Selected topics in education and educational
/// technology* (pp. 245–252). WSEAS Press.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, administered, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0, device = "auto",
))]
fn cat_next_item(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    administered: PyReadonlyArray1<'_, bool>,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    device: &str,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    let step = core_cat_next_item_device(
        &bank,
        y.as_slice()?,
        administered.as_slice()?,
        &prior,
        q_theta,
        rule,
        device,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_eap", step.theta_eap)?;
    out.set_item("theta_sd", step.theta_sd)?;
    out.set_item("xi_eap", step.xi_eap)?;
    out.set_item("target_dim", step.target_dim)?;
    out.set_item("ranked_items", step.ranked_items)?;
    out.set_item("ranked_info", step.ranked_info)?;
    Ok(out.into())
}

/// Posterior plausible values (Marsman et al., 2016), sampled on the
/// repository's fixed-bank quadrature grid without item-parameter uncertainty.
///
/// # References
///
/// Marsman, M., Maris, G., Bechger, T., & Glas, C. (2016). What can we learn
/// from plausible values? *Psychometrika, 81*(2), 274–289.
/// <https://doi.org/10.1007/s11336-016-9497-x>
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0, n_draws = 5, seed = 1, device = "auto",
))]
fn plausible_values(
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    n_draws: usize,
    seed: u64,
    device: &str,
) -> PyResult<Vec<f64>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    core_plausible_values_device(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        q_theta,
        rule,
        n_draws,
        seed,
        device,
    )
    .map_err(PyValueError::new_err)
}

/// Residual item fit (Haberman, Sinharay & Chon 2013).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, theta, xi, n_bins = 10,
))]
fn residual_item_fit(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    theta: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let res = core_residual_item_fit(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        theta.as_slice()?,
        xi.as_slice()?,
        n_bins,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("max_abs_z", res.max_abs_z)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("n_bins", res.n_bins)?;
    Ok(out.into())
}

/// Repository-specific exploratory pairwise adjusted chi2/df ratios.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0,
))]
fn adjusted_chi2_pairs(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let res = core_adjusted_chi2_pairs(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        &prior,
        q_theta,
        rule,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("ratio", res.ratio)?;
    out.set_item("mean_ratio", res.mean_ratio)?;
    out.set_item("max_ratio", res.max_ratio)?;
    Ok(out.into())
}

/// Fixed-estimate Monte Carlo person-fit p-values inspired by Sinharay (2016).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, theta, xi, prior_mean = None, n_replicates = 200, seed = 1,
))]
fn person_fit_resampling(
    y: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, bool>,
    n_persons: usize,
    alpha: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    zeta: PyReadonlyArray1<'_, f64>,
    tau: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    theta: PyReadonlyArray1<'_, f64>,
    xi: PyReadonlyArray1<'_, f64>,
    prior_mean: Option<PyReadonlyArray1<'_, f64>>,
    n_replicates: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    bank_from_args!(
        alpha,
        b,
        zeta,
        tau,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors,
        bank
    );
    let pm = match &prior_mean {
        Some(v) => v.as_slice()?.to_vec(),
        None => Vec::new(),
    };
    core_person_fit_resampling(
        &bank,
        y.as_slice()?,
        observed.as_slice()?,
        n_persons,
        theta.as_slice()?,
        xi.as_slice()?,
        &pm,
        n_replicates,
        seed,
    )
    .map_err(PyValueError::new_err)
}

/// Repository-specific fixed-threshold, backward-elimination TCC drift screen.
///
/// This heuristic is motivated by the TCC-difference objective of Guo et al.
/// (2015), but it does not implement their alternating entry/removal procedure
/// or locally optimal linking-set search.
///
/// # References
///
/// Guo, R., Zheng, Y., & Chang, H. H. (2015). A stepwise test characteristic
/// curve method to detect item parameter drift. *Journal of Educational
/// Measurement, 52*(3), 280–300. https://doi.org/10.1111/jedm.12077
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    alpha_old, b_old, zeta_old, tau_old, alpha_new, b_new, zeta_new, tau_new,
    factor_id, model, n_dims, latent_dim, eps_distance, prior_mean, prior_sd,
    q_theta = 21, xi_rule = "gh", q_xi = 11, xi_points = 256, xi_seed = 0,
    threshold = 0.05,
))]
fn tcc_drift(
    py: Python<'_>,
    alpha_old: PyReadonlyArray1<'_, f64>,
    b_old: PyReadonlyArray1<'_, f64>,
    zeta_old: PyReadonlyArray1<'_, f64>,
    tau_old: f64,
    alpha_new: PyReadonlyArray1<'_, f64>,
    b_new: PyReadonlyArray1<'_, f64>,
    zeta_new: PyReadonlyArray1<'_, f64>,
    tau_new: f64,
    factor_id: PyReadonlyArray1<'_, i64>,
    model: &str,
    n_dims: usize,
    latent_dim: usize,
    eps_distance: f64,
    prior_mean: PyReadonlyArray1<'_, f64>,
    prior_sd: PyReadonlyArray1<'_, f64>,
    q_theta: usize,
    xi_rule: &str,
    q_xi: usize,
    xi_points: usize,
    xi_seed: u64,
    threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(
        alpha_old,
        b_old,
        zeta_old,
        tau_old,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors_old,
        bank_old
    );
    bank_from_args!(
        alpha_new,
        b_new,
        zeta_new,
        tau_new,
        factor_id,
        model,
        n_dims,
        latent_dim,
        eps_distance,
        factors_new,
        bank_new
    );
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let res = core_tcc_drift(&bank_old, &bank_new, &prior, q_theta, rule, threshold)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("drifted", res.drifted)?;
    out.set_item("area_trace", res.area_trace)?;
    out.set_item("iterations", res.iterations)?;
    out.set_item("max_iterations", res.max_iterations)?;
    out.set_item("termination_reason", res.termination_reason)?;
    Ok(out.into())
}

/// Empirical (marginal) EAP reliability per trait dimension from the posterior
/// variance decomposition of Bechger et al. (2003); report it alongside model
/// fit as advised by Stanley and Edwards (2016).
///
/// # References
///
/// Bechger, T. M., Maris, G., Verstralen, H. H. F. M., & Béguin, A. A. (2003).
/// Using classical test theory in combination with item response theory.
/// *Applied Psychological Measurement, 27*(5), 319–334.
/// <https://doi.org/10.1177/0146621603257518>
///
/// Stanley, L. M., & Edwards, M. C. (2016). Reliability and model fit.
/// *Educational and Psychological Measurement, 76*(6), 976–985.
/// <https://doi.org/10.1177/0013164416638900>
#[pyfunction(signature = (theta_eap, theta_sd, n_persons, n_dims, device = "auto"))]
fn empirical_reliability(
    theta_eap: PyReadonlyArray1<'_, f64>,
    theta_sd: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_dims: usize,
    device: &str,
) -> PyResult<Vec<f64>> {
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of ['cpu', 'gpu', 'auto']"))?;
    core_empirical_reliability_device(
        theta_eap.as_slice()?,
        theta_sd.as_slice()?,
        n_persons,
        n_dims,
        device,
    )
    .map_err(PyValueError::new_err)
}


/// Chi-square upper-tail survival P(Chi2_df >= x).
#[pyfunction]
fn chi2_sf(x: f64, df: f64) -> f64 {
    core_chi2_sf(x, df)
}

/// Benjamini–Hochberg FDR rejection mask (NaN p-values skipped).
#[pyfunction]
fn benjamini_hochberg(p_values: PyReadonlyArray1<'_, f64>, q: f64) -> PyResult<Vec<bool>> {
    Ok(core_benjamini_hochberg(p_values.as_slice()?, q))
}

/// Assemble a dense finite-difference Hessian and return a row-major flat matrix.
///
/// Python evaluates the scalar objective at the requested offsets; Rust owns the
/// FD coefficients and symmetrisation so observed-information construction stays
/// single-sourced on the numeric core.
#[pyfunction]
#[pyo3(signature = (
    n,
    step,
    base,
    diag_plus,
    diag_minus,
    off_pp,
    off_pm,
    off_mp,
    off_mm,
))]
fn observed_information(
    n: usize,
    step: f64,
    base: f64,
    diag_plus: PyReadonlyArray1<'_, f64>,
    diag_minus: PyReadonlyArray1<'_, f64>,
    off_pp: PyReadonlyArray1<'_, f64>,
    off_pm: PyReadonlyArray1<'_, f64>,
    off_mp: PyReadonlyArray1<'_, f64>,
    off_mm: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    core_finite_difference_hessian(
        n,
        step,
        base,
        diag_plus.as_slice()?,
        diag_minus.as_slice()?,
        off_pp.as_slice()?,
        off_pm.as_slice()?,
        off_mp.as_slice()?,
        off_mm.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}

/// Positive-definiteness diagnostic for a square Hessian / information matrix.
#[pyfunction]
#[pyo3(signature = (hessian, tol = 1e-8))]
fn second_order_test(
    py: Python<'_>,
    hessian: PyReadonlyArray2<'_, f64>,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let shape = hessian.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("hessian must be a square matrix"));
    }
    let n = shape[0];
    let (passed, min_eigenvalue, eigenvalues) =
        core_second_order_test(hessian.as_slice()?, n, tol).map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("passed", passed)?;
    out.set_item("min_eigenvalue", min_eigenvalue)?;
    out.set_item("eigenvalues", eigenvalues)?;
    Ok(out.into())
}

/// Invert observed information / Hessian (pinv fallback). Returns row-major flat.
#[pyfunction]
fn vcov_from_hessian(hessian: PyReadonlyArray2<'_, f64>, rcond: f64) -> PyResult<Vec<f64>> {
    let shape = hessian.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("hessian must be a square matrix"));
    }
    let n = shape[0];
    core_vcov_from_hessian(hessian.as_slice()?, n, rcond).map_err(PyValueError::new_err)
}

/// Standard errors from a square covariance (negative diag clamped to 0).
#[pyfunction]
fn standard_errors_from_vcov(vcov: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    let shape = vcov.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("vcov must be a square matrix"));
    }
    let n = shape[0];
    core_standard_errors_from_vcov(vcov.as_slice()?, n).map_err(PyValueError::new_err)
}

/// Version of the Python-to-Rust marginal-MMLE call contract.
const MARGINAL_CAPABILITY_VERSION: u32 = 1;

#[pymodule]
#[pyo3(name = "_core")]
fn fast_mlsirm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MARGINAL_CAPABILITY_VERSION", MARGINAL_CAPABILITY_VERSION)?;
    m.add_function(wrap_pyfunction!(neg_loglik_and_grad, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mmle_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(fit_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gdina, m)?)?;
    m.add_function(wrap_pyfunction!(validate_q_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(gdina_wald_selection, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ho_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ho_gdina, m)?)?;
    m.add_function(wrap_pyfunction!(fit_seq_gdina, m)?)?;
    m.add_function(wrap_pyfunction!(fit_seq_gdina_qr, m)?)?;
    m.add_function(wrap_pyfunction!(fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mhrm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nominal_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_grm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_crm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rsm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_facets, m)?)?;
    m.add_function(wrap_pyfunction!(mokken_coef_h, m)?)?;
    m.add_function(wrap_pyfunction!(mokken_aisp, m)?)?;
    m.add_function(wrap_pyfunction!(ksirt_occ, m)?)?;
    m.add_function(wrap_pyfunction!(subscore_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(detect_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(py_dimtest, m)?)?;
    m.add_function(wrap_pyfunction!(py_wollack_omega, m)?)?;
    m.add_function(wrap_pyfunction!(py_k_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_gbt, m)?)?;
    m.add_function(wrap_pyfunction!(py_k_variants, m)?)?;
    m.add_function(wrap_pyfunction!(py_hofstee, m)?)?;
    m.add_function(wrap_pyfunction!(py_person_fit_np, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_plot, m)?)?;
    m.add_function(wrap_pyfunction!(py_eb_mh_dif, m)?)?;
    m.add_function(wrap_pyfunction!(py_mantel_smd_dif, m)?)?;
    m.add_function(wrap_pyfunction!(py_gmh_dif, m)?)?;
    m.add_function(wrap_pyfunction!(py_breslow_day_dif, m)?)?;
    m.add_function(wrap_pyfunction!(rudner_classification, m)?)?;
    m.add_function(wrap_pyfunction!(lee_classification, m)?)?;
    m.add_function(wrap_pyfunction!(livingston_lewis, m)?)?;
    m.add_function(wrap_pyfunction!(hanson_brennan, m)?)?;
    m.add_function(wrap_pyfunction!(hanson_brennan_from_params, m)?)?;
    m.add_function(wrap_pyfunction!(subkoviak_agreement, m)?)?;
    m.add_function(wrap_pyfunction!(livingston_k2, m)?)?;
    m.add_function(wrap_pyfunction!(livingston_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(woodruff_sawyer_sb, m)?)?;
    m.add_function(wrap_pyfunction!(woodruff_sawyer_normal, m)?)?;
    m.add_function(wrap_pyfunction!(gtheory_pi, m)?)?;
    m.add_function(wrap_pyfunction!(phi_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(gtheory_pio, m)?)?;
    m.add_function(wrap_pyfunction!(minres_fa, m)?)?;
    m.add_function(wrap_pyfunction!(minres_fa_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(omega_total_1f, m)?)?;
    m.add_function(wrap_pyfunction!(omega_total_1f_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(glb_fa, m)?)?;
    m.add_function(wrap_pyfunction!(glb_fa_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(velicer_map, m)?)?;
    m.add_function(wrap_pyfunction!(velicer_map_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(selection_utility, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_russell, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(py_sympson_hetter, m)?)?;
    m.add_function(wrap_pyfunction!(py_a_stratified, m)?)?;
    m.add_function(wrap_pyfunction!(py_kl_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_kl_select, m)?)?;
    m.add_function(wrap_pyfunction!(py_owen_update, m)?)?;
    m.add_function(wrap_pyfunction!(py_owen_cat, m)?)?;
    m.add_function(wrap_pyfunction!(py_ccat_select, m)?)?;
    m.add_function(wrap_pyfunction!(py_epv_select, m)?)?;
    m.add_function(wrap_pyfunction!(py_sprt_classify, m)?)?;
    m.add_function(wrap_pyfunction!(py_ci_classify, m)?)?;
    m.add_function(wrap_pyfunction!(py_flexilevel_administer, m)?)?;
    m.add_function(wrap_pyfunction!(py_flexilevel_score_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(py_stradaptive_administer, m)?)?;
    m.add_function(wrap_pyfunction!(py_pyramidal_administer, m)?)?;
    m.add_function(wrap_pyfunction!(py_two_stage_route, m)?)?;
    m.add_function(wrap_pyfunction!(py_two_stage_score, m)?)?;
    m.add_function(wrap_pyfunction!(guttman_lambdas, m)?)?;
    m.add_function(wrap_pyfunction!(tenberge_mu, m)?)?;
    m.add_function(wrap_pyfunction!(cronbach_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(feldt_alpha_ci, m)?)?;
    m.add_function(wrap_pyfunction!(icc, m)?)?;
    m.add_function(wrap_pyfunction!(kripp_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(finn_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(maxwell_re, m)?)?;
    m.add_function(wrap_pyfunction!(robinson_a, m)?)?;
    m.add_function(wrap_pyfunction!(mean_pairwise_cor, m)?)?;
    m.add_function(wrap_pyfunction!(mean_pairwise_rho, m)?)?;
    m.add_function(wrap_pyfunction!(stuart_maxwell_mh, m)?)?;
    m.add_function(wrap_pyfunction!(bhapkar_mh, m)?)?;
    m.add_function(wrap_pyfunction!(rater_bias, m)?)?;
    m.add_function(wrap_pyfunction!(n_cohen_kappa, m)?)?;
    m.add_function(wrap_pyfunction!(separation_reliability, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(fit_lltm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_testlet, m)?)?;
    m.add_function(wrap_pyfunction!(fit_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_eap, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_map, m)?)?;
    m.add_function(wrap_pyfunction!(eapsum_tables, m)?)?;
    m.add_function(wrap_pyfunction!(score_eapsum, m)?)?;
    m.add_function(wrap_pyfunction!(s_x2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(leniency_residuals_stat, m)?)?;
    m.add_function(wrap_pyfunction!(m2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(m2_structured_stat, m)?)?;
    m.add_function(wrap_pyfunction!(m2_cmle_rasch_stat, m)?)?;
    m.add_function(wrap_pyfunction!(factorized_trait_moments_stat, m)?)?;
    m.add_function(wrap_pyfunction!(factorized_multilevel_moments_stat, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_moment_covariance_stat, m)?)?;
    m.add_function(wrap_pyfunction!(projected_m2, m)?)?;
    m.add_function(wrap_pyfunction!(poly_m2, m)?)?;
    m.add_function(wrap_pyfunction!(poly_local_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(poly_dif, m)?)?;
    m.add_function(wrap_pyfunction!(mantel_haenszel_dif, m)?)?;
    m.add_function(wrap_pyfunction!(sibtest, m)?)?;
    m.add_function(wrap_pyfunction!(raju_area, m)?)?;
    m.add_function(wrap_pyfunction!(logistic_dif, m)?)?;
    m.add_function(wrap_pyfunction!(mantel_haenszel_dif_purified, m)?)?;
    m.add_function(wrap_pyfunction!(logistic_dif_purified, m)?)?;
    m.add_function(wrap_pyfunction!(score_wle, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rasch_cml, m)?)?;
    m.add_function(wrap_pyfunction!(andersen_lr_test, m)?)?;
    m.add_function(wrap_pyfunction!(u3_person_fit, m)?)?;
    m.add_function(wrap_pyfunction!(u3_bootstrap_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(irt_link, m)?)?;
    m.add_function(wrap_pyfunction!(link_fixed_item_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(equate_observed_scores, m)?)?;
    m.add_function(wrap_pyfunction!(equate_neat, m)?)?;
    m.add_function(wrap_pyfunction!(equate_neat_linear, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_see, m)?)?;
    m.add_function(wrap_pyfunction!(analytic_see, m)?)?;
    m.add_function(wrap_pyfunction!(equate_observed_scores_ext, m)?)?;
    m.add_function(wrap_pyfunction!(circle_arc_equate, m)?)?;
    m.add_function(wrap_pyfunction!(nominal_weights_mean_equate, m)?)?;
    m.add_function(wrap_pyfunction!(composite_linking, m)?)?;
    m.add_function(wrap_pyfunction!(thurstone_case_v, m)?)?;
    m.add_function(wrap_pyfunction!(bradley_terry_mm, m)?)?;
    m.add_function(wrap_pyfunction!(lsr_pairwise, m)?)?;
    m.add_function(wrap_pyfunction!(ilsr_pairwise, m)?)?;
    m.add_function(wrap_pyfunction!(rank_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(lsr_rankings, m)?)?;
    m.add_function(wrap_pyfunction!(ilsr_rankings, m)?)?;
    m.add_function(wrap_pyfunction!(lsr_top1, m)?)?;
    m.add_function(wrap_pyfunction!(ilsr_top1, m)?)?;
    m.add_function(wrap_pyfunction!(circular_triads, m)?)?;
    m.add_function(wrap_pyfunction!(kendall_u, m)?)?;
    m.add_function(wrap_pyfunction!(elo_rating, m)?)?;
    m.add_function(wrap_pyfunction!(glicko_rating, m)?)?;
    m.add_function(wrap_pyfunction!(glicko2_rating, m)?)?;
    m.add_function(wrap_pyfunction!(stephenson_rating, m)?)?;
    m.add_function(wrap_pyfunction!(elom_rating, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_rating, m)?)?;
    m.add_function(wrap_pyfunction!(fide_rating, m)?)?;
    m.add_function(wrap_pyfunction!(predict_rating_two, m)?)?;
    m.add_function(wrap_pyfunction!(predict_rating_multi, m)?)?;
    m.add_function(wrap_pyfunction!(bratt_mm, m)?)?;
    m.add_function(wrap_pyfunction!(fleiss_kappa, m)?)?;
    m.add_function(wrap_pyfunction!(light_kappa, m)?)?;
    m.add_function(wrap_pyfunction!(circle_arc_middle_anchor, m)?)?;
    m.add_function(wrap_pyfunction!(loglinear_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(person_fit_stat, m)?)?;
    m.add_function(wrap_pyfunction!(infit_outfit_stat, m)?)?;
    m.add_function(wrap_pyfunction!(validate_scoring, m)?)?;
    m.add_function(wrap_pyfunction!(vuong_nonnested, m)?)?;
    m.add_function(wrap_pyfunction!(dimensionality_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(oakes_standard_errors, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_sf, m)?)?;
    m.add_function(wrap_pyfunction!(benjamini_hochberg, m)?)?;
    m.add_function(wrap_pyfunction!(observed_information, m)?)?;
    m.add_function(wrap_pyfunction!(second_order_test, m)?)?;
    m.add_function(wrap_pyfunction!(vcov_from_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(standard_errors_from_vcov, m)?)?;
    m.add_function(wrap_pyfunction!(cat_ability_mle, m)?)?;
    m.add_function(wrap_pyfunction!(cat_ability_eap, m)?)?;
    m.add_function(wrap_pyfunction!(cat_ability_standard_error, m)?)?;
    m.add_function(wrap_pyfunction!(cat_item_information, m)?)?;
    m.add_function(wrap_pyfunction!(cat_select_item, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_test_form_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(jmle_optimize_adam, m)?)?;
    m.add_function(wrap_pyfunction!(jmle_optimize_lbfgs, m)?)?;
    m.add_function(wrap_pyfunction!(jmle_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(bank_information, m)?)?;
    m.add_function(wrap_pyfunction!(cat_next_item, m)?)?;
    m.add_function(wrap_pyfunction!(plausible_values, m)?)?;
    m.add_function(wrap_pyfunction!(residual_item_fit, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_chi2_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(person_fit_resampling, m)?)?;
    m.add_function(wrap_pyfunction!(tcc_drift, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_reliability, m)?)?;
    m.add_function(wrap_pyfunction!(gpcm_cell_logprobs, m)?)?;
    m.add_function(wrap_pyfunction!(grm_cell_logprobs, m)?)?;
    m.add_function(wrap_pyfunction!(polytomous_predictions, m)?)?;
    m.add_function(wrap_pyfunction!(fit_poly_unidim, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nominal, m)?)?;
    m.add_function(wrap_pyfunction!(poly_person_fit, m)?)?;
    m.add_function(wrap_pyfunction!(poly_cat_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(score_poly_eap, m)?)?;
    m.add_function(wrap_pyfunction!(score_wle_poly, m)?)?;
    m.add_function(wrap_pyfunction!(poly_information_curves, m)?)?;
    m.add_function(wrap_pyfunction!(poly_item_fit_sx2, m)?)?;
    m.add_function(wrap_pyfunction!(fit_poly_lsirm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mixed_items, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rt_lognormal, m)?)?;
    m.add_function(wrap_pyfunction!(fit_speed_accuracy_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(rt_person_fit, m)?)?;
    Ok(())
}

fn parse_model_type(model: &str) -> PyResult<ModelType> {
    match model.to_uppercase().as_str() {
        "MIRT" => Ok(ModelType::Mirt),
        "MLS2PLM" => Ok(ModelType::Mls2plm),
        "MLSRM" => Ok(ModelType::Mlsrm),
        "ULS2PLM" => Ok(ModelType::Uls2plm),
        "ULSRM" => Ok(ModelType::Ulsrm),
        "BIFAC2PLM" => Ok(ModelType::Bifac2plm),
        _ => Err(PyValueError::new_err(
            "model must be one of ['MIRT', 'MLS2PLM', 'MLSRM', 'ULS2PLM', 'ULSRM', 'BIFAC2PLM']",
        )),
    }
}

fn convert_factor_id(raw: &[i64], n_dims: usize) -> PyResult<Vec<usize>> {
    raw.iter()
        .map(|&value| {
            if value < 0 || value as usize >= n_dims {
                Err(PyValueError::new_err(
                    "factor_id values must be in 0..n_dims-1",
                ))
            } else {
                Ok(value as usize)
            }
        })
        .collect()
}

fn convert_item_indices(raw: &[i64]) -> PyResult<Vec<usize>> {
    raw.iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("item indices must be non-negative"))
        })
        .collect()
}

fn validate_shapes(
    y: &[usize],
    factor_id: &[usize],
    theta: &[usize],
    alpha: &[usize],
    b: &[usize],
    xi: &[usize],
    zeta: &[usize],
) -> PyResult<()> {
    let n_persons = y[0];
    let n_items = y[1];
    let n_dims = theta[1];
    let latent_dim = xi[1];

    if factor_id != [n_items] {
        return Err(PyValueError::new_err(
            "factor_id length must match number of items",
        ));
    }
    if theta[0] != n_persons {
        return Err(PyValueError::new_err(
            "theta row count must match number of persons",
        ));
    }
    if alpha != [n_items] {
        return Err(PyValueError::new_err(
            "alpha length must match number of items",
        ));
    }
    if b != [n_items] {
        return Err(PyValueError::new_err("b length must match number of items"));
    }
    if xi[0] != n_persons {
        return Err(PyValueError::new_err(
            "xi row count must match number of persons",
        ));
    }
    if zeta != [n_items, latent_dim] {
        return Err(PyValueError::new_err(
            "zeta shape must match number of items and xi latent dimension",
        ));
    }
    if n_dims == 0 || latent_dim == 0 {
        return Err(PyValueError::new_err(
            "parameter dimensions must be positive",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_models() {
        assert!(matches!(parse_model_type("MIRT").unwrap(), ModelType::Mirt));
        assert!(matches!(
            parse_model_type("mls2plm").unwrap(),
            ModelType::Mls2plm
        ));
        assert!(matches!(
            parse_model_type("MLSRM").unwrap(),
            ModelType::Mlsrm
        ));
        assert!(matches!(
            parse_model_type("ULS2PLM").unwrap(),
            ModelType::Uls2plm
        ));
        assert!(matches!(
            parse_model_type("ULSRM").unwrap(),
            ModelType::Ulsrm
        ));
        assert!(parse_model_type("GGUM").is_err());
    }

    #[test]
    fn rejects_invalid_factor_ids() {
        assert_eq!(convert_factor_id(&[0, 1], 2).unwrap(), vec![0, 1]);
        assert!(convert_factor_id(&[-1], 2).is_err());
        assert!(convert_factor_id(&[2], 2).is_err());
    }

    #[test]
    fn validates_wrapper_shapes() {
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[2], &[2], &[2, 2], &[2, 2]).is_ok());
        assert!(validate_shapes(&[2, 2], &[1], &[2, 1], &[2], &[2], &[2, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[1, 1], &[2], &[2], &[2, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[1], &[2], &[2, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[2], &[1], &[2, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[2], &[2], &[1, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[2], &[2], &[2, 2], &[2, 3]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 0], &[2], &[2], &[2, 2], &[2, 2]).is_err());
        assert!(validate_shapes(&[2, 2], &[2], &[2, 1], &[2], &[2], &[2, 0], &[2, 0]).is_err());
    }
}
