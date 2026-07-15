use std::collections::HashMap;

use mlsirm_core::fitstats::{
    infit_outfit as core_infit_outfit, m2_rmsea2 as core_m2, person_fit as core_person_fit,
    poly_local_dependence as core_poly_ld, poly_m2 as core_poly_m2, s_x2 as core_s_x2, SX2Config,
};
use mlsirm_core::agreement::validate_scoring as core_validate_scoring;
use mlsirm_core::marginal::{
    fit_marginal_full as core_fit_marginal_full, Anchors, ItemCovariate, MarginalConfig,
    PopulationSpec, XiRuleKind,
};
use mlsirm_core::nodes::XiRule;
use mlsirm_core::equating::{
    analytic_see as core_analytic_see, bootstrap_see as core_bootstrap_see,
    equate_eg as core_equate_eg, equate_eg_ext as core_equate_eg_ext,
    equate_neat as core_equate_neat, equate_neat_linear as core_equate_neat_linear,
    loglinear_smooth as core_loglinear_smooth, AnchorKind, Continuization, EgSmoothOptions,
    EquateMethod, EquateResult, NeatLinearMethod, NeatMethod, SeeResult,
};
use mlsirm_core::linking::{irt_link as core_irt_link, LinkMethod};

use mlsirm_core::fitstats::{
    adjusted_chi2_pairs as core_adjusted_chi2_pairs,
    person_fit_resampling as core_person_fit_resampling,
    residual_item_fit as core_residual_item_fit, tcc_drift as core_tcc_drift,
};
use mlsirm_core::scoring::{
    bank_information as core_bank_information, cat_next_item as core_cat_next_item,
    empirical_reliability as core_empirical_reliability,
    eapsum_tables as core_eapsum_tables, plausible_values as core_plausible_values,
    score_eap_device as core_score_eap_device, score_map as core_score_map, ItemBank,
    PriorSpec,
};
use mlsirm_core::mmle::{fit_mmle_2pl as core_fit_mmle_2pl, MmleConfig};
use mlsirm_core::cdm::{
    fit_cdm as core_fit_cdm, fit_gdina as core_fit_gdina, fit_ho_cdm as core_fit_ho_cdm,
    fit_ho_gdina as core_fit_ho_gdina, gdina_wald_selection as core_gdina_wald_selection,
    validate_q_matrix as core_validate_q_matrix, CdmConfig, CdmModel,
};
use mlsirm_core::crm::fit_crm as core_fit_crm;
use mlsirm_core::mixture::{fit_mixture as core_fit_mixture, MixtureConfig, MixtureModel};
use mlsirm_core::rsm::fit_rsm as core_fit_rsm;
use mlsirm_core::lltm::{fit_lltm as core_fit_lltm, LltmConfig};
use mlsirm_core::mixed::{fit_mixed_items as core_fit_mixed_items, MixedItemKind, MixedItemSpec};
use mlsirm_core::testlet::{fit_testlet as core_fit_testlet, TestletConfig, TestletModel};
use mlsirm_core::poly::{
    fit_nominal as core_fit_nominal, fit_poly_unidim as core_fit_poly_unidim,
    gpcm_logprobs as core_gpcm_logprobs, grm_logprobs as core_grm_logprobs,
    poly_cat_simulate as core_poly_cat_simulate, poly_dif_sweep as core_poly_dif,
    poly_information_curves as core_poly_information_curves, poly_person_fit as core_poly_person_fit,
    poly_s_x2 as core_poly_s_x2, score_poly_eap as core_score_poly_eap,
    u3_poly_bootstrap_cutoff as core_u3_poly_cutoff, u3_poly_person_fit as core_u3_poly_person_fit,
    PolyModel,
};
use mlsirm_core::poly_marginal::fit_poly_lsirm as core_fit_poly_lsirm;
use mlsirm_core::rt::{fit_rt_lognormal as core_fit_rt, rt_person_fit as core_rt_person_fit, RtConfig};
use mlsirm_core::rt_joint::{
    fit_speed_accuracy_covariance as core_fit_sa, SpeedAccuracyConfig,
};

fn parse_poly_model(model: &str) -> PyResult<PolyModel> {
    match model.to_lowercase().as_str() {
        "grm" => Ok(PolyModel::Grm),
        "gpcm" => Ok(PolyModel::Gpcm),
        other => Err(PyValueError::new_err(format!("model must be grm or gpcm, got {other}"))),
    }
}

fn poly_responses(y: &[i64], n_cat: usize) -> PyResult<Vec<usize>> {
    let mut yv = Vec::with_capacity(y.len());
    for &v in y {
        if v < 0 || v as usize >= n_cat {
            return Err(PyValueError::new_err("responses must be integer categories in 0..n_cat-1"));
        }
        yv.push(v as usize);
    }
    Ok(yv)
}
use mlsirm_core::{
    neg_loglik_and_grad_device as core_neg_loglik_and_grad_device, Device, ModelConfig, ModelType,
    Params, PenaltyConfig,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
    let cfg = MmleConfig { max_iter, tol, ..MmleConfig::default() };
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
        other => return Err(PyValueError::new_err(format!("model must be 'dina' or 'dino'; got {other}"))),
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
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
            _ => Err(PyValueError::new_err("provisional_q entries must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
        other => return Err(PyValueError::new_err(format!("model must be 'dina' or 'dino'; got {other}"))),
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
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
/// `n_iter`, `converged`, `n_parameters`.
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
    let cfg = CdmConfig { max_iter, tol, ..CdmConfig::default() };
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
    out.set_item("n_parameters", res.n_parameters)?;
    Ok(out.into())
}

/// Continuous Response Model fit (Samejima, 1973; `mlsirm_core::crm::fit_crm`).
/// `responses`/`observed` are row-major `n_persons * n_items` with responses in
/// `(0, 1)`. The logit of the response is conditionally normal and linear in the
/// trait, `logit(Z) | theta ~ N(slope*theta + intercept, resid_sd^2)`,
/// `theta ~ N(0,1)`. Returns a dict with `slope`, `intercept`, `resid_sd`,
/// `discrimination` (`= slope/resid_sd`), `difficulty` (`= -intercept/slope`),
/// `theta` (per-person EAP), `loglik_trace`, `n_iter`, `converged`, `n_parameters`.
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
        .map(|&v| if v >= 0 { Ok(v as usize) } else { Err(PyValueError::new_err("y must be non-negative category indices")) })
        .collect::<PyResult<_>>()?;
    let obs = observed.as_slice()?;
    let res = core_fit_rsm(&yy, Some(obs), n_persons, n_items, n_cat, q_theta, max_iter, tol)
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

/// Marginal-EM fit of a mixed Rasch / mixture-IRT model (`mlsirm_core::mixture`, Rost,
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
        other => return Err(PyValueError::new_err(format!("model must be 'rasch' or '2pl'; got {other}"))),
    };
    let cfg = MixtureConfig { max_iter, tol, n_starts, seed, ..MixtureConfig::default() };
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
    let cfg = LltmConfig { max_iter, tol, fit_intercept, compute_lr, ..LltmConfig::default() };
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
        other => return Err(PyValueError::new_err(format!("model must be 'rasch' or '2pl'; got {other}"))),
    };
    let tid: Vec<usize> = testlet_id
        .as_slice()?
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyValueError::new_err("testlet_id entries must be non-negative"))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<PyResult<_>>()?;
    let cfg = TestletConfig { max_iter, tol, q_gamma, estimate_sigma, init_sigma2, ..TestletConfig::default() };
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
            cluster_id: ids
                .ok_or_else(|| PyValueError::new_err("multilevel requires pop_id"))?,
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
    let anchors: Option<Anchors> = match (&anchor_fixed, &anchor_alpha, &anchor_b, &anchor_zeta)
    {
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
        Some(XiRuleKind::Halton) => Ok(XiRule::Halton { n: xi_points, shift_seed: xi_seed }),
        Some(XiRuleKind::MonteCarlo) => {
            Ok(XiRule::MonteCarlo { n: xi_points, seed: xi_seed.max(1) })
        }
        None => Err(PyValueError::new_err("xi_rule must be one of ['gh', 'qmc', 'mc']")),
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
    xi_points = 256, xi_seed = 0, device = "cpu",
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let dev = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err(format!("unknown device: {device}")))?;
    let res = core_score_eap_device(&bank, y.as_slice()?, observed.as_slice()?, n_persons, &prior,
        q_theta, rule, dev)
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
    eps_distance, prior_mean, prior_sd, max_iter = 100, tol = 1e-8,
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let res = core_score_map(&bank, y.as_slice()?, observed.as_slice()?, n_persons, &prior,
        max_iter, tol)
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("theta_map", res.theta_map)?;
    out.set_item("theta_se", res.theta_se)?;
    out.set_item("xi_map", res.xi_map)?;
    out.set_item("log_posterior", res.log_posterior)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

/// Summed-score EAP conversion tables (Lord-Wingersky / Thissen et al. 1995).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim, eps_distance,
    prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11, xi_points = 256,
    xi_seed = 0,
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
) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let tables = core_eapsum_tables(&bank, &prior, q_theta, rule)
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
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
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("rms_residual", res.rms_residual)?;
    out.set_item("flagged_bh", res.flagged_bh)?;
    out.set_item("n_score_groups", res.n_score_groups)?;
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
    let cont = Continuization::parse(continuization)
        .ok_or_else(|| PyValueError::new_err(format!("unknown continuization: {continuization}")))?;
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
    let res = core_bootstrap_see(x_scores.as_slice()?, y_scores.as_slice()?, k_x, k_y, m, n_boot, ci_level, seed)
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
    let res = core_analytic_see(x_scores.as_slice()?, y_scores.as_slice()?, k_x, k_y, m, ci_level)
        .map_err(PyValueError::new_err)?;
    see_result_dict(py, res)
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
    Ok(core_gpcm_logprobs(base, scores.as_slice()?, intercepts.as_slice()?))
}

/// GRM cumulative-logit cell log-probabilities at one node.
#[pyfunction]
#[pyo3(signature = (base, thresholds))]
fn grm_cell_logprobs(base: f64, thresholds: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(core_grm_logprobs(base, thresholds.as_slice()?))
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let fit = core_fit_poly_unidim(&yv, obs, n_persons, n_items, n_cat, m, q_theta, max_iter, tol)
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let cfg = RtConfig { max_iter, tol, var_floor, sigma_floor, fix_sigma_tau };
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
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
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
    let cfg = SpeedAccuracyConfig { q, max_iter, tol, fix_sigma_tau, ..Default::default() };
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let obs = observed.as_ref().map(|o| o.as_slice()).transpose()?;
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
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
#[pyo3(signature = (auto, human, k, human_a = None, human_b = None, subgroup = None))]
fn validate_scoring(
    py: Python<'_>,
    auto: PyReadonlyArray1<'_, u32>,
    human: PyReadonlyArray1<'_, u32>,
    k: usize,
    human_a: Option<PyReadonlyArray1<'_, u32>>,
    human_b: Option<PyReadonlyArray1<'_, u32>>,
    subgroup: Option<PyReadonlyArray1<'_, u32>>,
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
    let verdict = core_validate_scoring(
        auto.as_slice()?,
        human.as_slice()?,
        k,
        hh_storage.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())),
        sg_storage.as_deref(),
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

/// Q3 / GDDM residual dimensionality diagnostics (Svetina & Levy 2014 usable
/// subset).
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
            cluster_id: ids
                .ok_or_else(|| PyValueError::new_err("multilevel requires pop_id"))?,
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


/// Item/test information at supplied (theta, xi) points (Magis 2013 4PL
/// formula, c=0/d=1 logistic case; Lord test-information tradition).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    theta, xi, n_points, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance,
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
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let (item_info, test_info) =
        core_bank_information(&bank, theta.as_slice()?, xi.as_slice()?, n_points)
            .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("item_info", item_info)?;
    out.set_item("test_info", test_info)?;
    Ok(out.into())
}

/// One adaptive-EAP CAT step (Bock & Mislevy 1982; Wang, Kuo & Chao 2010).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, administered, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0,
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
) -> PyResult<Py<pyo3::types::PyDict>> {
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let step = core_cat_next_item(
        &bank, y.as_slice()?, administered.as_slice()?, &prior, q_theta, rule,
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

/// Posterior plausible values (Marsman et al. 2016).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    y, observed, n_persons, alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
    eps_distance, prior_mean, prior_sd, q_theta = 21, xi_rule = "gh", q_xi = 11,
    xi_points = 256, xi_seed = 0, n_draws = 5, seed = 1,
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
) -> PyResult<Vec<f64>> {
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    core_plausible_values(
        &bank, y.as_slice()?, observed.as_slice()?, n_persons, &prior, q_theta, rule,
        n_draws, seed,
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let res = core_residual_item_fit(
        &bank, y.as_slice()?, observed.as_slice()?, n_persons, theta.as_slice()?,
        xi.as_slice()?, n_bins,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("max_abs_z", res.max_abs_z)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("n_bins", res.n_bins)?;
    Ok(out.into())
}

/// Adjusted pairwise chi2/df ratios (Tay & Drasgow 2012).
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let prior = PriorSpec {
        mean: prior_mean.as_slice()?.to_vec(),
        sd: prior_sd.as_slice()?.to_vec(),
    };
    let rule = parse_xi_rule(xi_rule, q_xi, xi_points, xi_seed)?;
    let res = core_adjusted_chi2_pairs(
        &bank, y.as_slice()?, observed.as_slice()?, n_persons, &prior, q_theta, rule,
    )
    .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("ratio", res.ratio)?;
    out.set_item("mean_ratio", res.mean_ratio)?;
    out.set_item("max_ratio", res.max_ratio)?;
    Ok(out.into())
}

/// Parametric-bootstrap person-fit p-values (Sinharay 2016).
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
    bank_from_args!(alpha, b, zeta, tau, factor_id, model, n_dims, latent_dim,
        eps_distance, factors, bank);
    let pm = match &prior_mean {
        Some(v) => v.as_slice()?.to_vec(),
        None => Vec::new(),
    };
    core_person_fit_resampling(
        &bank, y.as_slice()?, observed.as_slice()?, n_persons, theta.as_slice()?,
        xi.as_slice()?, &pm, n_replicates, seed,
    )
    .map_err(PyValueError::new_err)
}

/// Stepwise TCC drift detection between two calibrations (Guo et al. 2015).
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
    bank_from_args!(alpha_old, b_old, zeta_old, tau_old, factor_id, model, n_dims,
        latent_dim, eps_distance, factors_old, bank_old);
    bank_from_args!(alpha_new, b_new, zeta_new, tau_new, factor_id, model, n_dims,
        latent_dim, eps_distance, factors_new, bank_new);
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
    Ok(out.into())
}


/// Empirical (marginal) EAP reliability per trait dimension
/// (Stanley & Edwards 2016; Milanzi et al. 2015).
#[pyfunction]
fn empirical_reliability(
    theta_eap: PyReadonlyArray1<'_, f64>,
    theta_sd: PyReadonlyArray1<'_, f64>,
    n_persons: usize,
    n_dims: usize,
) -> PyResult<Vec<f64>> {
    core_empirical_reliability(theta_eap.as_slice()?, theta_sd.as_slice()?, n_persons, n_dims)
        .map_err(PyValueError::new_err)
}

#[pymodule]
#[pyo3(name = "_core")]
fn fast_mlsirm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg_loglik_and_grad, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mmle_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(fit_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gdina, m)?)?;
    m.add_function(wrap_pyfunction!(validate_q_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(gdina_wald_selection, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ho_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ho_gdina, m)?)?;
    m.add_function(wrap_pyfunction!(fit_crm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rsm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(fit_lltm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_testlet, m)?)?;
    m.add_function(wrap_pyfunction!(fit_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_eap, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_map, m)?)?;
    m.add_function(wrap_pyfunction!(eapsum_tables, m)?)?;
    m.add_function(wrap_pyfunction!(s_x2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(m2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(poly_m2, m)?)?;
    m.add_function(wrap_pyfunction!(poly_local_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(poly_dif, m)?)?;
    m.add_function(wrap_pyfunction!(u3_person_fit, m)?)?;
    m.add_function(wrap_pyfunction!(u3_bootstrap_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(irt_link, m)?)?;
    m.add_function(wrap_pyfunction!(equate_observed_scores, m)?)?;
    m.add_function(wrap_pyfunction!(equate_neat, m)?)?;
    m.add_function(wrap_pyfunction!(equate_neat_linear, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_see, m)?)?;
    m.add_function(wrap_pyfunction!(analytic_see, m)?)?;
    m.add_function(wrap_pyfunction!(equate_observed_scores_ext, m)?)?;
    m.add_function(wrap_pyfunction!(loglinear_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(person_fit_stat, m)?)?;
    m.add_function(wrap_pyfunction!(infit_outfit_stat, m)?)?;
    m.add_function(wrap_pyfunction!(validate_scoring, m)?)?;
    m.add_function(wrap_pyfunction!(vuong_nonnested, m)?)?;
    m.add_function(wrap_pyfunction!(dimensionality_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(oakes_standard_errors, m)?)?;
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
    m.add_function(wrap_pyfunction!(fit_poly_unidim, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nominal, m)?)?;
    m.add_function(wrap_pyfunction!(poly_person_fit, m)?)?;
    m.add_function(wrap_pyfunction!(poly_cat_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(score_poly_eap, m)?)?;
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
