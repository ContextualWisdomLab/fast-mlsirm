use std::collections::HashMap;

use mlsirm_core::fitstats::{
    infit_outfit as core_infit_outfit, m2_rmsea2 as core_m2, person_fit as core_person_fit,
    s_x2 as core_s_x2, SX2Config,
};
use mlsirm_core::agreement::validate_scoring as core_validate_scoring;
use mlsirm_core::marginal::{
    fit_marginal_full as core_fit_marginal_full, Anchors, ItemCovariate, MarginalConfig,
    PopulationSpec, XiRuleKind,
};
use mlsirm_core::nodes::XiRule;
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
use mlsirm_core::poly::{
    fit_poly_unidim as core_fit_poly_unidim, gpcm_logprobs as core_gpcm_logprobs,
    grm_logprobs as core_grm_logprobs, poly_information_curves as core_poly_information_curves,
    score_poly_eap as core_score_poly_eap, PolyModel,
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
#[pyo3(signature = (y, n_persons, n_items, n_cat, model = "grm", q_theta = 21, max_iter = 80, tol = 1e-6))]
fn fit_poly_unidim(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    model: &str,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let fit = core_fit_poly_unidim(&yv, n_persons, n_items, n_cat, m, q_theta, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("slope", fit.slope)?;
    out.set_item("cat_params", fit.cat_params)?;
    out.set_item("loglik", fit.loglik)?;
    out.set_item("n_iter", fit.n_iter)?;
    Ok(out.into())
}

/// EAP trait scores from polytomous responses given fitted item parameters
/// (Rust compute path). Returns a dict with `theta_eap` and `theta_sd`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (y, n_persons, n_items, n_cat, slope, cat_params, model = "grm", q_theta = 21))]
fn score_poly_eap(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, i64>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: PyReadonlyArray1<'_, f64>,
    cat_params: PyReadonlyArray1<'_, f64>,
    model: &str,
    q_theta: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let m = parse_poly_model(model)?;
    let yv = poly_responses(y.as_slice()?, n_cat)?;
    let (eap, sd) = core_score_poly_eap(
        &yv,
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
    out.set_item("n_moments", res.n_moments)?;
    out.set_item("n_parameters", res.n_parameters)?;
    out.set_item("n_complete", res.n_complete)?;
    Ok(out.into())
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
    m.add_function(wrap_pyfunction!(fit_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_eap, m)?)?;
    m.add_function(wrap_pyfunction!(score_bank_map, m)?)?;
    m.add_function(wrap_pyfunction!(eapsum_tables, m)?)?;
    m.add_function(wrap_pyfunction!(s_x2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(m2_stat, m)?)?;
    m.add_function(wrap_pyfunction!(irt_link, m)?)?;
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
    m.add_function(wrap_pyfunction!(score_poly_eap, m)?)?;
    m.add_function(wrap_pyfunction!(poly_information_curves, m)?)?;
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
