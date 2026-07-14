use std::collections::HashMap;

use mlsirm_core::marginal::{
    fit_marginal as core_fit_marginal, MarginalConfig, PopulationSpec,
};
use mlsirm_core::mmle::{fit_mmle_2pl as core_fit_mmle_2pl, MmleConfig};
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
    let mcfg = MarginalConfig {
        q_theta,
        q_xi,
        q_u,
        max_iter,
        tol,
        m_steps,
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
    let res = core_fit_marginal(
        y.as_slice()?,
        observed.as_slice()?,
        &factors,
        &config,
        &pop,
        &mcfg,
        &penalty,
        device,
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
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out.into())
}

#[pymodule]
#[pyo3(name = "_core")]
fn fast_mlsirm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg_loglik_and_grad, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mmle_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(fit_marginal, m)?)?;
    Ok(())
}

fn parse_model_type(model: &str) -> PyResult<ModelType> {
    match model.to_uppercase().as_str() {
        "MIRT" => Ok(ModelType::Mirt),
        "MLS2PLM" => Ok(ModelType::Mls2plm),
        "MLSRM" => Ok(ModelType::Mlsrm),
        "ULS2PLM" => Ok(ModelType::Uls2plm),
        "ULSRM" => Ok(ModelType::Ulsrm),
        _ => Err(PyValueError::new_err(
            "model must be one of ['MIRT', 'MLS2PLM', 'MLSRM', 'ULS2PLM', 'ULSRM']",
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
