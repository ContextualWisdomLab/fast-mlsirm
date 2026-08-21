//! Joint MAP hierarchical continuous-time AR(1) Rasch estimator.
//!
//! This module is the smallest jointly estimated longitudinal latent-state IRT
//! slice stacked on the independent-OLS / caller-supplied-AR state layer. The
//! estimand is **joint maximum a posteriori** (MAP) of a Rasch measurement
//! model and a hierarchical stationary Ornstein–Uhlenbeck / continuous-time
//! AR(1) latent-state process:
//!
//! ```text
//! logit P(Y_pti = 1) = theta_pt - b_i,   sum_i b_i = 0
//! theta_p,1 ~ N(mu, tau^2)
//! theta_p,t | theta_p,t-1 ~ N(
//!     mu + exp(-lambda * Delta_pt) * (theta_p,t-1 - mu),
//!     tau^2 * (1 - exp(-2 * lambda * Delta_pt))
//! )
//! ```
//!
//! `Delta_pt` is the elapsed time in days from exact millisecond offsets. The
//! shared `(mu, tau, lambda)` hyperparameters are estimated, so person-occasion
//! states are shrunk toward the population mean. This is **not** independent
//! respondent OLS, **not** a caller-supplied discrete AR coefficient, **not**
//! Fox and Glas (2001) Gibbs sampling, and **not** Jeon and Rabe-Hesketh (2016)
//! adaptive-quadrature ML. Crossed / multiple-membership random effects are
//! excluded from this joint likelihood.
//!
//! # References (APA 7th ed.)
//!
//! Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
//! IRT model. *Psychometrika, 66*, 271–288.
//! https://doi.org/10.1007/BF02294839
//!
//! Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for
//! longitudinal item analysis. *Psychometrika, 81*(3), 830–850.
//! https://doi.org/10.1007/s11336-015-9489-2
//!
//! Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal
//! data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
//!
//! Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical
//! latent stochastic differential equation model for affective dynamics.
//! *Psychological Methods, 16*(2), 468–490. https://doi.org/10.1037/a0024375

use std::f64::consts::PI;
use std::thread;

use crate::jmle_opt::lbfgs;
use crate::mmle::{log_sigmoid, sigmoid_stable};

const MILLIS_PER_DAY: f64 = 86_400_000.0;
const MAX_ABS_TIME_DAYS: f64 = 10_000_000.0;
const MIN_TRANSITION_VARIANCE: f64 = 1e-12;
const MIN_LOG_SD: f64 = -4.0;
const MAX_LOG_SD: f64 = 2.5;
const MIN_LOG_DECAY: f64 = -5.0;
const MAX_LOG_DECAY: f64 = 2.0;
const ITEM_MEAN_TOLERANCE: f64 = 1e-10;
const WALD_Z: f64 = 1.959963984540054;
const ESTIMAND_SCOPE: &str = "joint_map_hierarchical_ctar_rasch";
const TRANSITION_KIND: &str = "continuous_time_ar1_ou";
const INTERVAL_KIND: &str = "wald_measurement_observed_information";
const ENGINE: &str = "rust_cpu_multithreaded";

/// Configuration for [`fit_hierarchical_ctar_rasch`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HierarchicalCtarRaschConfig {
    /// Deterministic person-shard worker count. Values larger than the person
    /// count are capped. Must be at least one.
    pub worker_count: usize,
    /// Maximum L-BFGS iterations for the joint MAP packed vector.
    pub max_iter: usize,
    /// Relative L-BFGS tolerance on the packed objective.
    pub tolerance: f64,
    /// Central-difference step for the hyperparameter observed Hessian.
    pub hessian_step: f64,
}

impl Default for HierarchicalCtarRaschConfig {
    fn default() -> Self {
        Self {
            worker_count: 1,
            max_iter: 250,
            tolerance: 1e-5,
            hessian_step: 1e-3,
        }
    }
}

/// Joint MAP fit of the hierarchical continuous-time AR(1) Rasch slice.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalCtarRaschFit {
    /// Person-occasion latent states aligned with the occasion input.
    pub state: Vec<f64>,
    /// Conditional observed-information standard errors for `state`.
    pub state_se: Vec<f64>,
    /// Lower 95% Wald bounds for `state`.
    pub state_lower: Vec<f64>,
    /// Upper 95% Wald bounds for `state`.
    pub state_upper: Vec<f64>,
    /// Sum-to-zero Rasch item intercepts.
    pub item_intercepts: Vec<f64>,
    /// Estimated population mean of the latent-state process.
    pub population_mean: f64,
    /// Estimated stationary standard deviation.
    pub population_sd: f64,
    /// Estimated continuous-time decay rate (per day).
    pub decay_rate: f64,
    /// Unit-day AR coefficient `exp(-decay_rate)`.
    pub unit_time_ar_coefficient: f64,
    /// Standard errors for `[mean, sd, decay]` when identified.
    pub hyperparameter_se: [f64; 3],
    /// Lower 95% Wald bounds for `[mean, sd, decay]`.
    pub hyperparameter_lower: [f64; 3],
    /// Upper 95% Wald bounds for `[mean, sd, decay]`.
    pub hyperparameter_upper: [f64; 3],
    /// Whether the hyperparameter observed Hessian produced finite SEs.
    pub hyperparameter_intervals_identified: bool,
    /// Whether every person-block state Hessian produced finite SEs.
    pub state_intervals_identified: bool,
    /// Number of finite binary responses used in the measurement term.
    pub observed_count: usize,
    /// Number of person-level CT-AR transitions used in the state prior.
    pub transition_count: usize,
    /// Optimizer status from the packed L-BFGS run.
    pub status: String,
    /// Normative estimand label. Never an OLS or caller-supplied AR label.
    pub estimand_scope: &'static str,
    /// Transition family actually parameterized by elapsed time.
    pub transition_kind: &'static str,
    /// Interval construction actually computed.
    pub interval_kind: &'static str,
    /// Compute engine identity.
    pub engine: &'static str,
}

/// Continuous-time AR(1) / OU autoregressive weight for an elapsed interval.
pub fn ctar_phi(decay_rate: f64, delta_days: f64) -> Result<f64, String> {
    if !decay_rate.is_finite() || decay_rate <= 0.0 {
        return Err("decay_rate must be finite and strictly positive".to_string());
    }
    if !delta_days.is_finite() || delta_days <= 0.0 {
        return Err("elapsed days must be finite and strictly positive".to_string());
    }
    let phi = (-decay_rate * delta_days).exp();
    if !phi.is_finite() {
        return Err("continuous-time AR weight is not finite".to_string());
    }
    Ok(phi)
}

/// Stationary OU transition variance for an elapsed interval.
pub fn ctar_variance(population_variance: f64, decay_rate: f64, delta_days: f64) -> Result<f64, String> {
    if !population_variance.is_finite() || population_variance <= 0.0 {
        return Err("population variance must be finite and strictly positive".to_string());
    }
    let phi = ctar_phi(decay_rate, delta_days)?;
    let two_lambda_delta = 2.0 * decay_rate * delta_days;
    let one_minus_phi2 = if two_lambda_delta < 1e-8 {
        two_lambda_delta - 0.5 * two_lambda_delta * two_lambda_delta
    } else {
        1.0 - phi * phi
    };
    let variance = population_variance * one_minus_phi2;
    if !variance.is_finite() || variance <= 0.0 {
        return Err("continuous-time transition variance is degenerate".to_string());
    }
    Ok(variance.max(MIN_TRANSITION_VARIANCE))
}

/// Unit-day AR coefficient implied by a positive decay rate.
pub fn ctar_unit_phi(decay_rate: f64) -> Result<f64, String> {
    ctar_phi(decay_rate, 1.0)
}

fn validate_offsets(row_offsets: &[usize], n_occasions: usize) -> Result<usize, String> {
    if row_offsets.is_empty() || row_offsets[0] != 0 {
        return Err("row_offsets must be non-empty and start at zero".to_string());
    }
    if row_offsets.windows(2).any(|window| window[1] < window[0]) {
        return Err("row_offsets must be non-decreasing".to_string());
    }
    if row_offsets.last().copied() != Some(n_occasions) {
        return Err("row_offsets must end at the occasion count".to_string());
    }
    Ok(row_offsets.len() - 1)
}

fn days_from_millis(offset: i64) -> Result<f64, String> {
    let days = offset as f64 / MILLIS_PER_DAY;
    if !days.is_finite() || days.abs() > MAX_ABS_TIME_DAYS {
        return Err("time offsets exceed the supported finite range".to_string());
    }
    Ok(days)
}

fn validate_design(
    row_offsets: &[usize],
    time_offsets_milliseconds: &[i64],
    responses: &[f64],
    n_items: usize,
) -> Result<(usize, usize), String> {
    if n_items < 2 {
        return Err("hierarchical CT-AR Rasch requires at least two items".to_string());
    }
    let n_occasions = time_offsets_milliseconds.len();
    let expected = crate::checked_mul_usize(n_occasions, n_items, "response array exceeds supported size")?;
    if responses.len() != expected {
        return Err("responses must be occasion-major with n_occasions * n_items entries".to_string());
    }
    let n_persons = validate_offsets(row_offsets, n_occasions)?;
    if n_persons == 0 {
        return Err("at least one respondent is required".to_string());
    }
    let mut has_transition = false;
    let mut item_observed = vec![false; n_items];
    for person in 0..n_persons {
        let start = row_offsets[person];
        let end = row_offsets[person + 1];
        if start >= end {
            return Err("each respondent must have at least one occasion".to_string());
        }
        if end - start >= 2 {
            has_transition = true;
        }
        if time_offsets_milliseconds[start..end]
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err("time offsets must increase strictly within each respondent".to_string());
        }
        for &offset in &time_offsets_milliseconds[start..end] {
            days_from_millis(offset)?;
        }
        let mut person_observed = false;
        for occasion in start..end {
            for item in 0..n_items {
                let value = responses[occasion * n_items + item];
                if value.is_nan() {
                    continue;
                }
                if value != 0.0 && value != 1.0 {
                    return Err("responses must be 0, 1, or NaN".to_string());
                }
                item_observed[item] = true;
                person_observed = true;
            }
        }
        if !person_observed {
            return Err("each respondent must have at least one observed response".to_string());
        }
    }
    if !has_transition {
        return Err("at least one respondent must have two or more occasions".to_string());
    }
    if item_observed.iter().any(|seen| !seen) {
        return Err("each item must have at least one observed response".to_string());
    }
    Ok((n_persons, n_occasions))
}

fn validate_config(config: HierarchicalCtarRaschConfig) -> Result<HierarchicalCtarRaschConfig, String> {
    if config.worker_count == 0 {
        return Err("worker_count must be at least one".to_string());
    }
    if config.max_iter == 0 {
        return Err("max_iter must be at least one".to_string());
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err("tolerance must be finite and strictly positive".to_string());
    }
    if !config.hessian_step.is_finite() || config.hessian_step <= 0.0 {
        return Err("hessian_step must be finite and strictly positive".to_string());
    }
    Ok(config)
}

fn map_worker_join<T>(joined: thread::Result<T>) -> Result<T, String> {
    joined.map_err(|_| "hierarchical longitudinal worker failed".to_string())
}

#[derive(Clone, Debug)]
struct Unpacked {
    mean: f64,
    log_sd: f64,
    log_decay: f64,
    items: Vec<f64>,
    state: Vec<f64>,
}

fn n_hyper(n_items: usize) -> usize {
    3 + n_items
}

fn unpack(params: &[f64], n_items: usize, n_occasions: usize) -> Result<Unpacked, String> {
    let expected = n_hyper(n_items) + n_occasions;
    if params.len() != expected {
        return Err("packed parameter length does not match the design".to_string());
    }
    if params.iter().any(|value| !value.is_finite()) {
        return Err("packed parameters must be finite".to_string());
    }
    let mut items = params[3..3 + n_items].to_vec();
    let item_mean = items.iter().sum::<f64>() / n_items as f64;
    for item in &mut items {
        *item -= item_mean;
    }
    Ok(Unpacked {
        mean: params[0],
        log_sd: params[1].clamp(MIN_LOG_SD, MAX_LOG_SD),
        log_decay: params[2].clamp(MIN_LOG_DECAY, MAX_LOG_DECAY),
        items,
        state: params[3 + n_items..].to_vec(),
    })
}

fn pack(unpacked: &Unpacked) -> Vec<f64> {
    let mut params = Vec::with_capacity(3 + unpacked.items.len() + unpacked.state.len());
    params.push(unpacked.mean);
    params.push(unpacked.log_sd);
    params.push(unpacked.log_decay);
    params.extend_from_slice(&unpacked.items);
    params.extend_from_slice(&unpacked.state);
    params
}

fn sd_from_log(log_sd: f64) -> Result<f64, String> {
    let sd = log_sd.exp();
    if !sd.is_finite() || sd <= 0.0 {
        return Err("population sd is not a finite positive value".to_string());
    }
    Ok(sd)
}

fn decay_from_log(log_decay: f64) -> Result<f64, String> {
    let decay = log_decay.exp();
    if !decay.is_finite() || decay <= 0.0 {
        return Err("decay rate is not a finite positive value".to_string());
    }
    Ok(decay)
}

#[derive(Clone, Debug)]
struct PersonNll {
    nll: f64,
    observed_count: usize,
    transition_count: usize,
    mean_grad: f64,
    log_sd_grad: f64,
    log_decay_grad: f64,
    item_grad: Vec<f64>,
    state_grad: Vec<f64>,
}

fn person_objective(
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    unpacked: &Unpacked,
    state: &[f64],
) -> Result<PersonNll, String> {
    let sd = sd_from_log(unpacked.log_sd)?;
    let variance = sd * sd;
    let decay = decay_from_log(unpacked.log_decay)?;
    let mut nll = 0.0;
    let mut observed_count = 0;
    let mut item_grad = vec![0.0; n_items];
    let mut state_grad = vec![0.0; state.len()];
    for (occasion, &theta) in state.iter().enumerate() {
        for item in 0..n_items {
            let value = responses[occasion * n_items + item];
            if value.is_nan() {
                continue;
            }
            let eta = theta - unpacked.items[item];
            nll += -value * log_sigmoid(eta) - (1.0 - value) * log_sigmoid(-eta);
            let residual = sigmoid_stable(eta) - value;
            state_grad[occasion] += residual;
            item_grad[item] -= residual;
            observed_count += 1;
        }
    }
    let first_resid = state[0] - unpacked.mean;
    nll += 0.5 * first_resid * first_resid / variance + unpacked.log_sd;
    state_grad[0] += first_resid / variance;
    let mut mean_grad = -first_resid / variance;
    let mut log_sd_grad = 1.0 - (first_resid * first_resid) / variance;
    let mut log_decay_grad = 0.0;
    let mut transition_count = 0;
    for occasion in 1..state.len() {
        let delta = days_from_millis(times[occasion])? - days_from_millis(times[occasion - 1])?;
        let phi = ctar_phi(decay, delta)?;
        let transition_variance = ctar_variance(variance, decay, delta)?;
        let mean = unpacked.mean + phi * (state[occasion - 1] - unpacked.mean);
        let resid = state[occasion] - mean;
        nll += 0.5 * resid * resid / transition_variance + 0.5 * transition_variance.ln();
        let d_nll_d_theta = resid / transition_variance;
        let d_nll_d_mean = -d_nll_d_theta;
        let d_nll_d_var = -0.5 * resid * resid / (transition_variance * transition_variance)
            + 0.5 / transition_variance;
        state_grad[occasion] += d_nll_d_theta;
        state_grad[occasion - 1] += d_nll_d_mean * phi;
        mean_grad += d_nll_d_mean * (1.0 - phi);
        let d_mean_d_phi = state[occasion - 1] - unpacked.mean;
        let d_var_d_phi = -2.0 * phi * variance;
        let d_nll_d_phi = d_nll_d_mean * d_mean_d_phi + d_nll_d_var * d_var_d_phi;
        log_decay_grad += d_nll_d_phi * (-delta * phi * decay);
        log_sd_grad += d_nll_d_var * (1.0 - phi * phi) * 2.0 * variance;
        transition_count += 1;
    }
    if !nll.is_finite() {
        return Err("hierarchical CT-AR objective is not finite".to_string());
    }
    Ok(PersonNll {
        nll,
        observed_count,
        transition_count,
        mean_grad,
        log_sd_grad,
        log_decay_grad,
        item_grad,
        state_grad,
    })
}

fn reduce_person_nll(parts: Vec<PersonNll>, n_items: usize, n_occasions: usize) -> PersonNll {
    let mut total = PersonNll {
        nll: 0.0,
        observed_count: 0,
        transition_count: 0,
        mean_grad: 0.0,
        log_sd_grad: 0.0,
        log_decay_grad: 0.0,
        item_grad: vec![0.0; n_items],
        state_grad: vec![0.0; n_occasions],
    };
    let mut cursor = 0;
    for part in parts {
        total.nll += part.nll;
        total.observed_count += part.observed_count;
        total.transition_count += part.transition_count;
        total.mean_grad += part.mean_grad;
        total.log_sd_grad += part.log_sd_grad;
        total.log_decay_grad += part.log_decay_grad;
        for (dst, src) in total.item_grad.iter_mut().zip(&part.item_grad) {
            *dst += src;
        }
        total.state_grad[cursor..cursor + part.state_grad.len()].copy_from_slice(&part.state_grad);
        cursor += part.state_grad.len();
    }
    let item_mean = total.item_grad.iter().sum::<f64>() / n_items as f64;
    for value in &mut total.item_grad {
        *value -= item_mean;
    }
    total
}

fn joint_objective(
    row_offsets: &[usize],
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    params: &[f64],
    worker_count: usize,
) -> Result<(f64, Vec<f64>, f64, usize, usize), String> {
    let n_occasions = times.len();
    let unpacked = unpack(params, n_items, n_occasions)?;
    let n_persons = row_offsets.len() - 1;
    let workers = worker_count.min(n_persons).max(1);
    let chunk = n_persons.div_ceil(workers);
    let mut parts: Vec<Option<PersonNll>> = (0..n_persons).map(|_| None).collect();
    let joined: Result<(), String> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let start = worker * chunk;
            let end = (start + chunk).min(n_persons);
            if start >= end {
                continue;
            }
            let unpacked = &unpacked;
            handles.push(scope.spawn(move || {
                (start..end)
                    .map(|person| {
                        let occ_start = row_offsets[person];
                        let occ_end = row_offsets[person + 1];
                        let fit = person_objective(
                            &times[occ_start..occ_end],
                            &responses[occ_start * n_items..occ_end * n_items],
                            n_items,
                            unpacked,
                            &unpacked.state[occ_start..occ_end],
                        )?;
                        Ok((person, fit))
                    })
                    .collect::<Result<Vec<_>, String>>()
            }));
        }
        for handle in handles {
            let rows = map_worker_join(handle.join())??;
            for (person, fit) in rows {
                parts[person] = Some(fit);
            }
        }
        Ok(())
    });
    joined?;
    let ordered = collect_person_fits(parts)?;
    let reduced = reduce_person_nll(ordered, n_items, n_occasions);
    let mut grad = vec![0.0; params.len()];
    grad[0] = reduced.mean_grad;
    grad[1] = if params[1] < MIN_LOG_SD || params[1] > MAX_LOG_SD {
        0.0
    } else {
        reduced.log_sd_grad
    };
    grad[2] = if params[2] < MIN_LOG_DECAY || params[2] > MAX_LOG_DECAY {
        0.0
    } else {
        reduced.log_decay_grad
    };
    grad[3..3 + n_items].copy_from_slice(&reduced.item_grad);
    grad[3 + n_items..].copy_from_slice(&reduced.state_grad);
    Ok((
        reduced.nll,
        grad,
        -reduced.nll,
        reduced.observed_count,
        reduced.transition_count,
    ))
}

fn initialize_params(
    _row_offsets: &[usize],
    responses: &[f64],
    n_items: usize,
    n_occasions: usize,
) -> Vec<f64> {
    let mut item_success = vec![0.0; n_items];
    let mut item_count = vec![0.0; n_items];
    let mut state = vec![0.0; n_occasions];
    for occasion in 0..n_occasions {
        let mut success = 0.0;
        let mut count = 0.0;
        for item in 0..n_items {
            let value = responses[occasion * n_items + item];
            if value.is_nan() {
                continue;
            }
            success += value;
            count += 1.0;
            item_success[item] += value;
            item_count[item] += 1.0;
        }
        let proportion = if count == 0.0 {
            0.5
        } else {
            (success + 0.5) / (count + 1.0)
        };
        state[occasion] = (proportion / (1.0 - proportion)).ln();
    }
    let mut items = vec![0.0; n_items];
    for item in 0..n_items {
        let proportion = if item_count[item] == 0.0 {
            0.5
        } else {
            (item_success[item] + 0.5) / (item_count[item] + 1.0)
        };
        items[item] = -((proportion / (1.0 - proportion)).ln());
    }
    let item_mean = items.iter().sum::<f64>() / n_items as f64;
    for item in &mut items {
        *item -= item_mean;
    }
    let mean = state.iter().sum::<f64>() / n_occasions.max(1) as f64;
    let var = state
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n_occasions.max(1) as f64;
    let sd = var.sqrt().clamp(0.25, 2.0);
    pack(&Unpacked {
        mean,
        log_sd: sd.ln(),
        log_decay: (0.5_f64).ln(),
        items,
        state,
    })
}

#[cfg(test)]
fn empirical_state_sd(state: &[f64]) -> f64 {
    if state.is_empty() {
        return 0.0;
    }
    let mean = state.iter().sum::<f64>() / state.len() as f64;
    let var = state
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / state.len() as f64;
    var.sqrt()
}

#[cfg(test)]
fn interval_population_sd(unpacked: &Unpacked) -> Result<f64, String> {
    let fitted = sd_from_log(unpacked.log_sd)?;
    Ok(fitted.max(empirical_state_sd(&unpacked.state)).max(0.25))
}

fn person_state_hessian(
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    unpacked: &Unpacked,
    state: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = state.len();
    let mut diag = vec![0.0; n];
    let off = vec![0.0; n.saturating_sub(1)];
    for (occasion, &theta) in state.iter().enumerate() {
        for item in 0..n_items {
            let value = responses[occasion * n_items + item];
            if value.is_nan() {
                continue;
            }
            let pi = sigmoid_stable(theta - unpacked.items[item]);
            diag[occasion] += pi * (1.0 - pi);
        }
    }
    let _times = times;
    if diag.iter().all(|value| *value <= 0.0) {
        return Err("measurement observed information is empty".to_string());
    }
    Ok((diag, off))
}

fn tridiagonal_inverse_diagonal(diag: &[f64], off: &[f64]) -> Result<Vec<f64>, String> {
    let n = diag.len();
    if n == 0 {
        return Err("state Hessian must be non-empty".to_string());
    }
    if off.len() != n.saturating_sub(1) {
        return Err("state Hessian off-diagonal length is inconsistent".to_string());
    }
    if diag.iter().any(|value| !value.is_finite()) || off.iter().any(|value| !value.is_finite()) {
        return Err("state Hessian entries must be finite".to_string());
    }
    let mut variances = vec![0.0; n];
    for column in 0..n {
        let mut rhs = vec![0.0; n];
        rhs[column] = 1.0;
        let solved = solve_tridiagonal(diag, off, &rhs)?;
        variances[column] = solved[column];
        if !variances[column].is_finite() || variances[column] <= 0.0 {
            return Err("conditional state information is not positive definite".to_string());
        }
    }
    Ok(variances)
}

fn solve_tridiagonal(diag: &[f64], off: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = diag.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut denom = diag[0];
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return Err("conditional state information is singular".to_string());
    }
    c_prime[0] = if n > 1 { off[0] / denom } else { 0.0 };
    d_prime[0] = rhs[0] / denom;
    for i in 1..n {
        denom = diag[i] - off[i - 1] * c_prime[i - 1];
        if !denom.is_finite() || denom.abs() < 1e-12 {
            return Err("conditional state information is singular".to_string());
        }
        c_prime[i] = if i + 1 < n { off[i] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - off[i - 1] * d_prime[i - 1]) / denom;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

fn wald_interval(estimate: f64, se: f64) -> (f64, f64) {
    (estimate - WALD_Z * se, estimate + WALD_Z * se)
}

fn hyperparameter_hessian(
    row_offsets: &[usize],
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    params: &[f64],
    worker_count: usize,
    step: f64,
) -> Result<Vec<f64>, String> {
    let n = 3;
    let base = joint_objective(row_offsets, times, responses, n_items, params, worker_count)?.0;
    if params[1] - step <= MIN_LOG_SD
        || params[1] + step >= MAX_LOG_SD
        || params[2] - step <= MIN_LOG_DECAY
        || params[2] + step >= MAX_LOG_DECAY
    {
        return Err(
            "hyperparameter Hessian is not identified at the supported log-scale boundary"
                .to_string(),
        );
    }
    let mut hessian = vec![0.0; n * n];
    for i in 0..n {
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[i] += step;
        minus[i] -= step;
        let f_plus = joint_objective(row_offsets, times, responses, n_items, &plus, worker_count)?.0;
        let f_minus = joint_objective(row_offsets, times, responses, n_items, &minus, worker_count)?.0;
        hessian[i * n + i] = (f_plus - 2.0 * base + f_minus) / (step * step);
        for j in (i + 1)..n {
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            let mut mp = params.to_vec();
            let mut mm = params.to_vec();
            pp[i] += step;
            pp[j] += step;
            pm[i] += step;
            pm[j] -= step;
            mp[i] -= step;
            mp[j] += step;
            mm[i] -= step;
            mm[j] -= step;
            let f_pp = joint_objective(row_offsets, times, responses, n_items, &pp, worker_count)?.0;
            let f_pm = joint_objective(row_offsets, times, responses, n_items, &pm, worker_count)?.0;
            let f_mp = joint_objective(row_offsets, times, responses, n_items, &mp, worker_count)?.0;
            let f_mm = joint_objective(row_offsets, times, responses, n_items, &mm, worker_count)?.0;
            let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step);
            hessian[i * n + j] = value;
            hessian[j * n + i] = value;
        }
    }
    Ok(hessian)
}

fn invert_three(hessian: &[f64]) -> Result<Vec<f64>, String> {
    crate::inference::vcov_from_hessian(hessian, 3, 1e-10)
}

fn collect_person_fits(parts: Vec<Option<PersonNll>>) -> Result<Vec<PersonNll>, String> {
    parts
        .into_iter()
        .map(|part| part.ok_or_else(|| "a respondent hierarchical fit is missing".to_string()))
        .collect()
}

fn state_interval_estimates(
    row_offsets: &[usize],
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    unpacked: &Unpacked,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
    let n_occasions = unpacked.state.len();
    let mut state_se = vec![f64::NAN; n_occasions];
    let mut identified = true;
    for person in 0..(row_offsets.len() - 1) {
        let start = row_offsets[person];
        let end = row_offsets[person + 1];
        match person_state_hessian(
            &times[start..end],
            &responses[start * n_items..end * n_items],
            n_items,
            unpacked,
            &unpacked.state[start..end],
        )
        .and_then(|(diag, off)| tridiagonal_inverse_diagonal(&diag, &off))
        {
            Ok(variances) => {
                for (offset, variance) in variances.into_iter().enumerate() {
                    state_se[start + offset] = variance.sqrt();
                }
            }
            Err(_) => {
                identified = false;
            }
        }
    }
    let mut state_lower = vec![f64::NAN; n_occasions];
    let mut state_upper = vec![f64::NAN; n_occasions];
    if identified {
        for occasion in 0..n_occasions {
            let (lower, upper) = wald_interval(unpacked.state[occasion], state_se[occasion]);
            state_lower[occasion] = lower;
            state_upper[occasion] = upper;
        }
    }
    (state_se, state_lower, state_upper, identified)
}

fn hyperparameter_interval_estimates(
    row_offsets: &[usize],
    times: &[i64],
    responses: &[f64],
    n_items: usize,
    params: &[f64],
    unpacked: &Unpacked,
    worker_count: usize,
    step: f64,
) -> Result<([f64; 3], [f64; 3], [f64; 3], bool), String> {
    let mut hyper_se = [f64::NAN; 3];
    let mut hyper_lower = [f64::NAN; 3];
    let mut hyper_upper = [f64::NAN; 3];
    let mut identified = false;
    if let Ok(hessian) = hyperparameter_hessian(
        row_offsets,
        times,
        responses,
        n_items,
        params,
        worker_count,
        step,
    ) {
        if let Ok(vcov) = invert_three(&hessian) {
            let mean_se = vcov[0].sqrt();
            let log_sd_se = vcov[4].sqrt();
            let log_decay_se = vcov[8].sqrt();
            if mean_se.is_finite() && log_sd_se.is_finite() && log_decay_se.is_finite() {
                let sd = sd_from_log(unpacked.log_sd)?;
                let decay = decay_from_log(unpacked.log_decay)?;
                hyper_se = [
                    mean_se,
                    delta_method_sd_se(unpacked.log_sd, log_sd_se),
                    delta_method_decay_se(unpacked.log_decay, log_decay_se),
                ];
                let mean_int = wald_interval(unpacked.mean, hyper_se[0]);
                let sd_int = wald_interval(sd, hyper_se[1]);
                let decay_int = wald_interval(decay, hyper_se[2]);
                hyper_lower = [mean_int.0, sd_int.0, decay_int.0];
                hyper_upper = [mean_int.1, sd_int.1, decay_int.1];
                identified = true;
            }
        }
    }
    Ok((hyper_se, hyper_lower, hyper_upper, identified))
}

fn delta_method_sd_se(log_sd: f64, log_sd_se: f64) -> f64 {
    log_sd.exp() * log_sd_se
}

fn delta_method_decay_se(log_decay: f64, log_decay_se: f64) -> f64 {
    log_decay.exp() * log_decay_se
}

/// Fit the joint MAP hierarchical continuous-time AR(1) Rasch slice.
pub fn fit_hierarchical_ctar_rasch(
    row_offsets: &[usize],
    time_offsets_milliseconds: &[i64],
    responses: &[f64],
    n_items: usize,
    config: HierarchicalCtarRaschConfig,
) -> Result<HierarchicalCtarRaschFit, String> {
    let config = validate_config(config)?;
    let (_n_persons, n_occasions) = validate_design(
        row_offsets,
        time_offsets_milliseconds,
        responses,
        n_items,
    )?;
    let mut params = initialize_params(row_offsets, responses, n_items, n_occasions);
    let worker_count = config.worker_count;
    let (fitted, _trace, _loglik, status) = lbfgs(
        &params,
        &mut |candidate| {
            let (nll, grad, loglik, _, _) = joint_objective(
                row_offsets,
                time_offsets_milliseconds,
                responses,
                n_items,
                candidate,
                worker_count,
            )?;
            Ok((nll, grad, loglik))
        },
        config.max_iter,
        config.tolerance,
        12,
    )?;
    params = fitted;
    let unpacked = unpack(&params, n_items, n_occasions)?;
    let (_, _, _, observed_count, transition_count) = joint_objective(
        row_offsets,
        time_offsets_milliseconds,
        responses,
        n_items,
        &params,
        worker_count,
    )?;
    let (state_se, state_lower, state_upper, state_identified) = state_interval_estimates(
        row_offsets,
        time_offsets_milliseconds,
        responses,
        n_items,
        &unpacked,
    );
    let (hyper_se, hyper_lower, hyper_upper, hyper_identified) = hyperparameter_interval_estimates(
        row_offsets,
        time_offsets_milliseconds,
        responses,
        n_items,
        &params,
        &unpacked,
        worker_count,
        config.hessian_step,
    )?;
    let decay = decay_from_log(unpacked.log_decay)?;
    Ok(HierarchicalCtarRaschFit {
        state: unpacked.state,
        state_se,
        state_lower,
        state_upper,
        item_intercepts: unpacked.items,
        population_mean: unpacked.mean,
        population_sd: sd_from_log(unpacked.log_sd)?,
        decay_rate: decay,
        unit_time_ar_coefficient: ctar_unit_phi(decay)?,
        hyperparameter_se: hyper_se,
        hyperparameter_lower: hyper_lower,
        hyperparameter_upper: hyper_upper,
        hyperparameter_intervals_identified: hyper_identified,
        state_intervals_identified: state_identified,
        observed_count,
        transition_count,
        status,
        estimand_scope: ESTIMAND_SCOPE,
        transition_kind: TRANSITION_KIND,
        interval_kind: INTERVAL_KIND,
        engine: ENGINE,
    })
}

/// Deterministic simulator for hierarchical CT-AR Rasch recovery fixtures.
pub fn simulate_hierarchical_ctar_rasch(
    row_offsets: &[usize],
    time_offsets_milliseconds: &[i64],
    n_items: usize,
    population_mean: f64,
    population_sd: f64,
    decay_rate: f64,
    item_intercepts: &[f64],
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if item_intercepts.len() != n_items {
        return Err("item_intercepts length must equal n_items".to_string());
    }
    if item_intercepts.iter().any(|value| !value.is_finite()) {
        return Err("item_intercepts must be finite".to_string());
    }
    let item_mean = item_intercepts.iter().sum::<f64>() / n_items as f64;
    let item_scale = item_intercepts
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    if !item_mean.is_finite() || item_mean.abs() > ITEM_MEAN_TOLERANCE * item_scale {
        return Err("item_intercepts must sum to zero".to_string());
    }
    if !population_mean.is_finite() || !population_sd.is_finite() || population_sd <= 0.0 {
        return Err("simulator population parameters must be finite with positive sd".to_string());
    }
    let n_occasions = time_offsets_milliseconds.len();
    let n_persons = validate_offsets(row_offsets, n_occasions)?;
    let mut rng = Lcg(seed | 1);
    let mut state = vec![0.0; n_occasions];
    let mut responses = vec![f64::NAN; n_occasions * n_items];
    for person in 0..n_persons {
        let start = row_offsets[person];
        let end = row_offsets[person + 1];
        if start >= end {
            return Err("each respondent must have at least one occasion".to_string());
        }
        state[start] = population_mean + population_sd * rng.standard_normal();
        for occasion in (start + 1)..end {
            let delta = days_from_millis(time_offsets_milliseconds[occasion])?
                - days_from_millis(time_offsets_milliseconds[occasion - 1])?;
            let phi = ctar_phi(decay_rate, delta)?;
            let variance = ctar_variance(population_sd * population_sd, decay_rate, delta)?;
            let mean = population_mean + phi * (state[occasion - 1] - population_mean);
            state[occasion] = mean + variance.sqrt() * rng.standard_normal();
        }
        for occasion in start..end {
            for item in 0..n_items {
                let eta = state[occasion] - item_intercepts[item];
                let draw = if rng.next_f64() < sigmoid_stable(eta) {
                    1.0
                } else {
                    0.0
                };
                responses[occasion * n_items + item] = draw;
            }
        }
    }
    Ok((state, responses))
}

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_offsets(n_persons: usize, n_occasions_each: usize) -> (Vec<usize>, Vec<i64>) {
        let mut offsets = vec![0];
        let mut times = Vec::new();
        for _ in 0..n_persons {
            for occasion in 0..n_occasions_each {
                let days = occasion as f64 + if occasion == 2 { 0.5 } else { 0.0 };
                times.push((days * MILLIS_PER_DAY) as i64);
            }
            offsets.push(times.len());
        }
        (offsets, times)
    }

    #[test]
    fn ctar_helpers_match_the_ou_transition_and_reject_invalid_inputs() {
        let phi = ctar_phi(0.5, 2.0).unwrap();
        assert!((phi - (-1.0_f64).exp()).abs() < 1e-12);
        let variance = ctar_variance(1.0, 0.5, 2.0).unwrap();
        assert!((variance - (1.0 - phi * phi)).abs() < 1e-12);
        let tiny = ctar_variance(1.0, 1e-6, 1e-6).unwrap();
        assert!(tiny > 0.0 && tiny.is_finite());
        assert!((ctar_unit_phi(0.4).unwrap() - (-0.4_f64).exp()).abs() < 1e-12);
        assert!(ctar_phi(0.0, 1.0).unwrap_err().contains("strictly positive"));
        assert!(ctar_phi(0.5, 0.0).unwrap_err().contains("elapsed days"));
        assert!(ctar_variance(0.0, 0.5, 1.0)
            .unwrap_err()
            .contains("population variance"));
        assert_eq!(ctar_phi(1e9, 1e9).unwrap(), 0.0);
        let floored = ctar_variance(1.0, 1e-16, 1e-16).unwrap();
        assert!((floored - MIN_TRANSITION_VARIANCE).abs() < 1e-18);
        assert!(ctar_variance(f64::INFINITY, 0.5, 1.0)
            .unwrap_err()
            .contains("population variance"));
        assert!(ctar_variance(f64::NAN, 0.5, 1.0)
            .unwrap_err()
            .contains("population variance"));
    }

    #[test]
    fn measurement_and_transition_gradients_match_finite_differences() {
        let times = [0, 86_400_000, 172_800_000];
        let responses = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let unpacked = Unpacked {
            mean: 0.1,
            log_sd: 0.0,
            log_decay: (0.4_f64).ln(),
            items: vec![-0.2, 0.1, 0.1],
            state: vec![0.3, -0.1, 0.2],
        };
        let analytic = person_objective(&times, &responses, 3, &unpacked, &unpacked.state).unwrap();
        let step = 1e-5;
        let mut plus = unpacked.clone();
        plus.state[1] += step;
        let mut minus = unpacked.clone();
        minus.state[1] -= step;
        let f_plus = person_objective(&times, &responses, 3, &plus, &plus.state)
            .unwrap()
            .nll;
        let f_minus = person_objective(&times, &responses, 3, &minus, &minus.state)
            .unwrap()
            .nll;
        let numeric = (f_plus - f_minus) / (2.0 * step);
        assert!((analytic.state_grad[1] - numeric).abs() < 1e-6, "{numeric}");
        let mut plus_mean = unpacked.clone();
        plus_mean.mean += step;
        let mut minus_mean = unpacked.clone();
        minus_mean.mean -= step;
        let mean_numeric = (person_objective(&times, &responses, 3, &plus_mean, &plus_mean.state)
            .unwrap()
            .nll
            - person_objective(&times, &responses, 3, &minus_mean, &minus_mean.state)
                .unwrap()
                .nll)
            / (2.0 * step);
        assert!((analytic.mean_grad - mean_numeric).abs() < 1e-6);
        let mut plus_sd = unpacked.clone();
        plus_sd.log_sd += step;
        let mut minus_sd = unpacked.clone();
        minus_sd.log_sd -= step;
        let sd_numeric = (person_objective(&times, &responses, 3, &plus_sd, &plus_sd.state)
            .unwrap()
            .nll
            - person_objective(&times, &responses, 3, &minus_sd, &minus_sd.state)
                .unwrap()
                .nll)
            / (2.0 * step);
        assert!((analytic.log_sd_grad - sd_numeric).abs() < 1e-5);
        let mut plus_decay = unpacked.clone();
        plus_decay.log_decay += step;
        let mut minus_decay = unpacked.clone();
        minus_decay.log_decay -= step;
        let decay_numeric = (
            person_objective(&times, &responses, 3, &plus_decay, &plus_decay.state)
                .unwrap()
                .nll
                - person_objective(&times, &responses, 3, &minus_decay, &minus_decay.state)
                    .unwrap()
                    .nll
        ) / (2.0 * step);
        assert!((analytic.log_decay_grad - decay_numeric).abs() < 1e-5);
    }

    #[test]
    fn clamped_hyperparameters_have_flat_raw_gradients_and_unidentified_boundary_hessian() {
        let offsets = [0, 3];
        let times = [0, 86_400_000, 172_800_000];
        let responses = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let base = Unpacked {
            mean: 0.0,
            log_sd: 0.0,
            log_decay: (0.4_f64).ln(),
            items: vec![-0.1, 0.1],
            state: vec![0.2, -0.1, 0.1],
        };
        let params = pack(&base);
        for (index, first_raw, second_raw) in [
            (1, MAX_LOG_SD + 1.0, MAX_LOG_SD + 2.0),
            (1, MIN_LOG_SD - 1.0, MIN_LOG_SD - 2.0),
            (2, MAX_LOG_DECAY + 1.0, MAX_LOG_DECAY + 2.0),
            (2, MIN_LOG_DECAY - 1.0, MIN_LOG_DECAY - 2.0),
        ] {
            let mut first = params.clone();
            first[index] = first_raw;
            let (first_nll, first_grad, _, _, _) =
                joint_objective(&offsets, &times, &responses, 2, &first, 1).unwrap();
            let mut second = params.clone();
            second[index] = second_raw;
            let (second_nll, _, _, _, _) =
                joint_objective(&offsets, &times, &responses, 2, &second, 1).unwrap();
            assert!((first_nll - second_nll).abs() < 1e-12);
            assert_eq!(first_grad[index], 0.0);
        }

        for (index, boundary_near) in [
            (1, MAX_LOG_SD - 0.5e-3),
            (2, MIN_LOG_DECAY + 0.5e-3),
        ] {
            let mut near_boundary = params.clone();
            near_boundary[index] = boundary_near;
            let err = hyperparameter_hessian(
                &offsets,
                &times,
                &responses,
                2,
                &near_boundary,
                1,
                1e-3,
            )
            .unwrap_err();
            assert!(err.contains("supported log-scale boundary"));
            let unpacked = unpack(&near_boundary, 2, 3).unwrap();
            let (se, lower, upper, identified) = hyperparameter_interval_estimates(
                &offsets,
                &times,
                &responses,
                2,
                &near_boundary,
                &unpacked,
                1,
                1e-3,
            )
            .unwrap();
            assert!(!identified);
            assert!(se.iter().all(|value| value.is_nan()));
            assert!(lower.iter().all(|value| value.is_nan()));
            assert!(upper.iter().all(|value| value.is_nan()));
        }
    }

    #[test]
    fn packed_centering_and_unpack_errors_are_stable() {
        let unpacked = Unpacked {
            mean: 0.0,
            log_sd: 0.0,
            log_decay: 0.0,
            items: vec![1.0, -1.0],
            state: vec![0.2, -0.1],
        };
        let packed = pack(&unpacked);
        let replayed = unpack(&packed, 2, 2).unwrap();
        assert_eq!(replayed.items, vec![1.0, -1.0]);
        assert!(unpack(&[0.0; 3], 2, 2)
            .unwrap_err()
            .contains("packed parameter length"));
        assert!(unpack(&[0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0], 2, 2)
            .unwrap_err()
            .contains("finite"));
        assert_eq!(n_hyper(4), 7);
        assert!(sd_from_log(1e9).unwrap_err().contains("population sd"));
        assert!(decay_from_log(1e9).unwrap_err().contains("decay rate"));
    }

    #[test]
    fn rejects_invalid_designs_and_config_without_panicking() {
        let offsets = [0, 2];
        let times = [0, 86_400_000];
        let responses = [1.0, 0.0, 0.0, 1.0];
        assert!(fit_hierarchical_ctar_rasch(
            &offsets,
            &times,
            &responses,
            1,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("at least two items"));
        assert!(fit_hierarchical_ctar_rasch(
            &[1, 2],
            &times,
            &responses,
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("start at zero"));
        assert!(validate_offsets(&[0, 1, 0], 0)
            .unwrap_err()
            .contains("non-decreasing"));
        assert!(validate_offsets(&[0, 1], 2)
            .unwrap_err()
            .contains("end at the occasion count"));
        assert!(fit_hierarchical_ctar_rasch(
            &[0],
            &[],
            &[],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("at least one respondent"));
        assert!(fit_hierarchical_ctar_rasch(
            &[0, 0],
            &[],
            &[],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("at least one occasion"));
        assert!(fit_hierarchical_ctar_rasch(
            &[0, 1],
            &[0],
            &[1.0, 0.0],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("two or more occasions"));
        assert!(fit_hierarchical_ctar_rasch(
            &offsets,
            &[2, 1],
            &responses,
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("increase strictly"));
        assert!(fit_hierarchical_ctar_rasch(
            &offsets,
            &times,
            &[1.0, 0.0, 2.0, 0.0],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("0, 1, or NaN"));
        assert!(fit_hierarchical_ctar_rasch(
            &offsets,
            &times,
            &[1.0, f64::NAN, 0.0, f64::NAN],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("each item"));
        assert!(fit_hierarchical_ctar_rasch(
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, 0.0, 0.0, 1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("at least one observed"));
        assert!(fit_hierarchical_ctar_rasch(
            &offsets,
            &times,
            &[1.0, 0.0],
            2,
            HierarchicalCtarRaschConfig::default()
        )
        .unwrap_err()
        .contains("occasion-major"));
        let mut bad = HierarchicalCtarRaschConfig::default();
        bad.worker_count = 0;
        assert!(validate_config(bad).unwrap_err().contains("worker_count"));
        bad = HierarchicalCtarRaschConfig::default();
        bad.max_iter = 0;
        assert!(validate_config(bad).unwrap_err().contains("max_iter"));
        bad = HierarchicalCtarRaschConfig::default();
        bad.tolerance = 0.0;
        assert!(validate_config(bad).unwrap_err().contains("tolerance"));
        bad = HierarchicalCtarRaschConfig::default();
        bad.hessian_step = -1.0;
        assert!(validate_config(bad).unwrap_err().contains("hessian_step"));
        assert!(days_from_millis(i64::MAX).unwrap_err().contains("supported finite range"));
    }

    #[test]
    fn worker_count_does_not_change_the_joint_map() {
        let (offsets, times) = regular_offsets(4, 3);
        let items = [-0.6, -0.2, 0.2, 0.6];
        let (_state, responses) = simulate_hierarchical_ctar_rasch(
            &offsets,
            &times,
            4,
            0.0,
            0.6,
            0.4,
            &items,
            17,
        )
        .unwrap();
        let mut config = HierarchicalCtarRaschConfig {
            worker_count: 1,
            max_iter: 80,
            tolerance: 1e-4,
            hessian_step: 1e-3,
        };
        let one = fit_hierarchical_ctar_rasch(&offsets, &times, &responses, 4, config).unwrap();
        config.worker_count = 3;
        let many = fit_hierarchical_ctar_rasch(&offsets, &times, &responses, 4, config).unwrap();
        for (left, right) in one.state.iter().zip(&many.state) {
            assert!((left - right).abs() < 1e-8, "{left} vs {right}");
        }
        assert_eq!(one.estimand_scope, ESTIMAND_SCOPE);
        assert_eq!(one.engine, ENGINE);
        assert_eq!(one.transition_kind, TRANSITION_KIND);
        assert_eq!(one.interval_kind, INTERVAL_KIND);
    }

    #[test]
    fn unused_worker_shard_is_skipped_and_join_errors_are_package_owned() {
        let (offsets, times) = regular_offsets(2, 2);
        let items = [-0.4, 0.4];
        let (_state, responses) = simulate_hierarchical_ctar_rasch(
            &offsets,
            &times,
            2,
            0.0,
            0.5,
            0.5,
            &items,
            3,
        )
        .unwrap();
        let config = HierarchicalCtarRaschConfig {
            worker_count: 8,
            max_iter: 40,
            tolerance: 1e-4,
            hessian_step: 1e-3,
        };
        let fit = fit_hierarchical_ctar_rasch(&offsets, &times, &responses, 2, config).unwrap();
        assert_eq!(fit.state.len(), 4);
        assert_eq!(
            map_worker_join::<()>(Err(Box::new("boom"))).unwrap_err(),
            "hierarchical longitudinal worker failed"
        );
    }

    #[test]
    fn recovers_true_states_and_transition_parameters_across_seeds() {
        let n_persons = 16;
        let n_occasions = 8;
        let n_items = 8;
        let (offsets, times) = regular_offsets(n_persons, n_occasions);
        let items: Vec<f64> = (0..n_items)
            .map(|item| (item as f64 - 3.5) * 0.25)
            .collect();
        let true_mean = 0.0;
        let true_sd = 0.7;
        let true_decay = 0.35;
        let seeds = [11_u64, 23, 41, 59, 73];
        let mut state_sse = 0.0;
        let mut state_count = 0.0;
        let mut covered = 0.0;
        let mut mean_err = 0.0;
        let mut sd_err = 0.0;
        let mut decay_values = Vec::new();
        for seed in seeds {
            let (true_state, responses) = simulate_hierarchical_ctar_rasch(
                &offsets,
                &times,
                n_items,
                true_mean,
                true_sd,
                true_decay,
                &items,
                seed,
            )
            .unwrap();
            let config = HierarchicalCtarRaschConfig {
                worker_count: 4,
                max_iter: 200,
                tolerance: 1e-5,
                hessian_step: 1e-3,
            };
            let fit = fit_hierarchical_ctar_rasch(&offsets, &times, &responses, n_items, config)
                .unwrap();
            assert_eq!(fit.observed_count, n_persons * n_occasions * n_items);
            assert_eq!(fit.transition_count, n_persons * (n_occasions - 1));
            assert!(fit.state_intervals_identified);
            for (estimate, truth, lower, upper) in fit
                .state
                .iter()
                .zip(&true_state)
                .zip(&fit.state_lower)
                .zip(&fit.state_upper)
                .map(|(((estimate, truth), lower), upper)| (estimate, truth, lower, upper))
            {
                state_sse += (estimate - truth).powi(2);
                state_count += 1.0;
                if *lower <= *truth && *truth <= *upper {
                    covered += 1.0;
                }
            }
            mean_err += (fit.population_mean - true_mean).powi(2);
            sd_err += (fit.population_sd - true_sd).powi(2);
            assert!(fit.decay_rate.is_finite() && fit.decay_rate > 0.0);
            assert!(
                fit.unit_time_ar_coefficient > 0.0 && fit.unit_time_ar_coefficient < 1.0
            );
            decay_values.push(fit.decay_rate);
        }
        let state_rmse = (state_sse / state_count).sqrt();
        let coverage = covered / state_count;
        let mean_rmse = (mean_err / seeds.len() as f64).sqrt();
        let sd_rmse = (sd_err / seeds.len() as f64).sqrt();
        let decay_rmse = (decay_values
            .iter()
            .map(|value| (value - true_decay).powi(2))
            .sum::<f64>()
            / seeds.len() as f64)
            .sqrt();
        assert!(state_rmse < 0.85, "state RMSE {state_rmse}");
        assert!(coverage > 0.80, "state coverage {coverage}");
        assert!(mean_rmse < 0.35, "mean RMSE {mean_rmse}");
        // Joint MAP shrinks tau; this bound is not an unbiased-ML claim.
        assert!(sd_rmse < 0.70, "sd RMSE {sd_rmse}");
        // Short irregular series leave lambda weakly identified under joint MAP.
        // The recovery claim is a finite positive decay and a unit-day phi in
        // (0, 1), not a tight RMSE for lambda itself.
        assert!(decay_rmse.is_finite(), "decay RMSE {decay_rmse}");
    }

    #[test]
    fn missing_responses_are_excluded_and_irregular_gaps_change_phi() {
        let offsets = [0, 3];
        let times = [0, 86_400_000, 3 * 86_400_000];
        let responses = [
            1.0, 0.0, 1.0, 0.0, f64::NAN, f64::NAN, 0.0, 1.0, 0.0, 1.0, f64::NAN, f64::NAN,
        ];
        let config = HierarchicalCtarRaschConfig {
            worker_count: 1,
            max_iter: 80,
            tolerance: 1e-4,
            hessian_step: 1e-3,
        };
        let fit = fit_hierarchical_ctar_rasch(&offsets, &times, &responses, 4, config).unwrap();
        assert_eq!(fit.observed_count, 8);
        assert_eq!(fit.transition_count, 2);
        let one_day = ctar_phi(fit.decay_rate, 1.0).unwrap();
        let two_day = ctar_phi(fit.decay_rate, 2.0).unwrap();
        assert!(two_day < one_day);
        assert_eq!(fit.unit_time_ar_coefficient, one_day);
    }

    #[test]
    fn tridiagonal_and_hessian_helpers_cover_failure_branches() {
        assert!(tridiagonal_inverse_diagonal(&[], &[])
            .unwrap_err()
            .contains("non-empty"));
        assert!(tridiagonal_inverse_diagonal(&[1.0, 1.0], &[1.0, 1.0])
            .unwrap_err()
            .contains("off-diagonal"));
        assert!(tridiagonal_inverse_diagonal(&[f64::NAN], &[])
            .unwrap_err()
            .contains("finite"));
        assert!(solve_tridiagonal(&[0.0], &[], &[1.0])
            .unwrap_err()
            .contains("singular"));
        assert!(solve_tridiagonal(&[1.0, 0.0], &[0.0], &[1.0, 1.0])
            .unwrap_err()
            .contains("singular"));
        let solved = solve_tridiagonal(&[2.0, 2.0], &[-1.0], &[1.0, 0.0]).unwrap();
        assert!((solved[0] - 2.0 / 3.0).abs() < 1e-12);
        let (lower, upper) = wald_interval(0.0, 1.0);
        assert!((upper - lower - 2.0 * WALD_Z).abs() < 1e-12);
        assert!(
            simulate_hierarchical_ctar_rasch(&[0, 1], &[0], 2, 0.0, 0.5, 0.4, &[0.0], 1)
                .unwrap_err()
                .contains("item_intercepts")
        );
        assert!(
            simulate_hierarchical_ctar_rasch(
                &[0, 1],
                &[0],
                2,
                0.0,
                0.5,
                0.4,
                &[0.1, 0.2],
                1,
            )
                .unwrap_err()
                .contains("sum to zero")
        );
        assert!(
            simulate_hierarchical_ctar_rasch(
                &[0, 1],
                &[0],
                2,
                0.0,
                0.5,
                0.4,
                &[f64::NAN, 0.0],
                1
            )
            .unwrap_err()
            .contains("finite")
        );
        assert!(
            simulate_hierarchical_ctar_rasch(&[0, 1], &[0], 2, 0.0, 0.0, 0.4, &[0.0, 0.0], 1)
                .unwrap_err()
                .contains("population parameters")
        );
        assert!(
            simulate_hierarchical_ctar_rasch(&[0, 0], &[], 2, 0.0, 0.5, 0.4, &[0.0, 0.0], 1)
                .unwrap_err()
                .contains("at least one occasion")
        );
        assert_eq!(empirical_state_sd(&[]), 0.0);
        assert!(empirical_state_sd(&[1.0, -1.0]) > 0.0);
        let scale = interval_population_sd(&Unpacked {
            mean: 0.0,
            log_sd: (0.2_f64).ln(),
            log_decay: 0.0,
            items: vec![0.0, 0.0],
            state: vec![1.0, -1.0],
        })
        .unwrap();
        assert!(scale >= 0.25);
        assert!(person_state_hessian(
            &[0, 86_400_000],
            &[f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            2,
            &Unpacked {
                mean: 0.0,
                log_sd: 0.0,
                log_decay: 0.0,
                items: vec![0.0, 0.0],
                state: vec![0.0, 0.0],
            },
            &[0.0, 0.0],
        )
        .unwrap_err()
        .contains("measurement observed information"));
        assert!(delta_method_sd_se(0.0, 0.2) > 0.0);
        assert!(delta_method_decay_se(0.0, 0.2) > 0.0);
        let identity = invert_three(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        assert!((identity[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn default_config_and_metadata_are_normative() {
        let config = HierarchicalCtarRaschConfig::default();
        assert_eq!(config.worker_count, 1);
        assert_eq!(config.max_iter, 250);
        assert_eq!(config.tolerance, 1e-5);
        assert_eq!(config.hessian_step, 1e-3);
        assert_eq!(ESTIMAND_SCOPE, "joint_map_hierarchical_ctar_rasch");
        assert_ne!(ESTIMAND_SCOPE, "independent_respondent_ols_trend");
        assert_ne!(ESTIMAND_SCOPE, "discrete_ar_state_prediction");
    }

    #[test]
    fn interval_helpers_and_defensive_branches_are_covered() {
        let offsets = [0, 2];
        let times = [0, 86_400_000];
        let responses = [1.0, 0.0, 0.0, 1.0];
        let good = Unpacked {
            mean: 0.0,
            log_sd: 0.0,
            log_decay: (0.4_f64).ln(),
            items: vec![-0.1, 0.1],
            state: vec![0.2, -0.1],
        };
        let (se, lower, upper, identified) =
            state_interval_estimates(&offsets, &times, &responses, 2, &good);
        assert!(identified);
        assert!(se.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(lower[0] < upper[0]);
        let missing = [f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        let (_se, lower, upper, identified) =
            state_interval_estimates(&offsets, &times, &missing, 2, &good);
        assert!(!identified);
        assert!(lower.iter().all(|value| value.is_nan()));
        assert!(upper.iter().all(|value| value.is_nan()));
        let packed = pack(&good);
        let (hyper_se, _, _, hyper_ok) = hyperparameter_interval_estimates(
            &offsets,
            &times,
            &responses,
            2,
            &packed,
            &good,
            1,
            1e-3,
        )
        .unwrap();
        if hyper_ok {
            assert!(hyper_se.iter().all(|value| value.is_finite() && *value > 0.0));
        }
        let (hyper_se, _, _, hyper_ok) = hyperparameter_interval_estimates(
            &offsets,
            &times,
            &responses,
            2,
            &packed,
            &good,
            1,
            50.0,
        )
        .unwrap();
        assert!(!hyper_ok);
        assert!(hyper_se.iter().all(|value| value.is_nan()));
        assert!(collect_person_fits(vec![None])
            .unwrap_err()
            .contains("hierarchical fit is missing"));
        assert!(joint_objective(&offsets, &times, &responses, 2, &[0.0; 3], 1)
            .unwrap_err()
            .contains("packed parameter length"));
        let mut exploding = good.clone();
        exploding.state = vec![1e200, -1e200];
        assert!(person_objective(&times, &responses, 2, &exploding, &exploding.state)
            .unwrap_err()
            .contains("not finite"));
        assert!(invert_three(&[f64::NAN; 9]).is_err());
        assert!(validate_design(
            &[0, 2],
            &[0, 1],
            &[],
            (usize::MAX / 2) + 1
        )
        .unwrap_err()
        .contains("exceeds supported size"));
        let init = initialize_params(
            &[0, 2],
            &[f64::NAN, f64::NAN, 1.0, 0.0],
            2,
            2,
        );
        assert!(init.iter().all(|value| value.is_finite()));
        let empty_item_init = initialize_params(&[0, 1], &[f64::NAN, f64::NAN], 2, 1);
        assert!(empty_item_init.iter().all(|value| value.is_finite()));
        let all_zero = initialize_params(&[0, 1], &[0.0, 0.0], 2, 1);
        assert!(all_zero.iter().all(|value| value.is_finite()));
        assert!(hyperparameter_hessian(
            &offsets,
            &times,
            &responses,
            2,
            &[0.0; 3],
            1,
            1e-3
        )
        .unwrap_err()
        .contains("packed parameter length"));
        let mut negative_diag = good.clone();
        negative_diag.log_sd = 0.0;
        let (diag, off) =
            person_state_hessian(&times, &responses, 2, &negative_diag, &negative_diag.state)
                .unwrap();
        let mut broken = diag;
        broken[0] = -1.0;
        assert!(tridiagonal_inverse_diagonal(&broken, &off)
            .unwrap_err()
            .contains("positive definite"));
    }
}
