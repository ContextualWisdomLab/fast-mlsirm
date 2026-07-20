//! IRT scale linking for separately-calibrated common-item designs (Kolen &
//! Brennan 2014, ch. 6): the moment methods (mean/mean, mean/sigma) and the
//! characteristic-curve methods of Haebara (1980) and Stocking & Lord (1983).
//! Motivated by the mixed-format/multi-study linking papers in the corpus
//! (Kim & Lee 2006; Yao & Boughton 2009; Brossman & Lee 2013) — this module
//! covers the unidimensional 2PL/1PL common-item case those procedures reduce
//! to for the serving scale.
//!
//! Convention: new-form abilities relate to the old (reference) scale by
//! `theta_old = A * theta_new + B`. Item parameters are carried in the engine's
//! `eta = a*theta + b` form (a = slope, b = intercept). Substituting
//! `theta_new = (theta_old - B) / A` transforms a new-form item onto the old
//! scale as
//!   a* = a_new / A,   b* = b_new - (a_new / A) * B
//! (equivalently the classical `a_O = a_N/A`, `b_O = A b_N + B` on the
//! slope/difficulty parameterization, with difficulty `-b/a`).
//!
//! # References (APA 7th ed.)
//!
//! Haebara, T. (1980). Equating logistic ability scales by a weighted least
//! squares method. *Japanese Psychological Research, 22*(3), 144–149.
//! https://doi.org/10.4992/psycholres1954.22.144
//!
//! Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
//! linking: Methods and practices* (3rd ed.). Springer.
//! https://doi.org/10.1007/978-1-4939-0317-7
//!
//! Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item
//! response theory. *Applied Psychological Measurement, 7*(2), 201–210.
//! https://doi.org/10.1177/014662168300700208

const NM_MAX_ITER: usize = 500;
const NM_OBJECTIVE_RTOL: f64 = 1e-14;
const NM_PARAMETER_RTOL: f64 = 1e-10;

/// Linking coefficients `theta_old = slope * theta_new + intercept`.
#[derive(Clone, Copy, Debug)]
pub struct LinkResult {
    pub slope: f64,
    pub intercept: f64,
    /// Objective at the solution (0 for the moment methods, which are closed
    /// form; the characteristic-curve loss for Haebara / Stocking-Lord).
    pub criterion: f64,
    pub n_iter: usize,
    /// `true` for a closed-form moment solution or when both Nelder–Mead
    /// simplex stopping criteria are met.
    pub converged: bool,
    /// `closed_form`, `tolerance_met`, or `max_iter_reached`.
    pub termination_reason: &'static str,
    pub max_iter: usize,
    pub final_objective_span: f64,
    pub objective_tolerance: f64,
    pub final_parameter_span: f64,
    pub parameter_tolerance: f64,
}

/// Linking method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkMethod {
    MeanMean,
    MeanSigma,
    Haebara,
    StockingLord,
}

impl LinkMethod {
    pub fn parse(name: &str) -> Option<LinkMethod> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "meanmean" | "mm" => Some(LinkMethod::MeanMean),
            "meansigma" | "ms" => Some(LinkMethod::MeanSigma),
            "haebara" | "hb" => Some(LinkMethod::Haebara),
            "stockinglord" | "sl" => Some(LinkMethod::StockingLord),
            _ => None,
        }
    }
}

#[inline]
fn p2pl(a: f64, b: f64, theta: f64) -> f64 {
    1.0 / (1.0 + (-(a * theta + b)).exp())
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Closed-form moment coefficients. `difficulty_i = -b_i / a_i`.
fn moment(
    a_old: &[f64],
    b_old: &[f64],
    a_new: &[f64],
    b_new: &[f64],
    sigma: bool,
) -> Result<(f64, f64), String> {
    let d_old: Vec<f64> = a_old.iter().zip(b_old).map(|(&a, &b)| -b / a).collect();
    let d_new: Vec<f64> = a_new.iter().zip(b_new).map(|(&a, &b)| -b / a).collect();
    let slope = if sigma {
        let (so, sn) = (sd(&d_old), sd(&d_new));
        if !(so > 0.0 && sn > 0.0) {
            return Err(
                "mean/sigma linking requires non-zero difficulty spread on both scales".into(),
            );
        }
        so / sn
    } else {
        // mean/mean uses the discriminations: a_O = a_N / A
        let mo = mean(a_old);
        mean(a_new) / mo
    };
    let intercept = mean(&d_old) - slope * mean(&d_new);
    if !(slope.is_finite() && slope > 0.0 && intercept.is_finite()) {
        return Err("linking coefficients must be finite with a positive slope".into());
    }
    Ok((slope, intercept))
}

/// Characteristic-curve objective at `(slope A, intercept B)`.
fn cc_objective(
    slope: f64,
    intercept: f64,
    a_old: &[f64],
    b_old: &[f64],
    a_new: &[f64],
    b_new: &[f64],
    theta: &[f64],
    weight: &[f64],
    stocking_lord: bool,
) -> f64 {
    if !(slope > 1e-6) || !slope.is_finite() || !intercept.is_finite() {
        return 1e18;
    }
    let n_items = a_old.len();
    let mut total = 0.0;
    for (q, &th) in theta.iter().enumerate() {
        if stocking_lord {
            let mut tcc_old = 0.0;
            let mut tcc_new = 0.0;
            for i in 0..n_items {
                tcc_old += p2pl(a_old[i], b_old[i], th);
                let a_star = a_new[i] / slope;
                let b_star = b_new[i] - a_star * intercept;
                tcc_new += p2pl(a_star, b_star, th);
            }
            let d = tcc_old - tcc_new;
            total += weight[q] * d * d;
        } else {
            let mut acc = 0.0;
            for i in 0..n_items {
                let a_star = a_new[i] / slope;
                let b_star = b_new[i] - a_star * intercept;
                let d = p2pl(a_old[i], b_old[i], th) - p2pl(a_star, b_star, th);
                acc += d * d;
            }
            total += weight[q] * acc;
        }
    }
    total
}

#[derive(Clone, Copy, Debug)]
struct NelderMeadResult {
    x: [f64; 2],
    objective: f64,
    n_iter: usize,
    converged: bool,
    final_objective_span: f64,
    objective_tolerance: f64,
    final_parameter_span: f64,
    parameter_tolerance: f64,
}

fn link_termination_reason(converged: bool) -> &'static str {
    if converged {
        "tolerance_met"
    } else {
        "max_iter_reached"
    }
}

fn simplex_diagnostics(
    simplex: &[[f64; 2]; 3],
    fval: &[f64; 3],
    best: usize,
    worst: usize,
) -> (f64, f64, f64, f64) {
    let objective_span = (fval[worst] - fval[best]).abs();
    let parameter_span = simplex
        .iter()
        .map(|vertex| {
            (vertex[0] - simplex[best][0])
                .abs()
                .max((vertex[1] - simplex[best][1]).abs())
        })
        .fold(0.0, f64::max);
    let objective_tolerance = NM_OBJECTIVE_RTOL * (1.0 + fval[best].abs());
    let parameter_scale = simplex[best][0].abs().max(simplex[best][1].abs());
    let parameter_tolerance = NM_PARAMETER_RTOL * (1.0 + parameter_scale);
    (
        objective_span,
        objective_tolerance,
        parameter_span,
        parameter_tolerance,
    )
}

/// Nelder–Mead minimization of a 2-parameter objective from `x0`.
fn nelder_mead(f: &dyn Fn(f64, f64) -> f64, x0: [f64; 2]) -> NelderMeadResult {
    // simplex vertices
    let mut simplex = [
        x0,
        [x0[0] + 0.10 * (1.0 + x0[0].abs()), x0[1]],
        [x0[0], x0[1] + 0.10 * (1.0 + x0[1].abs())],
    ];
    let mut fval = [
        f(simplex[0][0], simplex[0][1]),
        f(simplex[1][0], simplex[1][1]),
        f(simplex[2][0], simplex[2][1]),
    ];
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    let mut iters = 0;
    let mut converged = false;
    for it in 0..NM_MAX_ITER {
        iters = it + 1;
        // order vertices by value
        let mut order = [0usize, 1, 2];
        order.sort_by(|&i, &j| fval[i].total_cmp(&fval[j]));
        let (lo, mid, hi) = (order[0], order[1], order[2]);
        let (objective_span, objective_tolerance, parameter_span, parameter_tolerance) =
            simplex_diagnostics(&simplex, &fval, lo, hi);
        if objective_span <= objective_tolerance && parameter_span <= parameter_tolerance {
            converged = true;
            break;
        }
        // centroid of the two best
        let cen = [
            0.5 * (simplex[lo][0] + simplex[mid][0]),
            0.5 * (simplex[lo][1] + simplex[mid][1]),
        ];
        // reflection
        let refl = [
            cen[0] + alpha * (cen[0] - simplex[hi][0]),
            cen[1] + alpha * (cen[1] - simplex[hi][1]),
        ];
        let f_refl = f(refl[0], refl[1]);
        if f_refl < fval[lo] {
            // expansion
            let exp = [
                cen[0] + gamma * (refl[0] - cen[0]),
                cen[1] + gamma * (refl[1] - cen[1]),
            ];
            let f_exp = f(exp[0], exp[1]);
            if f_exp < f_refl {
                simplex[hi] = exp;
                fval[hi] = f_exp;
            } else {
                simplex[hi] = refl;
                fval[hi] = f_refl;
            }
        } else if f_refl < fval[mid] {
            simplex[hi] = refl;
            fval[hi] = f_refl;
        } else {
            // contraction
            let con = [
                cen[0] + rho * (simplex[hi][0] - cen[0]),
                cen[1] + rho * (simplex[hi][1] - cen[1]),
            ];
            let f_con = f(con[0], con[1]);
            if f_con < fval[hi] {
                simplex[hi] = con;
                fval[hi] = f_con;
            } else {
                // shrink toward the best
                for &v in &[mid, hi] {
                    simplex[v] = [
                        simplex[lo][0] + sigma * (simplex[v][0] - simplex[lo][0]),
                        simplex[lo][1] + sigma * (simplex[v][1] - simplex[lo][1]),
                    ];
                    fval[v] = f(simplex[v][0], simplex[v][1]);
                }
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| fval[i].total_cmp(&fval[j]));
    let (best, worst) = (order[0], order[2]);
    let (final_objective_span, objective_tolerance, final_parameter_span, parameter_tolerance) =
        simplex_diagnostics(&simplex, &fval, best, worst);
    NelderMeadResult {
        x: simplex[best],
        objective: fval[best],
        n_iter: iters,
        converged,
        final_objective_span,
        objective_tolerance,
        final_parameter_span,
        parameter_tolerance,
    }
}

/// Link a separately-calibrated new form onto the old (reference) scale using
/// common items. `theta`/`weight` are the quadrature the characteristic-curve
/// methods integrate over (ignored by the moment methods).
#[allow(clippy::too_many_arguments)]
pub fn irt_link(
    a_old: &[f64],
    b_old: &[f64],
    a_new: &[f64],
    b_new: &[f64],
    theta: &[f64],
    weight: &[f64],
    method: LinkMethod,
) -> Result<LinkResult, String> {
    let n = a_old.len();
    if n < 2 || b_old.len() != n || a_new.len() != n || b_new.len() != n {
        return Err("need >= 2 common items and matching-length parameter slices".into());
    }
    if a_old
        .iter()
        .chain(a_new)
        .any(|&a| !a.is_finite() || a <= 0.0)
    {
        return Err("slopes must be positive and finite".into());
    }
    if b_old.iter().chain(b_new).any(|b| !b.is_finite()) {
        return Err("intercepts must be finite".into());
    }
    match method {
        LinkMethod::MeanMean | LinkMethod::MeanSigma => {
            let (slope, intercept) =
                moment(a_old, b_old, a_new, b_new, method == LinkMethod::MeanSigma)?;
            Ok(LinkResult {
                slope,
                intercept,
                criterion: 0.0,
                n_iter: 0,
                converged: true,
                termination_reason: "closed_form",
                max_iter: 0,
                final_objective_span: 0.0,
                objective_tolerance: 0.0,
                final_parameter_span: 0.0,
                parameter_tolerance: 0.0,
            })
        }
        LinkMethod::Haebara | LinkMethod::StockingLord => {
            if theta.len() != weight.len() || theta.is_empty() {
                return Err("theta and weight must be non-empty and equal length".into());
            }
            if theta.iter().any(|value| !value.is_finite()) {
                return Err("theta nodes must be finite".into());
            }
            if weight
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err("quadrature weights must be finite and non-negative".into());
            }
            let weight_sum: f64 = weight.iter().sum();
            if !(weight_sum.is_finite() && weight_sum > 0.0) {
                return Err("quadrature weights must have a finite positive sum".into());
            }
            let normalized_weight: Vec<f64> =
                weight.iter().map(|value| value / weight_sum).collect();
            let sl = method == LinkMethod::StockingLord;
            // Prefer the mean/sigma start, but mean/mean remains identifiable
            // when one difficulty distribution has zero spread.
            let (a0, b0) = moment(a_old, b_old, a_new, b_new, true)
                .or_else(|_| moment(a_old, b_old, a_new, b_new, false))?;
            let optimization = nelder_mead(
                &|slope, intercept| {
                    cc_objective(
                        slope,
                        intercept,
                        a_old,
                        b_old,
                        a_new,
                        b_new,
                        theta,
                        &normalized_weight,
                        sl,
                    )
                },
                [a0, b0],
            );
            Ok(LinkResult {
                slope: optimization.x[0],
                intercept: optimization.x[1],
                criterion: optimization.objective,
                n_iter: optimization.n_iter,
                converged: optimization.converged,
                termination_reason: link_termination_reason(optimization.converged),
                max_iter: NM_MAX_ITER,
                final_objective_span: optimization.final_objective_span,
                objective_tolerance: optimization.objective_tolerance,
                final_parameter_span: optimization.final_parameter_span,
                parameter_tolerance: optimization.parameter_tolerance,
            })
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/linking_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/linking_branch_tests.rs"]
mod branch_tests;
