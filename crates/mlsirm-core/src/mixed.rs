//! Mixed-format marginal maximum-likelihood calibration.
//!
//! Each item keeps its own conditional response function while all items share
//! the same standard-normal trait distribution. LSIRM items additionally share
//! a standard-normal latent-space coordinate; non-spatial items are constant on
//! that integration axis and therefore integrate it out exactly.
//!
//! The heterogeneous likelihood is the product of the item-specific cells, as
//! in the random-coefficients multinomial-logit framework and `mirt`'s per-item
//! `itemtype` contract. The ideal-point, GGUM, nominal, and LSIRM formulas are
//! not blended into a surrogate common formula.
//!
//! # References
//!
//! Adams, R. J., Wilson, M., & Wang, W.-C. (1997). The multidimensional random
//! coefficients multinomial logit model. *Applied Psychological Measurement,
//! 21*(1), 1–23. https://doi.org/10.1177/0146621697211001
//!
//! Bock, R. D. (1972). Estimating item parameters and latent ability when
//! responses are scored in two or more nominal categories. *Psychometrika,
//! 37*(1), 29–51. https://doi.org/10.1007/BF02291411
//!
//! Maydeu-Olivares, A., Hernández, A., & McDonald, R. P. (2006). A
//! multidimensional ideal point item response theory model for binary data.
//! *Multivariate Behavioral Research, 41*(4), 445–472.
//! https://doi.org/10.1207/s15327906mbr4104_2
//!
//! Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (1998). The generalized
//! graded unfolding model: A general parametric item response model for
//! unfolding graded responses. *ETS Research Report Series, 1998*(2), i–53.
//! https://doi.org/10.1002/j.2333-8504.1998.tb01781.x
//!
//! Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
//! unobserved item-respondent interactions: A latent space item response model
//! with interaction map. *Psychometrika, 86*(2), 378–403.
//! https://doi.org/10.1007/s11336-021-09762-5

use std::thread;

use crate::poly::{gpcm_logprobs, grm_logprobs, solve_small};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedItemKind {
    TwoPl,
    Grm,
    Gpcm,
    Nominal,
    Ideal,
    Ggum,
    Lsirm,
    LsirmGrm,
    LsirmGpcm,
}

impl MixedItemKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "2pl" | "dichotomous" | "binary" => Ok(Self::TwoPl),
            "grm" | "graded" => Ok(Self::Grm),
            "gpcm" => Ok(Self::Gpcm),
            "nominal" | "nrm" => Ok(Self::Nominal),
            "ideal" | "ideal_point" => Ok(Self::Ideal),
            "ggum" => Ok(Self::Ggum),
            "lsirm" | "lsirm_2pl" => Ok(Self::Lsirm),
            "lsirm_grm" => Ok(Self::LsirmGrm),
            "lsirm_gpcm" => Ok(Self::LsirmGpcm),
            other => Err(format!(
                "unsupported mixed item model {other:?}; expected one of: 2pl, grm, gpcm, nominal, ideal, ggum, lsirm, lsirm_grm, lsirm_gpcm"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TwoPl => "2pl",
            Self::Grm => "grm",
            Self::Gpcm => "gpcm",
            Self::Nominal => "nominal",
            Self::Ideal => "ideal",
            Self::Ggum => "ggum",
            Self::Lsirm => "lsirm",
            Self::LsirmGrm => "lsirm_grm",
            Self::LsirmGpcm => "lsirm_gpcm",
        }
    }

    fn is_spatial(self) -> bool {
        matches!(self, Self::Lsirm | Self::LsirmGrm | Self::LsirmGpcm)
    }
}

#[derive(Clone, Debug)]
pub struct MixedItemSpec {
    pub kind: MixedItemKind,
    pub n_categories: usize,
}

#[derive(Clone, Debug)]
pub struct MixedItemEstimate {
    pub kind: MixedItemKind,
    pub n_categories: usize,
    pub slope: Option<f64>,
    pub intercepts: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub scores: Vec<f64>,
    pub location: Option<f64>,
    pub zeta: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MixedFit {
    pub items: Vec<MixedItemEstimate>,
    pub theta_eap: Vec<f64>,
    pub theta_sd: Vec<f64>,
    pub xi_eap: Vec<f64>,
    pub latent_dim: usize,
    pub loglik: f64,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub n_threads: usize,
}

#[derive(Clone)]
struct Grid {
    theta: Vec<f64>,
    theta_logw: Vec<f64>,
    xi: Vec<f64>,
    xi_logw: Vec<f64>,
    latent_dim: usize,
    n_xi: usize,
}

impl Grid {
    fn cell(&self) -> usize {
        self.theta.len() * self.n_xi
    }
}

fn tensor_grid(q_xi: usize, latent_dim: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (nodes, weights) =
        crate::quadrature::gh_rule(q_xi).ok_or_else(|| format!("unsupported q_xi {q_xi}"))?;
    let n_xi = nodes
        .len()
        .checked_pow(latent_dim as u32)
        .ok_or("q_xi ** latent_dim overflow")?;
    if n_xi > 200_000 {
        return Err("q_xi ** latent_dim exceeds the tensor-grid limit".into());
    }
    let mut grid = vec![0.0; n_xi * latent_dim];
    let mut logw = vec![0.0; n_xi];
    for idx in 0..n_xi {
        let mut rem = idx;
        for d in 0..latent_dim {
            let j = rem % nodes.len();
            rem /= nodes.len();
            grid[idx * latent_dim + d] = nodes[j];
            logw[idx] += weights[j].ln();
        }
    }
    Ok((grid, logw))
}

fn build_grid(
    specs: &[MixedItemSpec],
    latent_dim: usize,
    q_theta: usize,
    q_xi: usize,
) -> Result<Grid, String> {
    let (theta, theta_w) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let spatial = specs.iter().any(|s| s.kind.is_spatial());
    let (xi, xi_logw, used_dim) = if spatial {
        if !(1..=3).contains(&latent_dim) {
            return Err("latent_dim must be in 1..=3 when LSIRM items are present".into());
        }
        let (x, w) = tensor_grid(q_xi, latent_dim)?;
        (x, w, latent_dim)
    } else {
        (Vec::new(), vec![0.0], 0)
    };
    Ok(Grid {
        theta: theta.to_vec(),
        theta_logw: theta_w.iter().map(|w| w.ln()).collect(),
        xi,
        n_xi: xi_logw.len(),
        xi_logw,
        latent_dim: used_dim,
    })
}

fn ordered_values(raw: &[f64]) -> Vec<f64> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(raw.len());
    values.push(raw[0]);
    for &log_gap in &raw[1..] {
        let gap = log_gap.clamp(-8.0, 5.0).exp().max(1e-4);
        values.push(values.last().copied().unwrap() - gap);
    }
    values
}

fn ordered_raw(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut raw = Vec::with_capacity(values.len());
    raw.push(values[0]);
    for pair in values.windows(2) {
        raw.push((pair[0] - pair[1]).max(1e-4).ln());
    }
    raw
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn softmax_log(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = scores.iter().map(|v| (v - m).exp()).sum();
    scores.iter().map(|v| v - m - z.ln()).collect()
}

fn distance(xi: &[f64], zeta: &[f64]) -> f64 {
    let d2 = xi
        .iter()
        .zip(zeta)
        .map(|(x, z)| (x - z) * (x - z))
        .sum::<f64>();
    (d2 + 1e-8).sqrt()
}

fn item_logprobs(
    spec: &MixedItemSpec,
    params: &[f64],
    theta: f64,
    xi: &[f64],
    latent_dim: usize,
) -> Vec<f64> {
    let k = spec.n_categories;
    match spec.kind {
        MixedItemKind::TwoPl => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            gpcm_logprobs(a * theta, &[0.0, 1.0], &[0.0, params[1]])
        }
        MixedItemKind::Grm => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            grm_logprobs(a * theta, &ordered_values(&params[1..]))
        }
        MixedItemKind::Gpcm => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut intercepts = vec![0.0; k];
            intercepts[1..].copy_from_slice(&params[1..k]);
            gpcm_logprobs(a * theta, &scores, &intercepts)
        }
        MixedItemKind::Nominal => {
            let c = k - 1;
            let mut scores = vec![0.0; k];
            let mut intercepts = vec![0.0; k];
            scores[1..].copy_from_slice(&params[..c]);
            intercepts[1..].copy_from_slice(&params[c..2 * c]);
            gpcm_logprobs(theta, &scores, &intercepts)
        }
        MixedItemKind::Ideal => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            let z = a * (theta - params[1]);
            let p1 = (-0.5 * z * z).exp().clamp(1e-15, 1.0 - 1e-15);
            vec![(-p1).ln_1p(), p1.ln()]
        }
        MixedItemKind::Ggum => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            let b = params[1];
            let thresholds = ordered_values(&params[2..]);
            let dist = (a * (theta - b)).abs();
            let m = (2 * (k - 1) + 1) as f64;
            let mut cumulative = 0.0;
            let mut numerators = Vec::with_capacity(k);
            for z in 0..k {
                if z > 0 {
                    cumulative += a * thresholds[z - 1];
                }
                numerators.push(logaddexp(
                    z as f64 * dist + cumulative,
                    (m - z as f64) * dist + cumulative,
                ));
            }
            softmax_log(&numerators)
        }
        MixedItemKind::Lsirm | MixedItemKind::LsirmGrm | MixedItemKind::LsirmGpcm => {
            let a = params[0].clamp(-5.0, 4.0).exp();
            let cat_n = k - 1;
            let zeta = &params[1 + cat_n..1 + cat_n + latent_dim];
            let base = a * theta - distance(xi, zeta);
            match spec.kind {
                MixedItemKind::Lsirm => gpcm_logprobs(base, &[0.0, 1.0], &[0.0, params[1]]),
                MixedItemKind::LsirmGrm => grm_logprobs(base, &ordered_values(&params[1..k])),
                MixedItemKind::LsirmGpcm => {
                    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                    let mut intercepts = vec![0.0; k];
                    intercepts[1..].copy_from_slice(&params[1..k]);
                    gpcm_logprobs(base, &scores, &intercepts)
                }
                _ => unreachable!(),
            }
        }
    }
}

fn parameter_count(spec: &MixedItemSpec, latent_dim: usize) -> usize {
    match spec.kind {
        MixedItemKind::TwoPl | MixedItemKind::Ideal => 2,
        MixedItemKind::Grm | MixedItemKind::Gpcm => spec.n_categories,
        MixedItemKind::Nominal => 2 * (spec.n_categories - 1),
        MixedItemKind::Ggum => 2 + spec.n_categories - 1,
        MixedItemKind::Lsirm | MixedItemKind::LsirmGrm | MixedItemKind::LsirmGpcm => {
            spec.n_categories + latent_dim
        }
    }
}

fn initial_params(
    spec: &MixedItemSpec,
    freq: &[f64],
    item: usize,
    n_items: usize,
    latent_dim: usize,
) -> Vec<f64> {
    let k = spec.n_categories;
    let mut p = vec![0.0; parameter_count(spec, latent_dim)];
    match spec.kind {
        MixedItemKind::TwoPl | MixedItemKind::Lsirm => {
            p[1] = (freq[1] / freq[0]).ln();
        }
        MixedItemKind::Grm | MixedItemKind::LsirmGrm => {
            let mut thresholds = vec![0.0; k - 1];
            let mut cumulative = 0.0;
            for category in (1..k).rev() {
                cumulative += freq[category];
                let c = cumulative.clamp(1e-4, 1.0 - 1e-4);
                thresholds[category - 1] = (c / (1.0 - c)).ln();
            }
            p[1..k].copy_from_slice(&ordered_raw(&thresholds));
        }
        MixedItemKind::Gpcm | MixedItemKind::LsirmGpcm => {
            for category in 1..k {
                p[category] = (freq[category] / freq[0]).ln();
            }
        }
        MixedItemKind::Nominal => {
            let c = k - 1;
            for category in 1..k {
                p[category - 1] = category as f64;
                p[c + category - 1] = (freq[category] / freq[0]).ln();
            }
        }
        MixedItemKind::Ideal => {
            p[0] = 0.0;
            p[1] = if freq[1] < 0.4 { 1.0 } else { 0.0 };
        }
        MixedItemKind::Ggum => {
            p[0] = 0.0;
            p[1] = 0.0;
            let thresholds: Vec<f64> = (0..k - 1).map(|j| 1.0 - 0.5 * j as f64).collect();
            p[2..].copy_from_slice(&ordered_raw(&thresholds));
        }
    }
    if spec.kind.is_spatial() {
        let start = p.len() - latent_dim;
        let angle = 2.0 * std::f64::consts::PI * item as f64 / n_items.max(1) as f64;
        p[start] = 0.5 * angle.cos();
        if latent_dim > 1 {
            p[start + 1] = 0.5 * angle.sin();
        }
        if latent_dim > 2 {
            p[start + 2] = 0.25 * (2.0 * angle).cos();
        }
    }
    p
}

fn item_table(spec: &MixedItemSpec, params: &[f64], grid: &Grid) -> Vec<f64> {
    let mut table = vec![0.0; grid.cell() * spec.n_categories];
    for (t, &theta) in grid.theta.iter().enumerate() {
        for x in 0..grid.n_xi {
            let xi = if grid.latent_dim == 0 {
                &[][..]
            } else {
                &grid.xi[x * grid.latent_dim..(x + 1) * grid.latent_dim]
            };
            let lp = item_logprobs(spec, params, theta, xi, grid.latent_dim);
            let node = t * grid.n_xi + x;
            table[node * spec.n_categories..(node + 1) * spec.n_categories].copy_from_slice(&lp);
        }
    }
    table
}

fn build_tables(specs: &[MixedItemSpec], params: &[Vec<f64>], grid: &Grid) -> Vec<Vec<f64>> {
    specs
        .iter()
        .zip(params)
        .map(|(spec, p)| item_table(spec, p, grid))
        .collect()
}

#[derive(Debug)]
struct EStep {
    loglik: f64,
    counts: Vec<Vec<f64>>,
}

fn empty_counts(specs: &[MixedItemSpec], cell: usize) -> Vec<Vec<f64>> {
    specs
        .iter()
        .map(|spec| vec![0.0; cell * spec.n_categories])
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn e_step_range(
    y: &[usize],
    observed: &[bool],
    n_items: usize,
    specs: &[MixedItemSpec],
    tables: &[Vec<f64>],
    grid: &Grid,
    start: usize,
    end: usize,
) -> EStep {
    let cell = grid.cell();
    let mut counts = empty_counts(specs, cell);
    let mut log_node = vec![0.0; cell];
    let mut loglik = 0.0;
    for person in start..end {
        for t in 0..grid.theta.len() {
            for x in 0..grid.n_xi {
                log_node[t * grid.n_xi + x] = grid.theta_logw[t] + grid.xi_logw[x];
            }
        }
        for item in 0..n_items {
            if !observed[person * n_items + item] {
                continue;
            }
            let response = y[person * n_items + item];
            let k = specs[item].n_categories;
            for node in 0..cell {
                log_node[node] += tables[item][node * k + response];
            }
        }
        let mx = log_node.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = log_node.iter().map(|v| (v - mx).exp()).sum();
        loglik += mx + denom.ln();
        for item in 0..n_items {
            if !observed[person * n_items + item] {
                continue;
            }
            let response = y[person * n_items + item];
            let k = specs[item].n_categories;
            for node in 0..cell {
                counts[item][node * k + response] += (log_node[node] - mx).exp() / denom;
            }
        }
    }
    EStep { loglik, counts }
}

#[allow(clippy::too_many_arguments)]
fn e_step(
    y: &[usize],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    specs: &[MixedItemSpec],
    tables: &[Vec<f64>],
    grid: &Grid,
    n_threads: usize,
) -> EStep {
    let workers = n_threads.min(n_persons).max(1);
    if workers == 1 || n_persons < 256 {
        return e_step_range(y, observed, n_items, specs, tables, grid, 0, n_persons);
    }
    let chunk = n_persons.div_ceil(workers);
    let mut partials = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let start = worker * chunk;
            let end = (start + chunk).min(n_persons);
            if start >= end {
                break;
            }
            handles.push(scope.spawn(move || {
                e_step_range(y, observed, n_items, specs, tables, grid, start, end)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("mixed E-step worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut out = EStep {
        loglik: 0.0,
        counts: empty_counts(specs, grid.cell()),
    };
    for partial in partials.drain(..) {
        out.loglik += partial.loglik;
        for (dst_item, src_item) in out.counts.iter_mut().zip(partial.counts) {
            for (dst, src) in dst_item.iter_mut().zip(src_item) {
                *dst += src;
            }
        }
    }
    out
}

fn item_objective(spec: &MixedItemSpec, params: &[f64], grid: &Grid, counts: &[f64]) -> f64 {
    let table = item_table(spec, params, grid);
    -counts.iter().zip(table).map(|(r, lp)| r * lp).sum::<f64>()
}

fn numeric_gradient(spec: &MixedItemSpec, params: &[f64], grid: &Grid, counts: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];
    for j in 0..params.len() {
        let h = 1e-5 * (1.0 + params[j].abs());
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[j] += h;
        minus[j] -= h;
        grad[j] = (item_objective(spec, &plus, grid, counts)
            - item_objective(spec, &minus, grid, counts))
            / (2.0 * h);
    }
    grad
}

fn clamp_params(spec: &MixedItemSpec, values: &mut [f64], latent_dim: usize) {
    for value in values.iter_mut() {
        *value = value.clamp(-12.0, 12.0);
    }
    if !matches!(spec.kind, MixedItemKind::Nominal) {
        values[0] = values[0].clamp(-5.0, 4.0);
    }
    if spec.kind.is_spatial() {
        let start = values.len() - latent_dim;
        for value in &mut values[start..] {
            *value = value.clamp(-6.0, 6.0);
        }
    }
}

fn m_step_item(
    spec: &MixedItemSpec,
    start: &[f64],
    grid: &Grid,
    counts: &[f64],
    max_steps: usize,
) -> Vec<f64> {
    let mut params = start.to_vec();
    for _ in 0..max_steps {
        let f0 = item_objective(spec, &params, grid, counts);
        let grad = numeric_gradient(spec, &params, grid, counts);
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-6 {
            break;
        }
        let n = params.len();
        let mut hessian = vec![vec![0.0; n]; n];
        for j in 0..n {
            let h = 2e-4 * (1.0 + params[j].abs());
            let mut shifted = params.clone();
            shifted[j] += h;
            let next_grad = numeric_gradient(spec, &shifted, grid, counts);
            for row in 0..n {
                hessian[row][j] = (next_grad[row] - grad[row]) / h;
            }
        }
        for row in 0..n {
            for col in 0..n {
                hessian[row][col] = 0.5 * (hessian[row][col] + hessian[col][row]);
            }
            hessian[row][row] += 1e-4;
        }
        let mut step = solve_small(hessian, grad.clone());
        if !step.iter().all(|s| s.is_finite())
            || grad.iter().zip(&step).map(|(g, s)| g * s).sum::<f64>() <= 0.0
        {
            step = grad.clone();
        }
        let max_abs = step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        if max_abs > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_abs;
            }
        }
        let mut alpha = 1.0;
        let directional = grad.iter().zip(&step).map(|(g, s)| g * s).sum::<f64>();
        let mut accepted = false;
        for _ in 0..24 {
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(p, s)| p - alpha * s)
                .collect();
            clamp_params(spec, &mut candidate, grid.latent_dim);
            let fc = item_objective(spec, &candidate, grid, counts);
            if fc.is_finite() && fc <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_abs < 1e-7 {
            break;
        }
    }
    params
}

fn m_step(
    specs: &[MixedItemSpec],
    params: &[Vec<f64>],
    grid: &Grid,
    counts: &[Vec<f64>],
    n_threads: usize,
) -> Vec<Vec<f64>> {
    let n_items = specs.len();
    let workers = n_threads.min(n_items).max(1);
    if workers == 1 || n_items < 4 {
        return (0..n_items)
            .map(|i| m_step_item(&specs[i], &params[i], grid, &counts[i], 6))
            .collect();
    }
    let chunk = n_items.div_ceil(workers);
    let mut pieces = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let start = worker * chunk;
            let end = (start + chunk).min(n_items);
            if start >= end {
                break;
            }
            handles.push(scope.spawn(move || {
                let fitted = (start..end)
                    .map(|i| m_step_item(&specs[i], &params[i], grid, &counts[i], 6))
                    .collect::<Vec<_>>();
                (start, fitted)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("mixed M-step worker panicked"))
            .collect::<Vec<_>>()
    });
    pieces.sort_by_key(|(start, _)| *start);
    pieces.into_iter().flat_map(|(_, fitted)| fitted).collect()
}

fn public_estimate(spec: &MixedItemSpec, params: &[f64], latent_dim: usize) -> MixedItemEstimate {
    let k = spec.n_categories;
    let mut out = MixedItemEstimate {
        kind: spec.kind,
        n_categories: k,
        slope: None,
        intercepts: Vec::new(),
        thresholds: Vec::new(),
        scores: Vec::new(),
        location: None,
        zeta: Vec::new(),
    };
    match spec.kind {
        MixedItemKind::TwoPl => {
            out.slope = Some(params[0].exp());
            out.intercepts = vec![params[1]];
        }
        MixedItemKind::Grm => {
            out.slope = Some(params[0].exp());
            out.thresholds = ordered_values(&params[1..]);
        }
        MixedItemKind::Gpcm => {
            out.slope = Some(params[0].exp());
            out.intercepts = params[1..k].to_vec();
        }
        MixedItemKind::Nominal => {
            let c = k - 1;
            out.scores = params[..c].to_vec();
            out.intercepts = params[c..2 * c].to_vec();
        }
        MixedItemKind::Ideal => {
            out.slope = Some(params[0].exp());
            out.location = Some(params[1]);
        }
        MixedItemKind::Ggum => {
            out.slope = Some(params[0].exp());
            out.location = Some(params[1]);
            out.thresholds = ordered_values(&params[2..]);
        }
        MixedItemKind::Lsirm | MixedItemKind::LsirmGrm | MixedItemKind::LsirmGpcm => {
            out.slope = Some(params[0].exp());
            match spec.kind {
                MixedItemKind::Lsirm => out.intercepts = vec![params[1]],
                MixedItemKind::LsirmGrm => out.thresholds = ordered_values(&params[1..k]),
                MixedItemKind::LsirmGpcm => out.intercepts = params[1..k].to_vec(),
                _ => unreachable!(),
            }
            out.zeta = params[params.len() - latent_dim..].to_vec();
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn final_scores(
    y: &[usize],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    specs: &[MixedItemSpec],
    tables: &[Vec<f64>],
    grid: &Grid,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let cell = grid.cell();
    let mut theta_eap = vec![0.0; n_persons];
    let mut theta_sd = vec![0.0; n_persons];
    let mut xi_eap = vec![0.0; n_persons * grid.latent_dim];
    let mut log_node = vec![0.0; cell];
    for person in 0..n_persons {
        for t in 0..grid.theta.len() {
            for x in 0..grid.n_xi {
                log_node[t * grid.n_xi + x] = grid.theta_logw[t] + grid.xi_logw[x];
            }
        }
        for item in 0..n_items {
            if !observed[person * n_items + item] {
                continue;
            }
            let response = y[person * n_items + item];
            let k = specs[item].n_categories;
            for node in 0..cell {
                log_node[node] += tables[item][node * k + response];
            }
        }
        let mx = log_node.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = log_node.iter().map(|v| (v - mx).exp()).sum();
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        for t in 0..grid.theta.len() {
            for x in 0..grid.n_xi {
                let post = (log_node[t * grid.n_xi + x] - mx).exp() / denom;
                m1 += post * grid.theta[t];
                m2 += post * grid.theta[t] * grid.theta[t];
                for d in 0..grid.latent_dim {
                    xi_eap[person * grid.latent_dim + d] += post * grid.xi[x * grid.latent_dim + d];
                }
            }
        }
        theta_eap[person] = m1;
        theta_sd[person] = (m2 - m1 * m1).max(0.0).sqrt();
    }
    (theta_eap, theta_sd, xi_eap)
}

#[allow(clippy::too_many_arguments)]
pub fn fit_mixed_items(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    specs: &[MixedItemSpec],
    latent_dim: usize,
    q_theta: usize,
    q_xi: usize,
    max_iter: usize,
    tol: f64,
    requested_threads: usize,
) -> Result<MixedFit, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("responses must contain at least one person and one item".into());
    }
    let expected_len = n_persons
        .checked_mul(n_items)
        .ok_or("n_persons * n_items overflow")?;
    if y.len() != expected_len {
        return Err("y must have length n_persons * n_items".into());
    }
    if specs.len() != n_items {
        return Err("item specification count must match n_items".into());
    }
    if let Some(mask) = observed {
        if mask.len() != y.len() {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return Err("max_iter must be positive and tol must be finite and > 0".into());
    }
    for (item, spec) in specs.iter().enumerate() {
        if spec.n_categories < 2 {
            return Err(format!("item {item}: n_categories must be >= 2"));
        }
        if matches!(
            spec.kind,
            MixedItemKind::TwoPl | MixedItemKind::Ideal | MixedItemKind::Lsirm
        ) && spec.n_categories != 2
        {
            return Err(format!(
                "item {item}: {} requires exactly 2 categories",
                spec.kind.as_str()
            ));
        }
        let mut seen = vec![false; spec.n_categories];
        for person in 0..n_persons {
            let index = person * n_items + item;
            if observed.map_or(true, |m| m[index]) {
                let response = y[index];
                if response >= spec.n_categories {
                    return Err(format!(
                        "item {item}: observed response {response} is outside 0..{}",
                        spec.n_categories - 1
                    ));
                }
                seen[response] = true;
            }
        }
        if seen.iter().filter(|&&present| present).count() < 2 {
            return Err(format!(
                "item {item}: at least two observed categories are required"
            ));
        }
    }
    let observed_owned;
    let observed = if let Some(mask) = observed {
        mask
    } else {
        observed_owned = vec![true; y.len()];
        &observed_owned
    };
    let grid = build_grid(specs, latent_dim, q_theta, q_xi)?;
    let auto_threads = thread::available_parallelism().map_or(1, |n| n.get());
    let n_threads = if requested_threads == 0 {
        auto_threads
    } else {
        requested_threads.min(auto_threads)
    }
    .clamp(1, n_persons.max(1));

    let mut params = Vec::with_capacity(n_items);
    for item in 0..n_items {
        let mut freq = vec![1e-3; specs[item].n_categories];
        for person in 0..n_persons {
            let index = person * n_items + item;
            if observed[index] {
                freq[y[index]] += 1.0;
            }
        }
        let total: f64 = freq.iter().sum();
        for value in &mut freq {
            *value /= total;
        }
        params.push(initial_params(
            &specs[item],
            &freq,
            item,
            n_items,
            grid.latent_dim,
        ));
    }

    let mut tables = build_tables(specs, &params, &grid);
    let mut state = e_step(
        y, observed, n_persons, n_items, specs, &tables, &grid, n_threads,
    );
    if !state.loglik.is_finite() {
        return Err("initial mixed-format log-likelihood is not finite".into());
    }
    let mut trace = vec![state.loglik];
    let mut converged = false;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut completed = 0;
    for iteration in 1..=max_iter {
        let candidate = m_step(specs, &params, &grid, &state.counts, n_threads);
        let candidate_tables = build_tables(specs, &candidate, &grid);
        let candidate_state = e_step(
            y,
            observed,
            n_persons,
            n_items,
            specs,
            &candidate_tables,
            &grid,
            n_threads,
        );
        if !candidate_state.loglik.is_finite() {
            termination_reason = "non_finite_loglik".to_string();
            break;
        }
        let change = candidate_state.loglik - state.loglik;
        let monotone_slack = 1e-8 * (1.0 + state.loglik.abs());
        if change < -monotone_slack {
            termination_reason = "non_monotone_update".to_string();
            break;
        }
        params = candidate;
        tables = candidate_tables;
        state = candidate_state;
        trace.push(state.loglik);
        completed = iteration;
        if change.abs() <= tol * (1.0 + state.loglik.abs()) {
            converged = true;
            termination_reason = "converged".to_string();
            break;
        }
    }

    let (theta_eap, theta_sd, xi_eap) =
        final_scores(y, observed, n_persons, n_items, specs, &tables, &grid);
    let items = specs
        .iter()
        .zip(&params)
        .map(|(spec, p)| public_estimate(spec, p, grid.latent_dim))
        .collect();
    Ok(MixedFit {
        items,
        theta_eap,
        theta_sd,
        xi_eap,
        latent_dim: grid.latent_dim,
        loglik: state.loglik,
        loglik_trace: trace,
        n_iter: completed,
        converged,
        termination_reason,
        n_threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mixed_cell_normalizes() {
        let cases = [
            (MixedItemKind::TwoPl, 2),
            (MixedItemKind::Grm, 4),
            (MixedItemKind::Gpcm, 4),
            (MixedItemKind::Nominal, 4),
            (MixedItemKind::Ideal, 2),
            (MixedItemKind::Ggum, 4),
            (MixedItemKind::Lsirm, 2),
            (MixedItemKind::LsirmGrm, 4),
            (MixedItemKind::LsirmGpcm, 4),
        ];
        for (kind, n_categories) in cases {
            let spec = MixedItemSpec { kind, n_categories };
            let latent_dim = if kind.is_spatial() { 2 } else { 0 };
            let freq = vec![1.0 / n_categories as f64; n_categories];
            let params = initial_params(&spec, &freq, 0, 1, latent_dim);
            for theta in [-4.0, 0.0, 4.0] {
                let xi = if latent_dim == 0 {
                    &[][..]
                } else {
                    &[0.3, -0.2][..]
                };
                let lp = item_logprobs(&spec, &params, theta, xi, latent_dim);
                assert_eq!(lp.len(), n_categories);
                assert!(lp.iter().all(|v| v.is_finite()), "{kind:?}: {lp:?}");
                let total: f64 = lp.iter().map(|v| v.exp()).sum();
                assert!((total - 1.0).abs() < 1e-10, "{kind:?}: {total}");
            }
        }
    }

    #[test]
    fn binary_cells_match_their_defining_formulas() {
        let theta = 0.4;
        let two = MixedItemSpec {
            kind: MixedItemKind::TwoPl,
            n_categories: 2,
        };
        let lp = item_logprobs(&two, &[1.2_f64.ln(), -0.3], theta, &[], 0);
        let expected = 1.0 / (1.0 + (-(1.2 * theta - 0.3)).exp());
        assert!((lp[1].exp() - expected).abs() < 1e-12);

        let ideal = MixedItemSpec {
            kind: MixedItemKind::Ideal,
            n_categories: 2,
        };
        let lp = item_logprobs(&ideal, &[1.5_f64.ln(), -0.2], theta, &[], 0);
        let expected = (-0.5 * (1.5 * (theta + 0.2)).powi(2)).exp();
        assert!((lp[1].exp() - expected).abs() < 1e-12);
    }

    #[test]
    fn rejects_hidden_nonconvergence_as_success() {
        let y = vec![0, 0, 1, 1, 0, 1, 1, 0];
        let specs = vec![
            MixedItemSpec {
                kind: MixedItemKind::TwoPl,
                n_categories: 2,
            },
            MixedItemSpec {
                kind: MixedItemKind::TwoPl,
                n_categories: 2,
            },
        ];
        let fit = fit_mixed_items(&y, None, 4, 2, &specs, 1, 7, 7, 1, 1e-14, 1).unwrap();
        assert!(!fit.converged);
        assert_eq!(fit.termination_reason, "max_iter_reached");
        assert_eq!(fit.n_iter, 1);
        assert_eq!(fit.loglik_trace.len(), 2);
    }
}
