//! Independent symmetry-corrected reference for retained Newton oracles.
//! Basis: Higham (1989, reprint pp. 3–4, §2) defines (H + Hᵀ)/2;
//! Cai (2010, pp. 608–609, Appendix A) describes the conditional M-step.
//! References: Higham, N. J. (1989). Matrix nearness problems and applications.
//! In M. J. C. Gover & S. Barnett (Eds.), Applications of matrix theory
//! (pp. 1–27). Oxford University Press.
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. Psychometrika, 75(4), 581–612.
//! doi:10.1007/s11336-010-9178-0.
use super::*;

pub(super) fn symmetry_corrected_newton<F>(
    mut params: Vec<f64>,
    ridge: f64,
    n_newton: usize,
    objective: F,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g) = objective(&params);
        let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = objective(&pj);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        // Read both entries from a separate immutable matrix, independently
        // of the production in-place unordered-pair implementation.
        let raw = hess.clone();
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (raw[r][c] + raw[c][r]);
            }
            hess[r][r] += ridge;
        }
        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }
        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
            max_step = 2.0;
        }
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _) = objective(&candidate);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    params
}

/// Test-only same-count analytic assembly, separate from production Newton.
/// Chains the shared GRM node derivatives through the explicit item design,
/// preserving each count row and multiplication/accumulation order. The FD
/// oracle above remains historical; main analytic-versus-FD tests are unchanged.
/// Basis: the current main's item derivative contract and its existing cited
/// Gibbons et al. (2007), Equation 9 and Appendix A4–A6 implementation.
#[allow(clippy::too_many_arguments)]
pub(super) fn item_analytic_reference(
    params: &[f64], free: &[usize], specific: bool,
    coords: &[f64], ts: &[f64], n_primary: usize, n_grid: usize,
    qs: usize, counts: &[Vec<f64>], _n_cat: usize,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let slopes = free.len() + usize::from(specific);
    let thresholds = &params[slopes..];
    let mut objective = 0.0;
    let mut gradient = vec![0.0; params.len()];
    let mut curvature = vec![vec![0.0; params.len()]; params.len()];
    for (ordinal, row) in counts.iter().enumerate() {
        let (g, h) = if specific { (ordinal / qs, ordinal % qs) }
                     else { assert!(ordinal < n_grid); (ordinal, 0) };
        let mut design: Vec<f64> = free.iter().map(|&d| coords[g * n_primary + d]).collect();
        if specific { design.push(ts[h]); }
        let mut predictor = 0.0;
        for (x, a) in design.iter().zip(params) { predictor += a * x; }
        let cells = crate::poly::grm_logprobs(predictor, thresholds);
        objective += row.iter().zip(&cells).map(|(r, l)| r * l).sum::<f64>();
        let (base_grad, threshold_grad) = crate::poly::grm_node_gradient(predictor, thresholds, row);
        for (slot, x) in gradient.iter_mut().zip(&design) { *slot += base_grad * x; }
        for (j, value) in threshold_grad.iter().enumerate() { gradient[slopes + j] += value; }
        let (base_hess, cross, threshold_hess) = crate::poly::grm_node_hessian(predictor, thresholds, row);
        for t in 0..slopes {
            for u in 0..slopes { curvature[t][u] += design[t] * design[u] * base_hess; }
        }
        for (j, value) in cross.iter().enumerate() {
            let index = slopes + j;
            for t in 0..slopes {
                let pair = design[t] * value;
                curvature[t][index] += pair;
                curvature[index][t] += pair;
            }
            for (l, value) in threshold_hess[j].iter().enumerate() { curvature[index][slopes + l] += value; }
        }
    }
    for row in &mut curvature { for value in row { *value = -*value; } }
    (-objective, gradient.into_iter().map(|value| -value).collect(), curvature)
}

pub(super) fn analytic_reference_newton<F>(
    mut params: Vec<f64>,
    ridge: f64,
    n_newton: usize,
    objective: F,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>),
{
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g, mut hess) = objective(&params);
        let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        // Read both entries from a separate immutable matrix, independently
        // of the production in-place unordered-pair implementation.
        let raw = hess.clone();
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (raw[r][c] + raw[c][r]);
            }
            hess[r][r] += ridge;
        }
        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }
        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
            max_step = 2.0;
        }
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _, _) = objective(&candidate);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    params
}
