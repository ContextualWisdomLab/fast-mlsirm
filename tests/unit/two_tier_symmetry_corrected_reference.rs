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
