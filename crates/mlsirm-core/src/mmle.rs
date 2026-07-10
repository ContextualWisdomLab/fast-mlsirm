//! MMLE (marginal maximum likelihood) via EM for a unidimensional 2PL model.
//!
//! Penalized JMLE estimates each person's ability jointly with item parameters,
//! which is weakly identified and biased under missing/sparse data. MMLE
//! integrates ability out over a population distribution (Gauss-Hermite
//! quadrature), so each person contributes a product over their **observed**
//! items only — missingness (MAR) is handled by construction, no imputation.
//!
//! Scope: unidimensional 2PL, logit = a*theta + b, ability prior N(0, 1).
//! Item parameters (a, b) by EM; ability returned as EAP posterior mean.

/// 21-node probabilists' Gauss-Hermite rule, weights normalized to sum 1
/// (quadrature for a standard-normal N(0,1) ability prior).
const GH_NODES: [f64; 21] = [
    -7.849_382_90e0, -6.751_444_72e0, -5.829_382_01e0, -4.994_963_94e0, -4.214_343_98e0,
    -3.469_846_69e0, -2.750_592_98e0, -2.049_102_47e0, -1.359_765_82e0, -6.780_456_92e-1,
    0.0, 6.780_456_92e-1, 1.359_765_82e0, 2.049_102_47e0, 2.750_592_98e0, 3.469_846_69e0,
    4.214_343_98e0, 4.994_963_94e0, 5.829_382_01e0, 6.751_444_72e0, 7.849_382_90e0,
];
const GH_WEIGHTS: [f64; 21] = [
    2.098_991_22e-14, 4.975_368_60e-11, 1.450_661_28e-8, 1.225_354_84e-6, 4.219_234_74e-5,
    7.080_477_95e-4, 6.439_697_05e-3, 3.395_272_98e-2, 1.083_922_86e-1, 2.153_337_16e-1,
    2.702_601_84e-1, 2.153_337_16e-1, 1.083_922_86e-1, 3.395_272_98e-2, 6.439_697_05e-3,
    7.080_477_95e-4, 4.219_234_74e-5, 1.225_354_84e-6, 1.450_661_28e-8, 4.975_368_60e-11,
    2.098_991_22e-14,
];

#[derive(Clone, Debug)]
pub struct MmleResult {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MmleConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub ridge_a: f64,
    pub ridge_b: f64,
    pub newton_iter: usize,
}

impl Default for MmleConfig {
    fn default() -> Self {
        Self { max_iter: 500, tol: 1e-6, ridge_a: 1e-3, ridge_b: 1e-3, newton_iter: 25 }
    }
}

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[inline]
fn sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Calibrate a unidimensional 2PL by MMLE-EM under missing data.
/// `y` and `observed` are row-major `n_persons * n_items`. Missing cells (where
/// `observed[idx] == false`) are ignored.
pub fn fit_mmle_2pl(
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    cfg: &MmleConfig,
) -> MmleResult {
    assert_eq!(y.len(), n_persons * n_items);
    assert_eq!(observed.len(), n_persons * n_items);
    let q = GH_NODES.len();
    let log_w: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();

    let mut a = vec![1.0_f64; n_items];
    let mut b = vec![0.0_f64; n_items];
    for i in 0..n_items {
        let mut num = 0.0;
        let mut den = 0.0;
        for p in 0..n_persons {
            let idx = p * n_items + i;
            if observed[idx] {
                num += y[idx];
                den += 1.0;
            }
        }
        let prop = if den > 0.0 { (num / den).clamp(0.02, 0.98) } else { 0.5 };
        b[i] = (prop / (1.0 - prop)).ln();
    }

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut posterior = vec![0.0_f64; n_persons * q];
    let mut converged = false;

    for iteration in 0..cfg.max_iter {
        let mut log_p1 = vec![0.0_f64; q * n_items];
        let mut log_p0 = vec![0.0_f64; q * n_items];
        for (qi, &node) in GH_NODES.iter().enumerate() {
            for i in 0..n_items {
                let eta = a[i] * node + b[i];
                log_p1[qi * n_items + i] = log_sigmoid(eta);
                log_p0[qi * n_items + i] = log_sigmoid(-eta);
            }
        }

        let mut total_loglik = 0.0;
        for p in 0..n_persons {
            let mut log_joint = vec![0.0_f64; q];
            for (qi, item) in log_joint.iter_mut().enumerate() {
                let mut acc = log_w[qi];
                for i in 0..n_items {
                    let idx = p * n_items + i;
                    if observed[idx] {
                        let yy = y[idx];
                        acc += yy * log_p1[qi * n_items + i] + (1.0 - yy) * log_p0[qi * n_items + i];
                    }
                }
                *item = acc;
            }
            let max_lj = log_joint.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0;
            for &lj in &log_joint {
                denom += (lj - max_lj).exp();
            }
            total_loglik += max_lj + denom.ln();
            for qi in 0..q {
                posterior[p * q + qi] = (log_joint[qi] - max_lj).exp() / denom;
            }
        }
        loglik_trace.push(total_loglik);

        let mut n_iq = vec![0.0_f64; n_items * q];
        let mut r_iq = vec![0.0_f64; n_items * q];
        for p in 0..n_persons {
            for i in 0..n_items {
                let idx = p * n_items + i;
                if observed[idx] {
                    let yy = y[idx];
                    for qi in 0..q {
                        let post = posterior[p * q + qi];
                        n_iq[i * q + qi] += post;
                        r_iq[i * q + qi] += yy * post;
                    }
                }
            }
        }

        for i in 0..n_items {
            let (mut ai, mut bi) = (a[i], b[i]);
            for _ in 0..cfg.newton_iter {
                let (mut g_a, mut g_b, mut h_aa, mut h_bb, mut h_ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for (qi, &node) in GH_NODES.iter().enumerate() {
                    let p_correct = sigmoid_stable(ai * node + bi);
                    let n = n_iq[i * q + qi];
                    let w = n * p_correct * (1.0 - p_correct);
                    let resid = r_iq[i * q + qi] - n * p_correct;
                    g_a += resid * node;
                    g_b += resid;
                    h_aa -= w * node * node;
                    h_bb -= w;
                    h_ab -= w * node;
                }
                g_a -= cfg.ridge_a * ai;
                g_b -= cfg.ridge_b * bi;
                h_aa -= cfg.ridge_a;
                h_bb -= cfg.ridge_b;
                let det = h_aa * h_bb - h_ab * h_ab;
                if det.abs() < 1e-12 {
                    break;
                }
                let da = (h_bb * g_a - h_ab * g_b) / det;
                let db = (h_aa * g_b - h_ab * g_a) / det;
                ai = (ai - da).clamp(1e-3, 10.0);
                bi -= db;
                if da.abs() + db.abs() < 1e-8 {
                    break;
                }
            }
            a[i] = ai;
            b[i] = bi;
        }

        if iteration > 0 {
            let delta = (loglik_trace[iteration] - loglik_trace[iteration - 1]).abs();
            if delta < cfg.tol {
                converged = true;
                break;
            }
        }
    }

    let mut theta = vec![0.0_f64; n_persons];
    for p in 0..n_persons {
        let mut m = 0.0;
        for (qi, &node) in GH_NODES.iter().enumerate() {
            m += posterior[p * q + qi] * node;
        }
        theta[p] = m;
    }

    let n_iter = loglik_trace.len();
    MmleResult { a, b, theta, loglik_trace, n_iter, converged }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    #[test]
    fn recovers_2pl_under_30pct_missing() {
        let mut rng = Lcg(12345);
        let (n_persons, n_items) = (800usize, 20usize);
        let a_true: Vec<f64> = (0..n_items).map(|_| 0.7 + 1.3 * rng.next_f64()).collect();
        let b_true: Vec<f64> = (0..n_items).map(|_| -1.5 + 3.0 * rng.next_f64()).collect();
        let theta_true: Vec<f64> = (0..n_persons).map(|_| rng.normal()).collect();

        let mut y = vec![0.0_f64; n_persons * n_items];
        let mut observed = vec![true; n_persons * n_items];
        for p in 0..n_persons {
            for i in 0..n_items {
                let idx = p * n_items + i;
                let eta = a_true[i] * theta_true[p] + b_true[i];
                let prob = 1.0 / (1.0 + (-eta).exp());
                y[idx] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
                if rng.next_f64() < 0.30 {
                    observed[idx] = false;
                }
            }
        }

        let res = fit_mmle_2pl(&y, &observed, n_persons, n_items, &MmleConfig::default());
        assert!(res.converged, "EM should converge");
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "loglik decreased: {} -> {}", w[0], w[1]);
        }
        assert!(corr(&res.a, &a_true) > 0.85, "a recovery too low");
        assert!(corr(&res.b, &b_true) > 0.9, "b recovery too low");
        assert!(corr(&res.theta, &theta_true) > 0.8, "theta recovery too low");
    }

    #[test]
    fn all_missing_person_row_is_tolerated() {
        let (n_persons, n_items) = (3usize, 4usize);
        let y = vec![1.0; n_persons * n_items];
        let mut observed = vec![true; n_persons * n_items];
        for i in 0..n_items {
            observed[i] = false;
        }
        let res = fit_mmle_2pl(&y, &observed, n_persons, n_items, &MmleConfig::default());
        assert!(res.theta.iter().all(|t| t.is_finite()));
        assert!(res.theta[0].abs() < 1e-6, "all-missing person should shrink to prior mean 0");
    }

    #[test]
    fn newton_tolerates_singular_hessian_without_ridge() {
        // An item that nobody observed carries zero Fisher information. With the
        // ridge disabled the per-item Newton Hessian is exactly singular, so the
        // solver must hit the `det.abs() < 1e-12` guard and break out of the
        // Newton loop instead of dividing by (near-)zero. This exercises the
        // singular-Hessian branch in fit_mmle_2pl.
        let (n_persons, n_items) = (6usize, 3usize);
        let mut y = vec![0.0_f64; n_persons * n_items];
        let mut observed = vec![true; n_persons * n_items];
        for p in 0..n_persons {
            // Items 0 and 1 carry a varied, informative response pattern.
            y[p * n_items] = (p % 2) as f64;
            y[p * n_items + 1] = ((p / 2) % 2) as f64;
            // Item 2 is never observed -> zero information for its Newton step.
            observed[p * n_items + 2] = false;
        }
        let cfg =
            MmleConfig { ridge_a: 0.0, ridge_b: 0.0, max_iter: 50, ..MmleConfig::default() };
        let res = fit_mmle_2pl(&y, &observed, n_persons, n_items, &cfg);

        assert!(res.a.iter().all(|v| v.is_finite()), "item slopes must stay finite");
        assert!(res.b.iter().all(|v| v.is_finite()), "item intercepts must stay finite");
        assert!(res.theta.iter().all(|t| t.is_finite()), "abilities must stay finite");
        // The zero-information item keeps its initial (a = 1, b = 0) because the
        // Newton step breaks on the singular Hessian before any update applies.
        assert_eq!(res.a[2], 1.0, "unobserved item slope must stay at its initial value");
        assert_eq!(res.b[2], 0.0, "unobserved item intercept must stay at its initial value");
    }
}
