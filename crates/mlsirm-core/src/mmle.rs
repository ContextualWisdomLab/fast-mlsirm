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

/// 41-node probabilists' Gauss-Hermite rule, weights normalized to sum 1
/// (quadrature for a standard-normal N(0,1) ability prior). Values are the
/// shortest-roundtrip f64 output of `numpy.polynomial.hermite_e.hermegauss(41)`
/// (weights divided by their sum), so this table is bit-identical to the
/// default quadrature of the NumPy reference in
/// `python/fast_mlsirm/estimators/mmle.py` — the Rust<->NumPy parity contract.
pub(crate) const GH_NODES: [f64; 41] = [
    -11.614937254337464,
    -10.647536786319334,
    -9.843433249157995,
    -9.123069907984473,
    -8.45609908326939,
    -7.82688200405387,
    -7.226022663732788,
    -6.647308470747189,
    -6.0863491648784755,
    -5.539884440458124,
    -5.0053966834041255,
    -4.480878331594007,
    -3.9646840280332665,
    -3.4554322177809933,
    -2.9519370163811907,
    -2.453159345907048,
    -1.9581707119772913,
    -1.4661254572959665,
    -0.9762387671800493,
    -0.4877685693194346,
    0.0,
    0.4877685693194346,
    0.9762387671800493,
    1.4661254572959665,
    1.9581707119772913,
    2.453159345907048,
    2.9519370163811907,
    3.4554322177809933,
    3.9646840280332665,
    4.480878331594007,
    5.0053966834041255,
    5.539884440458124,
    6.0863491648784755,
    6.647308470747189,
    7.226022663732788,
    7.82688200405387,
    8.45609908326939,
    9.123069907984473,
    9.843433249157995,
    10.647536786319334,
    11.614937254337464,
];
pub(crate) const GH_WEIGHTS: [f64; 41] = [
    2.2578639565831077e-30,
    8.308558938782659e-26,
    2.7468912285223205e-22,
    2.3263841455871947e-19,
    7.655982291966907e-17,
    1.2203348742027809e-14,
    1.0778183949358929e-12,
    5.7698534280921236e-11,
    1.994794756757345e-9,
    4.66734770810732e-8,
    7.658186077982326e-7,
    9.058608622432971e-6,
    7.89471931950462e-5,
    0.000515801444343186,
    0.002561642428649783,
    0.009777902738208262,
    0.028937211747934403,
    0.06684765935446638,
    0.12114891701151059,
    0.17284953105060138,
    0.19454502775360044,
    0.17284953105060138,
    0.12114891701151059,
    0.06684765935446638,
    0.028937211747934403,
    0.009777902738208262,
    0.002561642428649783,
    0.000515801444343186,
    7.89471931950462e-5,
    9.058608622432971e-6,
    7.658186077982326e-7,
    4.66734770810732e-8,
    1.994794756757345e-9,
    5.7698534280921236e-11,
    1.0778183949358929e-12,
    1.2203348742027809e-14,
    7.655982291966907e-17,
    2.3263841455871947e-19,
    2.7468912285223205e-22,
    8.308558938782659e-26,
    2.2578639565831077e-30,
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
        Self {
            max_iter: 500,
            tol: 1e-6,
            ridge_a: 1e-3,
            ridge_b: 1e-3,
            newton_iter: 25,
        }
    }
}

#[inline]
pub(crate) fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[inline]
pub(crate) fn sigmoid_stable(x: f64) -> f64 {
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
        let prop = if den > 0.0 {
            (num / den).clamp(0.02, 0.98)
        } else {
            0.5
        };
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
                        acc +=
                            yy * log_p1[qi * n_items + i] + (1.0 - yy) * log_p0[qi * n_items + i];
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
    MmleResult {
        a,
        b,
        theta,
        loglik_trace,
        n_iter,
        converged,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/mmle_tests.rs"]
mod tests;
