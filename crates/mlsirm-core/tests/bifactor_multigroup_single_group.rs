//! Single-group equivalence: the multiple-group bifactor fitter with one group
//! reduces to the stage-1 single-group fitter exactly.
//!
//! With `n_groups == 1` the reference distribution `N(0, I)` is the only
//! population, the node-shift reparameterization
//! `theta_{g,t} = mu_g + sigma_g x_t` (Bock & Zimowski, 1997; Cai, Yang, &
//! Hansen, 2011) is the identity, and the pooled M-step is the single-group
//! M-step, so the same `(seed, n_starts, ...)` run must bit-reproduce the
//! stage-1 estimates, EAPs, and loglik trace. This test fails if the two paths
//! drift apart.
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm, fit_bifactor_grm_multigroup, BifactorGrmConfig, BifactorMultigroupConfig,
};

const N_PERSONS: usize = 300;
const N_ITEMS: usize = 6;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 1, 1, 1];
const TRUE_A_G: [f64; N_ITEMS] = [1.40, -1.10, 1.00, 1.20, 0.90, 1.10];
const TRUE_A_S: [f64; N_ITEMS] = [1.00, 0.90, 1.10, 1.00, 0.80, 0.90];
const TRUE_D: [[f64; 3]; N_ITEMS] = [
    [1.20, 0.00, -1.20],
    [1.00, -0.10, -1.30],
    [1.30, 0.20, -1.00],
    [1.10, 0.10, -1.10],
    [0.90, -0.20, -1.40],
    [1.20, 0.00, -1.20],
];

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((self.0 >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn single_group_multigroup_matches_stage1_exactly() {
    let mut rng = Lcg(20_260_917);
    let mut y = vec![0usize; N_PERSONS * N_ITEMS];
    for p in 0..N_PERSONS {
        let t_g = rng.standard_normal();
        let mut t_s = [0.0f64; N_SPECIFIC];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let s = SPECIFIC_MAP[i] as usize;
            let base = TRUE_A_G[i] * t_g + TRUE_A_S[i] * t_s[s];
            let mut cum = [0.0f64; 3];
            for k in 0..3 {
                cum[k] = sigmoid(base + TRUE_D[i][k]);
            }
            let probs = [1.0 - cum[0], cum[0] - cum[1], cum[1] - cum[2], cum[2]];
            let mut draw = rng.uniform();
            let mut cat = N_CAT - 1;
            for (k, pr) in probs.iter().enumerate() {
                if draw < *pr {
                    cat = k;
                    break;
                }
                draw -= *pr;
            }
            y[p * N_ITEMS + i] = cat;
        }
    }

    let single_cfg = BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 200,
        tol: 1e-5,
        n_starts: 2,
        seed: 20_260_916,
        newton_iter: 10,
        ridge: 1e-8,
        device: mlsirm_core::Device::Cpu,
        e_step_n_chunks: 1,
        e_step_n_threads: 1,
    };
    let single = fit_bifactor_grm(
        &y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &single_cfg,
    )
    .expect("single-group fit must succeed");

    let multi_cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 200,
        tol: 1e-5,
        n_starts: 2,
        seed: 20_260_916,
        newton_iter: 10,
        ridge: 1e-8,
        device: mlsirm_core::Device::Cpu,
            e_step_n_chunks: 1,
            e_step_n_threads: 1,
        estimate_specific_vars: false,
    };
    let group_id = vec![0usize; N_PERSONS];
    let multi = fit_bifactor_grm_multigroup(
        &y,
        None,
        &group_id,
        1,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        None,
        &multi_cfg,
    )
    .expect("single-group multigroup fit must succeed");

    assert_eq!(multi.a_general.len(), 1);
    for (a, b) in single.a_general.iter().zip(multi.a_general[0].iter()) {
        assert!(
            a.to_bits() == b.to_bits(),
            "a_general must bit-reproduce stage 1: {a} vs {b}"
        );
    }
    for (a, b) in single.a_specific.iter().zip(multi.a_specific[0].iter()) {
        assert!(
            a.to_bits() == b.to_bits(),
            "a_specific must bit-reproduce stage 1: {a} vs {b}"
        );
    }
    for (a, b) in single.threshold.iter().zip(multi.threshold[0].iter()) {
        assert!(
            a.to_bits() == b.to_bits(),
            "threshold must bit-reproduce stage 1: {a} vs {b}"
        );
    }
    for (a, b) in single.theta_g_eap.iter().zip(multi.theta_g_eap.iter()) {
        assert!(
            a.to_bits() == b.to_bits(),
            "theta EAP must bit-reproduce stage 1: {a} vs {b}"
        );
    }
    assert_eq!(single.loglik_trace.len(), multi.loglik_trace.len());
    for (a, b) in single.loglik_trace.iter().zip(multi.loglik_trace.iter()) {
        assert!(
            a.to_bits() == b.to_bits(),
            "loglik trace must bit-reproduce stage 1: {a} vs {b}"
        );
    }
    assert_eq!(single.converged, multi.converged);
    assert_eq!(single.n_iter, multi.n_iter);
    assert_eq!(single.best_start, multi.best_start);
}

#[test]
fn uncapped_start_budget_is_accepted() {
    // No magic upper caps on caller budgets (stage-1 review fix-up rule):
    // `n_starts = 40` validates and indexes its 40 runs.
    let mut rng = Lcg(20_260_921);
    let mut y = vec![0usize; 60 * N_ITEMS];
    for p in 0..60 {
        let t_g = rng.standard_normal();
        let mut t_s = [0.0f64; N_SPECIFIC];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let s = SPECIFIC_MAP[i] as usize;
            let base = TRUE_A_G[i] * t_g + TRUE_A_S[i] * t_s[s];
            let mut cum = [0.0f64; 3];
            for k in 0..3 {
                cum[k] = sigmoid(base + TRUE_D[i][k]);
            }
            let probs = [1.0 - cum[0], cum[0] - cum[1], cum[1] - cum[2], cum[2]];
            let mut draw = rng.uniform();
            let mut cat = N_CAT - 1;
            for (k, pr) in probs.iter().enumerate() {
                if draw < *pr {
                    cat = k;
                    break;
                }
                draw -= *pr;
            }
            y[p * N_ITEMS + i] = cat;
        }
    }
    // Nudge any missing category into the data so the fit validates.
    for i in 0..N_ITEMS {
        for k in 0..N_CAT {
            if !(0..60).any(|p| y[p * N_ITEMS + i] == k) {
                y[i] = k;
            }
        }
    }
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 1,
        tol: 1e-12,
        n_starts: 40,
        seed: 20_260_922,
        newton_iter: 10,
        ridge: 1e-8,
        device: mlsirm_core::Device::Cpu,
            e_step_n_chunks: 1,
            e_step_n_threads: 1,
        estimate_specific_vars: false,
    };
    let group_id = vec![0usize; 60];
    let fit = fit_bifactor_grm_multigroup(
        &y,
        None,
        &group_id,
        1,
        &SPECIFIC_MAP,
        60,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        None,
        &cfg,
    )
    .expect("n_starts=40 must be accepted (no upper cap)");
    assert!(
        fit.best_start < 40,
        "best_start must index the 40 runs; got {}",
        fit.best_start
    );
}
