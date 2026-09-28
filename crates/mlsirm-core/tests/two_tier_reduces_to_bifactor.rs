//! Two-tier GRM reduces exactly to the stage-1 bifactor GRM at `P = 1`
//! (stage 4 of #1912).
//!
//! The two-tier model with a single primary dimension IS the bifactor model
//! (mirt `bfactor` documentation: "The bifactor model is a special case of
//! the two-tier model when G above is a 1x1 matrix"). This test fits the same
//! simulated bifactor dataset once with each fitter (same quadrature
//! density, tolerance, and start-0 initialization) and asserts the two MLEs
//! coincide: primary slopes against general slopes, specific slopes,
//! boundary intercepts, and the observed-data log-likelihood.
//!
//! Tolerance basis: MEASURED bitwise equality on the committed design —
//! the GREEN run gave `worst_slope = 0`, `worst_intercept = 0`,
//! `ll_gap = 0` with both fitters taking 42 EM iterations from the same
//! start. The asserted bands (1e-9) exist only for cross-platform libm
//! noise (`exp`/`ln` need not be bitwise identical across targets); any gap
//! above 1e-9 is an implementation divergence, not quadrature error.
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
//! accessible — no equation locator is drawn from it)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use mlsirm_core::bifactor_grm::{fit_bifactor_grm, BifactorGrmConfig};
use mlsirm_core::two_tier_grm::{fit_two_tier_grm, TwoTierGrmConfig};

const N_PERSONS: usize = 500;
const N_ITEMS: usize = 6;
const N_CAT: usize = 4;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 1, 1, 1];

const TRUE_A_GENERAL: [f64; N_ITEMS] = [1.5, -1.1, 1.0, 1.3, 0.9, 1.2];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] = [1.0, 0.9, 1.1, 1.0, 0.8, 0.9];
const TRUE_D: [[f64; 3]; N_ITEMS] = [
    [1.2, 0.0, -1.2],
    [1.0, -0.1, -1.3],
    [1.3, 0.2, -1.0],
    [1.1, 0.1, -1.1],
    [0.9, -0.2, -1.4],
    [1.2, 0.0, -1.2],
];

// Bands from the measured bitwise-equality run (see module docs); nonzero
// only for cross-platform libm noise.
const MAX_SLOPE_GAP: f64 = 1e-9;
const MAX_INTERCEPT_GAP: f64 = 1e-9;
const MAX_LOGLIK_GAP: f64 = 1e-6;

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

fn simulate() -> Vec<usize> {
    let mut rng = Lcg(4_191_204);
    let mut y = vec![0usize; N_PERSONS * N_ITEMS];
    for p in 0..N_PERSONS {
        let t_g = rng.standard_normal();
        let mut t_s = [0.0f64; 2];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let base = TRUE_A_GENERAL[i] * t_g + TRUE_A_SPECIFIC[i] * t_s[SPECIFIC_MAP[i] as usize];
            let mut ge = [1.0f64; 5];
            for k in 1..N_CAT {
                ge[k] = 1.0 / (1.0 + (-(base + TRUE_D[i][k - 1])).exp());
            }
            ge[N_CAT] = 0.0;
            let mut draw = rng.uniform();
            let mut cat = N_CAT - 1;
            for k in 0..N_CAT {
                let pr = ge[k] - ge[k + 1];
                if draw < pr {
                    cat = k;
                    break;
                }
                draw -= pr;
            }
            y[p * N_ITEMS + i] = cat;
        }
    }
    y
}

#[test]
fn two_tier_with_single_primary_matches_bifactor_fit() {
    let y = simulate();
    let stage1 = fit_bifactor_grm(
        &y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        2,
        N_CAT,
        &BifactorGrmConfig {
            q_general: 15,
            q_specific: 11,
            max_iter: 1000,
            tol: 1e-7,
            n_starts: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
            newton_iter: 10,
            ridge: 1e-8,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("stage-1 fit must succeed");
    assert!(
        stage1.converged,
        "stage-1 fit must converge ({}); test design issue, not estimator drift",
        stage1.termination_reason
    );
    let primary_map = vec![true; N_ITEMS];
    let two_tier = fit_two_tier_grm(
        &y,
        None,
        &primary_map,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        1,
        2,
        N_CAT,
        &TwoTierGrmConfig {
            estimate_primary_correlation: false,
            q_primary: 15,
            q_specific: 11,
            max_iter: 1000,
            tol: 1e-7,
            n_starts: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
            newton_iter: 10,
            ridge: 1e-8,
        },
    )
    .expect("two-tier P=1 fit must succeed");
    assert!(
        two_tier.converged,
        "two-tier P=1 fit must converge ({}); the P=1 model IS the bifactor model",
        two_tier.termination_reason
    );
    assert_eq!(two_tier.primary_identification, "orthogonal");

    let mut worst_slope = 0.0f64;
    for i in 0..N_ITEMS {
        worst_slope = worst_slope
            .max((two_tier.a_primary[i] - stage1.a_general[i]).abs())
            .max((two_tier.a_specific[i] - stage1.a_specific[i]).abs());
    }
    let mut worst_intercept = 0.0f64;
    for (i, v) in stage1.threshold.iter().enumerate() {
        worst_intercept = worst_intercept.max((two_tier.threshold[i] - v).abs());
    }
    let ll1 = *stage1.loglik_trace.last().expect("trace is never empty");
    let ll2 = *two_tier.loglik_trace.last().expect("trace is never empty");
    let ll_gap = (ll1 - ll2).abs();
    // The 1x1 primary correlation is identically 1 by definition.
    // The 1x1 primary correlation is identically 1 by definition.
    assert!(
        (two_tier.phi[0] - 1.0).abs() <= 1e-12,
        "P=1 phi must be exactly [1.0]; got {:?}",
        two_tier.phi
    );
    assert!(
        worst_slope <= MAX_SLOPE_GAP,
        "P=1 two-tier slopes must match stage-1 (gap {worst_slope:.3e}):\n two-tier {:?}\n stage-1 {:?} / {:?}",
        two_tier.a_primary,
        stage1.a_general,
        stage1.a_specific
    );
    assert!(
        worst_intercept <= MAX_INTERCEPT_GAP,
        "P=1 two-tier intercepts must match stage-1 (gap {worst_intercept:.3e})"
    );
    assert!(
        ll_gap <= MAX_LOGLIK_GAP,
        "P=1 two-tier loglik must match stage-1 (gap {ll_gap:.3e}: {ll2:.6} vs {ll1:.6})"
    );
}
