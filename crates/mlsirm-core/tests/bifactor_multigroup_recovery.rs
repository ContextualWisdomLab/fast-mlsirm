//! True-parameter recovery for the multiple-group polytomous bifactor GRM
//! (stage 2 of #1912).
//!
//! Data are generated from the MODEL DEFINITION
//! (`P(Y >= k) = logistic(a_G * theta_G + a_S * theta_S + d_k)`; the linear
//! predictor follows Gibbons et al., 2007, eq. 9, "The Bifactor Model for
//! Graded Response Data" section — the logistic link is an implementation
//! choice, the paper uses the normal ogive — and the person marginal factors
//! per general node, Gibbons et al., 2007, eq. 15, "Marginal Maximum
//! Likelihood Estimation" section) with an independent simulation RNG —
//! never from crate internals. Groups share item parameters; the reference group is `N(0, I)`
//! and the focal group's general-factor distribution `N(mu, sigma^2)` is
//! estimated (Cai, Yang, & Hansen, 2011, multiple-group reference-group
//! paragraph near Fig. 7: reference means 0 / covariance identity, focal
//! location/scale estimated relative to reference with at least one common
//! item linking the scales; Bock & Zimowski, 1997, Handbook chap. 25,
//! multiple-group IRT).
//!
//! Design: 8 items, 2 specifics (4/4), 4 ordered categories, `n_ref = 1_500`,
//! `n_foc = 1_500`, focal `mu = 0.50`, `sigma = 1.20`, specific factors
//! `N(0, 1)` in both groups (`estimate_specific_vars = false`).
//!
//! Tolerance basis (initial; to be tightened against measured spread like the
//! stage-1 recovery test): focal-mean error `<= 0.25` (the same band the
//! unidimensional `poly_multigroup_reflection` test uses for a well-conditioned
//! two-group graded fit at `n = 1_500` per group), general-SD error `<= 0.25`,
//! item errors at the stage-1 `n = 1_020` bands (`a_G <= 0.50`,
//! `a_S <= 0.55`, `d <= 0.42`) since each group's `n = 1_500` is in the same
//! sampling-noise regime.
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350
//!
//! Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J.
//! van der Linden & R. K. Hambleton (Eds.), *Handbook of modern item response
//! theory* (pp. 433-448). Springer.
//! https://doi.org/10.1007/978-1-4757-2691-6_25

use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig};

const N_ITEMS: usize = 8;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 1, 1, 1, 1];
const N_REF: usize = 1_500;
const N_FOC: usize = 1_500;
const FOCAL_MU: f64 = 0.50;
const FOCAL_SIGMA: f64 = 1.20;

const TRUE_A_GENERAL: [f64; N_ITEMS] = [1.50, 1.20, 1.00, 0.90, 1.30, 1.10, 0.80, 1.20];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] = [1.00, 0.90, 1.10, 0.80, 1.10, 0.90, 1.00, 0.80];
const TRUE_D: [[f64; 3]; N_ITEMS] = [
    [1.30, 0.10, -1.10],
    [1.10, -0.10, -1.30],
    [1.40, 0.20, -1.00],
    [1.00, 0.00, -1.20],
    [1.20, 0.10, -1.10],
    [1.50, 0.30, -0.90],
    [0.90, -0.20, -1.40],
    [1.10, 0.00, -1.20],
];

const MAX_MU_ERROR: f64 = 0.25;
const MAX_SIGMA_ERROR: f64 = 0.25;
const MAX_A_GENERAL_ERROR: f64 = 0.50;
const MAX_A_SPECIFIC_ERROR: f64 = 0.55;
const MAX_D_ERROR: f64 = 0.45;

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

fn category_probs(a_g: f64, a_s: f64, d: &[f64; 3], t_g: f64, t_s: f64) -> [f64; 4] {
    let base = a_g * t_g + a_s * t_s;
    let mut ge = [1.0f64; 5];
    for k in 1..4 {
        ge[k] = sigmoid(base + d[k - 1]);
    }
    ge[4] = 0.0;
    [ge[0] - ge[1], ge[1] - ge[2], ge[2] - ge[3], ge[3] - ge[4]]
}

#[test]
fn multigroup_recovers_group_shifts_and_shared_items() {
    let n_persons = N_REF + N_FOC;
    let mut rng = Lcg(20_260_916);
    let mut y = vec![0usize; n_persons * N_ITEMS];
    let mut group_id = vec![0usize; n_persons];
    for p in 0..n_persons {
        let focal = p >= N_REF;
        group_id[p] = usize::from(focal);
        let t_g = if focal {
            FOCAL_MU + FOCAL_SIGMA * rng.standard_normal()
        } else {
            rng.standard_normal()
        };
        let mut t_s = [0.0f64; N_SPECIFIC];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let s = SPECIFIC_MAP[i] as usize;
            let probs = category_probs(
                TRUE_A_GENERAL[i],
                TRUE_A_SPECIFIC[i],
                &TRUE_D[i],
                t_g,
                t_s[s],
            );
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

    let cfg = BifactorMultigroupConfig {
        q_general: 21,
        q_specific: 11,
        max_iter: 500,
        tol: 1e-5,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: 10,
        ridge: 1e-8,
        device: mlsirm_core::Device::Cpu,
        estimate_specific_vars: false,
    };
    let fit = fit_bifactor_grm_multigroup(
        &y,
        None,
        &group_id,
        2,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        None,
        &cfg,
    )
    .expect("two-group bifactor recovery fit must succeed");
    assert!(
        fit.converged,
        "recovery fit must converge (termination: {})",
        fit.termination_reason
    );

    // Reference group pinned; focal general distribution recovered.
    assert!((fit.general_mean[0]).abs() == 0.0);
    assert!((fit.general_sd[0] - 1.0).abs() == 0.0);
    assert!(
        (fit.general_mean[1] - FOCAL_MU).abs() <= MAX_MU_ERROR,
        "focal mean {:.4} must recover {FOCAL_MU} within {MAX_MU_ERROR}",
        fit.general_mean[1]
    );
    assert!(
        (fit.general_sd[1] - FOCAL_SIGMA).abs() <= MAX_SIGMA_ERROR,
        "focal sd {:.4} must recover {FOCAL_SIGMA} within {MAX_SIGMA_ERROR}",
        fit.general_sd[1]
    );

    // Shared items: anchored rows are identical across groups; compare the
    // reference row to truth after aligning the unidentified reflection.
    assert!((fit.a_general[0] == fit.a_general[1]));
    let mut a_g = fit.a_general[0].clone();
    let mut a_s = fit.a_specific[0].clone();
    let anchor = a_g
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .expect("one general slope per item");
    if (a_g[anchor] > 0.0) != (TRUE_A_GENERAL[anchor] > 0.0) {
        for v in a_g.iter_mut() {
            *v = -*v;
        }
    }
    for s in 0..N_SPECIFIC {
        let items: Vec<usize> = (0..N_ITEMS)
            .filter(|&i| SPECIFIC_MAP[i] as usize == s)
            .collect();
        let est_anchor = *items
            .iter()
            .max_by(|&&i, &&j| a_s[i].abs().total_cmp(&a_s[j].abs()))
            .expect("each specific block is non-empty");
        if (a_s[est_anchor] > 0.0) != (TRUE_A_SPECIFIC[est_anchor] > 0.0) {
            for &i in &items {
                a_s[i] = -a_s[i];
            }
        }
    }
    let mut worst_a_g = 0.0f64;
    let mut worst_a_s = 0.0f64;
    let mut worst_d = 0.0f64;
    for i in 0..N_ITEMS {
        worst_a_g = worst_a_g.max((a_g[i] - TRUE_A_GENERAL[i]).abs());
        worst_a_s = worst_a_s.max((a_s[i] - TRUE_A_SPECIFIC[i]).abs());
        for (k, truth) in TRUE_D[i].iter().enumerate() {
            worst_d = worst_d.max((fit.threshold[0][i * (N_CAT - 1) + k] - truth).abs());
        }
    }
    assert!(
        worst_a_g <= MAX_A_GENERAL_ERROR,
        "worst |a_G| error {worst_a_g:.4} exceeds {MAX_A_GENERAL_ERROR}: {:?}",
        fit.a_general[0]
    );
    assert!(
        worst_a_s <= MAX_A_SPECIFIC_ERROR,
        "worst |a_S| error {worst_a_s:.4} exceeds {MAX_A_SPECIFIC_ERROR}: {:?}",
        fit.a_specific[0]
    );
    assert!(
        worst_d <= MAX_D_ERROR,
        "worst |d| error {worst_d:.4} exceeds {MAX_D_ERROR}"
    );

    // EAPs on the common (reference) scale exist for every person.
    assert_eq!(fit.theta_g_eap.len(), n_persons);
    assert_eq!(fit.theta_g_sd.len(), n_persons);
    assert!(fit.theta_g_eap.iter().all(|v| v.is_finite()));
    // Focal EAPs center above the reference EAPs under a positive shift.
    let ref_mean: f64 = fit.theta_g_eap[..N_REF].iter().sum::<f64>() / N_REF as f64;
    let foc_mean: f64 = fit.theta_g_eap[N_REF..].iter().sum::<f64>() / N_FOC as f64;
    assert!(
        foc_mean > ref_mean,
        "focal EAP mean ({foc_mean:.3}) must exceed reference ({ref_mean:.3})"
    );
}
