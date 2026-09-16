//! Reflection canonicalization for the multiple-group polytomous bifactor GRM
//! (stage 2 of #1912), mirroring `tests/poly_multigroup_reflection.rs`.
//!
//! With unconstrained slopes the graded model is invariant under the
//! per-dimension reflection `(a_.d, theta_d) -> (-a_.d, -theta_d)`, so the
//! latent orientation is not identified by the likelihood and must be fixed by
//! a restriction on the parameters (Bafumi et al., 2005). The fitter pins each
//! dimension so its largest-magnitude slope is positive (the crate rule in
//! `poly::canonicalize_slope_reflection` / `grm.rs`, applied consistently
//! across groups); for the general dimension that flip reverses every group's
//! mean on the common scale as well. Reporting the flipped slopes with the
//! unflipped means describes a different, worse-fitting model and inverts
//! every focal-group comparison (the #1879 mu-sign fix).
//!
//! # References (APA 7th ed.)
//!
//! Bafumi, J., Gelman, A., Park, D. K., & Kaplan, N. (2005). Practical issues in
//!   implementing and understanding Bayesian ideal point estimation. *Political
//!   Analysis, 13*(2), 171-187. https://doi.org/10.1093/pan/mpi010
//!
//! Samejima, F. (1969). Estimation of latent ability using a response pattern of
//!   graded scores. *Psychometrika Monograph Supplement, 34*(4, Pt. 2).
//!   https://doi.org/10.1007/BF03372160

use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig};

const PERSONS_PER_GROUP: usize = 1_500;
const N_ITEMS: usize = 6;
const N_SPECIFIC: usize = 1;
const N_CAT: usize = 4;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 0, 0];
const QUADRATURE_GENERAL: usize = 21;
const QUADRATURE_SPECIFIC: usize = 11;
const SEED: u64 = 20_260_916;
/// Most slopes and their sum are positive, so an EM started from `a = 1`
/// settles in the orientation of these values; item 0 has the largest magnitude
/// and is negative, so the fitter's canonicalization must reverse that
/// orientation before reporting.
const TRUE_A_GENERAL: [f64; N_ITEMS] = [-1.90, 1.20, 1.10, 1.00, 0.90, 1.20];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] = [1.00, 0.90, 1.10, 1.00, 0.80, 0.90];
const TRUE_THRESHOLDS: [f64; 3] = [1.20, 0.00, -1.20];
const FOCAL_MEAN: f64 = 0.80;
const MAX_MEAN_ERROR: f64 = 0.25;

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

fn graded_cells(a_g: f64, a_s: f64, t_g: f64, t_s: f64) -> [f64; N_CAT] {
    let base = a_g * t_g + a_s * t_s;
    let mut at_least = [1.0f64; N_CAT + 1];
    for k in 1..N_CAT {
        at_least[k] = sigmoid(base + TRUE_THRESHOLDS[k - 1]);
    }
    at_least[N_CAT] = 0.0;
    let mut cells = [0.0f64; N_CAT];
    for k in 0..N_CAT {
        cells[k] = at_least[k] - at_least[k + 1];
    }
    cells
}

#[test]
fn multigroup_means_follow_the_general_reflection() {
    let n_persons = 2 * PERSONS_PER_GROUP;
    let mut rng = Lcg(SEED);
    let mut responses = vec![0usize; n_persons * N_ITEMS];
    let mut group_id = vec![0usize; n_persons];
    for person in 0..n_persons {
        let group = usize::from(person >= PERSONS_PER_GROUP);
        group_id[person] = group;
        let t_g = rng.standard_normal() + if group == 1 { FOCAL_MEAN } else { 0.0 };
        let t_s = rng.standard_normal();
        for (item, &a_g) in TRUE_A_GENERAL.iter().enumerate() {
            let cells = graded_cells(a_g, TRUE_A_SPECIFIC[item], t_g, t_s);
            let mut draw = rng.uniform();
            let mut category = N_CAT - 1;
            for (k, p) in cells.iter().enumerate() {
                if draw < *p {
                    category = k;
                    break;
                }
                draw -= *p;
            }
            responses[person * N_ITEMS + item] = category;
        }
    }

    let cfg = BifactorMultigroupConfig {
        q_general: QUADRATURE_GENERAL,
        q_specific: QUADRATURE_SPECIFIC,
        max_iter: 500,
        tol: 1e-6,
        n_starts: 1,
        seed: SEED,
        newton_iter: 10,
        ridge: 1e-8,
        estimate_specific_vars: false,
    };
    let fit = fit_bifactor_grm_multigroup(
        &responses,
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
    .expect("two-group bifactor fit on well-conditioned simulated data must succeed");

    assert!(
        fit.a_general[0][0] > 0.0,
        "the largest-magnitude general slope must be reported positive: {:?}",
        fit.a_general[0]
    );
    // Reported orientation is the reverse of the generating one, so the focal
    // group's mean must be reported as -FOCAL_MEAN.
    assert!(
        (fit.general_mean[1] + FOCAL_MEAN).abs() <= MAX_MEAN_ERROR,
        "focal mean {:.4} must be reported in the orientation of the reported slopes \
         (expected about {:.2}); slopes {:?}",
        fit.general_mean[1],
        -FOCAL_MEAN,
        fit.a_general[0]
    );
    // Anchored items are identical across groups after the joint flip.
    for i in 0..N_ITEMS {
        assert!(
            (fit.a_general[0][i] - fit.a_general[1][i]).abs() == 0.0,
            "anchored general slopes must match across groups: {:?} vs {:?}",
            fit.a_general[0],
            fit.a_general[1]
        );
    }
}
