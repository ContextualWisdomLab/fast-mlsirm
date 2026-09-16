//! The reflection that `fit_poly_multigroup` applies to its slopes must also be
//! applied to the group means it reports.
//!
//! With an unconstrained slope the graded model is invariant under
//! `(a_i, theta) -> (-a_i, -theta)`, so the orientation of the latent variable is
//! not identified by the likelihood and must be fixed by a restriction on the
//! parameters (Bafumi et al., 2005). When the fitter reverses the slope vector to
//! meet that restriction, `theta` is reversed with it; a group whose mean was
//! `mu` in the estimated orientation has mean `-mu` in the reported one.
//! Reporting the flipped slopes with the unflipped means describes a different,
//! worse-fitting model and inverts every focal-group comparison.
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

use mlsirm_core::poly::{fit_poly_multigroup, PolyModel};

const PERSONS_PER_GROUP: usize = 1_500;
const N_CAT: usize = 4;
const QUADRATURE_POINTS: usize = 41;
const SEED: u64 = 20_260_916;
/// Most slopes and their sum are positive, so an EM started from `a = 1`
/// settles in the orientation of these values; item 0 has the largest magnitude
/// and is negative, so the fitter's canonicalization must reverse that
/// orientation before reporting.
const TRUE_SLOPES: [f64; 6] = [-1.90, 1.20, 1.10, 1.00, 0.90, 1.20];
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
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Graded-model cell probabilities from `P(Y >= k) = sigmoid(a*theta + beta_k)`.
fn graded_cells(slope: f64, theta: f64) -> [f64; N_CAT] {
    let mut at_least = [1.0_f64; N_CAT + 1];
    for k in 1..N_CAT {
        at_least[k] = sigmoid(slope * theta + TRUE_THRESHOLDS[k - 1]);
    }
    at_least[N_CAT] = 0.0;
    let mut cells = [0.0_f64; N_CAT];
    for k in 0..N_CAT {
        cells[k] = at_least[k] - at_least[k + 1];
    }
    cells
}

#[test]
fn reported_group_means_follow_the_slope_reflection() {
    let n_items = TRUE_SLOPES.len();
    let n_persons = 2 * PERSONS_PER_GROUP;
    let mut rng = Lcg(SEED);
    let mut responses = vec![0usize; n_persons * n_items];
    let mut group_id = vec![0usize; n_persons];
    for person in 0..n_persons {
        let group = usize::from(person >= PERSONS_PER_GROUP);
        group_id[person] = group;
        let theta = rng.standard_normal() + if group == 1 { FOCAL_MEAN } else { 0.0 };
        for (item, &slope) in TRUE_SLOPES.iter().enumerate() {
            let cells = graded_cells(slope, theta);
            let mut draw = rng.uniform();
            let mut category = N_CAT - 1;
            for (k, p) in cells.iter().enumerate() {
                if draw < *p {
                    category = k;
                    break;
                }
                draw -= *p;
            }
            responses[person * n_items + item] = category;
        }
    }

    let fit = fit_poly_multigroup(
        &responses,
        None,
        &group_id,
        2,
        n_persons,
        n_items,
        N_CAT,
        PolyModel::Grm,
        None,
        QUADRATURE_POINTS,
        500,
        1e-6,
    )
    .expect("two-group graded fit on well-conditioned simulated data must succeed");

    assert!(
        fit.slope[0] > 0.0,
        "the largest-magnitude slope must be reported positive: {:?}",
        fit.slope
    );
    // Reported orientation is the reverse of the generating one, so the focal
    // group's mean must be reported as -FOCAL_MEAN.
    assert!(
        (fit.mu[1] + FOCAL_MEAN).abs() <= MAX_MEAN_ERROR,
        "focal mean {:.4} must be reported in the orientation of the reported slopes \
         (expected about {:.2}); slopes {:?}",
        fit.mu[1],
        -FOCAL_MEAN,
        fit.slope
    );
}
