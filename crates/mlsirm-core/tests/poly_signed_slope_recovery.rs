//! `fit_poly_unidim` must be able to represent a reverse-keyed item.
//!
//! Both polytomous cells — `PolyModel::Grm` and `PolyModel::Gpcm` — are fitted by
//! the same estimator, so the slope parametrization is a property of the
//! estimator and not of either model. Samejima's graded model and Muraki's
//! generalized partial credit model both place the discrimination on the real
//! line; a negative `a_i` is the standard representation of an item keyed
//! against the construct, and the confirmatory `crate::grm` / `crate::gpcm`
//! fitters in this crate already estimate it unconstrained.
//!
//! The data are generated here from the model definition rather than from crate
//! internals, so the test fails if the estimator stops recovering the sign for
//! any reason — a re-imposed positivity constraint, a bad initial value, or a
//! reflection rule that flips the wrong way.
//!
//! # References (APA 7th ed.)
//!
//! Muraki, E. (1992). A generalized partial credit model: Application of an EM
//!   algorithm. *Applied Psychological Measurement, 16*(2), 159-176.
//!   https://doi.org/10.1177/014662169201600206
//!
//! Samejima, F. (1969). Estimation of latent ability using a response pattern of
//!   graded scores. *Psychometrika Monograph Supplement, 34*(4, Pt. 2).
//!   https://doi.org/10.1007/BF03372160

use mlsirm_core::poly::{fit_poly_unidim, PolyModel};

const N_PERSONS: usize = 2_000;
const N_CAT: usize = 4;
const QUADRATURE_POINTS: usize = 41;
const SEED: u64 = 20_260_914;

/// Item 0 carries the largest magnitude and is positive, so the fit's reflection
/// canonicalization is a no-op and the recovered signs are directly comparable.
const TRUE_SLOPES: [f64; 6] = [1.60, -1.20, 1.10, -1.50, 1.30, 0.90];
/// Strictly decreasing, as `PolyModel::Grm` requires; reused as the GPCM
/// intercepts, where no ordering is required.
const TRUE_CAT_PARAMS: [f64; 3] = [1.20, 0.00, -1.20];
const MAX_ABSOLUTE_SLOPE_ERROR: f64 = 0.30;

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

/// Category probabilities written straight from the model definitions.
fn cell(model: PolyModel, slope: f64, theta: f64) -> Vec<f64> {
    let base = slope * theta;
    match model {
        // P(Y >= k) = sigmoid(a*theta + beta_k); differences give the cells.
        PolyModel::Grm => {
            let mut ge = vec![1.0_f64; N_CAT + 1];
            for k in 1..N_CAT {
                ge[k] = sigmoid(base + TRUE_CAT_PARAMS[k - 1]);
            }
            ge[N_CAT] = 0.0;
            (0..N_CAT).map(|k| ge[k] - ge[k + 1]).collect()
        }
        // softmax_k(k*a*theta + c_k), with c_0 = 0.
        PolyModel::Gpcm => {
            let psi: Vec<f64> = (0..N_CAT)
                .map(|k| {
                    let intercept = if k == 0 { 0.0 } else { TRUE_CAT_PARAMS[k - 1] };
                    (k as f64) * base + intercept
                })
                .collect();
            let max = psi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let unnormalized: Vec<f64> = psi.iter().map(|p| (p - max).exp()).collect();
            let total: f64 = unnormalized.iter().sum();
            unnormalized.iter().map(|u| u / total).collect()
        }
    }
}

fn simulate(model: PolyModel) -> Vec<usize> {
    let mut rng = Lcg(SEED);
    let mut responses = vec![0usize; N_PERSONS * TRUE_SLOPES.len()];
    for person in 0..N_PERSONS {
        let theta = rng.standard_normal();
        for (item, &slope) in TRUE_SLOPES.iter().enumerate() {
            let probabilities = cell(model, slope, theta);
            let mut draw = rng.uniform();
            let mut category = N_CAT - 1;
            for (k, p) in probabilities.iter().enumerate() {
                if draw < *p {
                    category = k;
                    break;
                }
                draw -= *p;
            }
            responses[person * TRUE_SLOPES.len() + item] = category;
        }
    }
    responses
}

fn recovered_slopes(model: PolyModel) -> Vec<f64> {
    let responses = simulate(model);
    let fit = fit_poly_unidim(
        &responses,
        None,
        N_PERSONS,
        TRUE_SLOPES.len(),
        N_CAT,
        model,
        QUADRATURE_POINTS,
        200,
        1e-6,
    )
    .expect("polytomous fit on well-conditioned simulated data must succeed");
    fit.slope
}

fn assert_signed_recovery(model: PolyModel) {
    let slope = recovered_slopes(model);

    // The orientation of the latent trait is not identified, so compare the
    // recovered vector to the truth up to the global reflection: magnitudes
    // item by item, and the sign PATTERN across items. A fitter that floors
    // reverse-keyed items at zero fails both.
    for (item, (&estimated, &truth)) in slope.iter().zip(TRUE_SLOPES.iter()).enumerate() {
        assert!(
            (estimated.abs() - truth.abs()).abs() <= MAX_ABSOLUTE_SLOPE_ERROR,
            "item {item} recovered |slope| {:.5} is further than {MAX_ABSOLUTE_SLOPE_ERROR} \
             from true |{truth:.2}| ({model:?}); recovered vector {slope:?}",
            estimated.abs()
        );
    }
    for i in 0..TRUE_SLOPES.len() {
        for j in i + 1..TRUE_SLOPES.len() {
            assert_eq!(
                slope[i] * slope[j] > 0.0,
                TRUE_SLOPES[i] * TRUE_SLOPES[j] > 0.0,
                "items {i} and {j} must be recovered on the same side of zero as \
                 the true {:.2} / {:.2} ({model:?}); recovered vector {slope:?}",
                TRUE_SLOPES[i],
                TRUE_SLOPES[j]
            );
        }
    }
    assert_canonical_orientation(&slope, model);
}

/// The returned reflection is the one whose largest-magnitude slope is positive
/// (`poly::canonicalize_slope_reflection`, matching `crate::grm`).
fn assert_canonical_orientation(slope: &[f64], model: PolyModel) {
    let anchor = slope
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .expect("the fit returns one slope per item");
    assert!(
        slope[anchor] > 0.0,
        "the largest-magnitude recovered slope must be positive ({model:?}): {slope:?}"
    );
}

#[test]
fn graded_fit_recovers_negative_slopes_for_reverse_keyed_items() {
    assert_signed_recovery(PolyModel::Grm);
}

#[test]
fn partial_credit_fit_recovers_negative_slopes_for_reverse_keyed_items() {
    assert_signed_recovery(PolyModel::Gpcm);
}

/// A reverse-keyed item must not be reported as uninformative. This is the
/// symptom the defect produced: the constrained fitter returned slopes at the
/// zero floor, which reads as "this item measures nothing" rather than "this
/// item is keyed the other way".
#[test]
fn reverse_keyed_items_are_not_reported_at_the_zero_floor() {
    for model in [PolyModel::Grm, PolyModel::Gpcm] {
        let slope = recovered_slopes(model);
        for (item, (&estimated, &truth)) in slope.iter().zip(TRUE_SLOPES.iter()).enumerate() {
            if truth >= 0.0 {
                continue;
            }
            assert!(
                estimated.abs() > 0.5,
                "reverse-keyed item {item} was recovered at |{estimated:.5}|, near the zero \
                 floor the removed positivity constraint used to produce ({model:?})"
            );
        }
    }
}
