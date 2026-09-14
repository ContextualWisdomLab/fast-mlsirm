//! `fit_mixed_items` must represent a reverse-keyed item in a mixed bank.
//!
//! Mirrors the reproduction recorded against the module: an eight-item bank of
//! four binary and four graded items, one of each keyed against the trait.
//! Before the slope bound was made symmetric both reverse-keyed items came back
//! at `exp(-5) = 0.006738` — the bound itself, indistinguishable in the
//! reported number from a genuinely uninformative item.

use mlsirm_core::mixed::{fit_mixed_items, MixedItemKind, MixedItemSpec};

const N_PERSONS: usize = 2_000;
const SEED: u64 = 20_260_915;
const TRUE_SLOPES: [f64; 8] = [1.30, -1.10, 1.20, 0.95, 1.40, -1.25, 1.10, 1.05];
const THRESHOLDS: [f64; 3] = [1.2, 0.0, -1.2];
const MAX_MAGNITUDE_ERROR: f64 = 0.35;

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

fn specs() -> Vec<MixedItemSpec> {
    (0..TRUE_SLOPES.len())
        .map(|item| {
            if item < 4 {
                MixedItemSpec {
                    kind: MixedItemKind::TwoPl,
                    n_categories: 2,
                }
            } else {
                MixedItemSpec {
                    kind: MixedItemKind::Grm,
                    n_categories: 4,
                }
            }
        })
        .collect()
}

fn simulate() -> Vec<usize> {
    let mut rng = Lcg(SEED);
    let n_items = TRUE_SLOPES.len();
    let mut y = vec![0usize; N_PERSONS * n_items];
    for person in 0..N_PERSONS {
        let theta = rng.standard_normal();
        for (item, &slope) in TRUE_SLOPES.iter().enumerate() {
            y[person * n_items + item] = if item < 4 {
                let p = 1.0 / (1.0 + (-(slope * theta)).exp());
                usize::from(rng.uniform() < p)
            } else {
                let mut ge = [1.0_f64; 5];
                for k in 1..4 {
                    ge[k] = 1.0 / (1.0 + (-(slope * theta + THRESHOLDS[k - 1])).exp());
                }
                ge[4] = 0.0;
                let mut draw = rng.uniform();
                let mut category = 3;
                for k in 0..4 {
                    let p = ge[k] - ge[k + 1];
                    if draw < p {
                        category = k;
                        break;
                    }
                    draw -= p;
                }
                category
            };
        }
    }
    y
}

fn fitted_slopes() -> Vec<f64> {
    let specs = specs();
    let y = simulate();
    let fit = fit_mixed_items(
        &y,
        None,
        N_PERSONS,
        specs.len(),
        &specs,
        1,
        41,
        3,
        150,
        1e-5,
        1,
    )
    .expect("mixed fit on well-conditioned simulated data must succeed");
    fit.items
        .iter()
        .map(|item| item.slope.expect("every family here has a free slope"))
        .collect()
}

#[test]
fn reverse_keyed_items_are_recovered_with_negative_slopes() {
    let slope = fitted_slopes();

    // Identified up to the global reflection: magnitudes, then the sign pattern.
    for (item, (&estimated, &truth)) in slope.iter().zip(TRUE_SLOPES.iter()).enumerate() {
        assert!(
            (estimated.abs() - truth.abs()).abs() <= MAX_MAGNITUDE_ERROR,
            "item {item} recovered |{estimated:.5}| for true |{truth:.2}|: {slope:?}"
        );
    }
    for left in 0..TRUE_SLOPES.len() {
        for right in left + 1..TRUE_SLOPES.len() {
            assert_eq!(
                slope[left] * slope[right] > 0.0,
                TRUE_SLOPES[left] * TRUE_SLOPES[right] > 0.0,
                "items {left} and {right} must land on the same side of zero: {slope:?}"
            );
        }
    }
}

/// The exact symptom: the old bound returned `exp(-5)` for a reverse-keyed item.
#[test]
fn no_slope_rests_on_the_old_positivity_floor() {
    let floor = (-5.0_f64).exp();
    for (item, &estimated) in fitted_slopes().iter().enumerate() {
        assert!(
            (estimated - floor).abs() > 1e-6,
            "item {item} came back at the removed floor {floor:.6}"
        );
    }
}

/// The returned reflection is the canonical one.
#[test]
fn the_largest_magnitude_slope_is_returned_positive() {
    let slope = fitted_slopes();
    let anchor = slope
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .expect("one slope per item");
    assert!(
        slope[anchor] > 0.0,
        "the largest-magnitude slope must be positive: {slope:?}"
    );
}
