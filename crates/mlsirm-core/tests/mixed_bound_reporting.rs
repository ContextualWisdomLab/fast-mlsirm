//! A `fit_mixed_items` estimate resting on an optimizer bound must say so.
//!
//! The reported slope of a bounded item is a small positive number, which is
//! exactly what a genuinely low-discrimination item looks like. Without a
//! marker the two are indistinguishable in the output, and a reviewer
//! screening for weak items removes a working measurement rather than a bad
//! one. `MixedItemEstimate::at_bound` is that marker.
//!
//! The positivity floor that made a reverse-keyed item bounded has since been
//! removed, so these tests pin the marker's remaining contract: it stays silent
//! on every item that has an interior optimum, including a reverse-keyed one,
//! and it does not fire merely because a slope is small. The positive case —
//! the marker firing when a bound really binds — is a unit test next to
//! `bounds_in_force` itself.

use mlsirm_core::mixed::{fit_mixed_items, MixedItemKind, MixedItemSpec};

const N_PERSONS: usize = 1_200;
const SEED: u64 = 20_260_914;

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

/// Binary responses from `logit P = a*theta + b`, one column per slope.
fn simulate_two_pl(slopes: &[f64]) -> Vec<usize> {
    let mut rng = Lcg(SEED);
    let mut y = vec![0usize; N_PERSONS * slopes.len()];
    for person in 0..N_PERSONS {
        let theta = rng.standard_normal();
        for (item, &slope) in slopes.iter().enumerate() {
            let p = 1.0 / (1.0 + (-(slope * theta + 0.1)).exp());
            y[person * slopes.len() + item] = usize::from(rng.uniform() < p);
        }
    }
    y
}

fn fit(slopes: &[f64]) -> Vec<Vec<&'static str>> {
    let y = simulate_two_pl(slopes);
    let specs: Vec<MixedItemSpec> = slopes
        .iter()
        .map(|_| MixedItemSpec {
            kind: MixedItemKind::TwoPl,
            n_categories: 2,
        })
        .collect();
    let fit = fit_mixed_items(&y, None, N_PERSONS, slopes.len(), &specs, 1, 21, 3, 60, 1e-5, 1)
        .expect("mixed fit on well-conditioned simulated data must succeed");
    fit.items.into_iter().map(|item| item.at_bound).collect()
}

/// A reverse-keyed item used to be driven onto the positivity floor and had to
/// be reported as bounded. The bound is now symmetric, so the same item is
/// estimated properly and must report NOTHING — the marker tracks the bound, so
/// removing the bound must remove the report.
#[test]
fn a_reverse_keyed_item_no_longer_rests_on_a_bound() {
    let at_bound = fit(&[1.30, -1.10, 1.20, 0.95]);
    assert!(
        at_bound[1].is_empty(),
        "the reverse-keyed item is now estimable and must report no bound, got {:?}",
        at_bound[1]
    );
}

/// The marker must not fire on ordinary items, or it says nothing.
#[test]
fn ordinary_items_report_no_bound() {
    let at_bound = fit(&[1.30, -1.10, 1.20, 0.95]);
    for (item, roles) in at_bound.iter().enumerate() {
        assert!(
            roles.is_empty(),
            "item {item} has an interior optimum but reported {roles:?}"
        );
    }
}

/// A genuinely uninformative item is NOT at the bound: its slope is small but
/// interior. This is the discrimination the marker exists to make — without it
/// this item and the reverse-keyed one above report near-identical numbers.
#[test]
fn a_weak_but_interior_item_is_not_reported_at_the_bound() {
    let at_bound = fit(&[1.30, 0.15, 1.20, 0.95]);
    assert!(
        at_bound[1].is_empty(),
        "a weak item with an interior optimum must not be flagged, got {:?}",
        at_bound[1]
    );
}
