use super::*;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        assert!(
            w[1] >= w[0] - 1e-6,
            "loglik decreased: {} -> {}",
            w[0],
            w[1]
        );
    }
    assert!(corr(&res.a, &a_true) > 0.85, "a recovery too low");
    assert!(corr(&res.b, &b_true) > 0.9, "b recovery too low");
    assert!(
        corr(&res.theta, &theta_true) > 0.8,
        "theta recovery too low"
    );
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
    assert!(
        res.theta[0].abs() < 1e-6,
        "all-missing person should shrink to prior mean 0"
    );
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
    let cfg = MmleConfig {
        ridge_a: 0.0,
        ridge_b: 0.0,
        max_iter: 50,
        ..MmleConfig::default()
    };
    let res = fit_mmle_2pl(&y, &observed, n_persons, n_items, &cfg);

    assert!(
        res.a.iter().all(|v| v.is_finite()),
        "item slopes must stay finite"
    );
    assert!(
        res.b.iter().all(|v| v.is_finite()),
        "item intercepts must stay finite"
    );
    assert!(
        res.theta.iter().all(|t| t.is_finite()),
        "abilities must stay finite"
    );
    // The zero-information item keeps its initial (a = 1, b = 0) because the
    // Newton step breaks on the singular Hessian before any update applies.
    assert_eq!(
        res.a[2], 1.0,
        "unobserved item slope must stay at its initial value"
    );
    assert_eq!(
        res.b[2], 0.0,
        "unobserved item intercept must stay at its initial value"
    );
}

/// Simulate 2PL responses from given slopes, intercepts at zero.
fn simulate_2pl(slopes: &[f64], n_persons: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
    let mut rng = Lcg(seed);
    let n_items = slopes.len();
    let mut y = vec![0.0_f64; n_persons * n_items];
    for person in 0..n_persons {
        let theta = rng.normal();
        for (item, &slope) in slopes.iter().enumerate() {
            let p = 1.0 / (1.0 + (-(slope * theta)).exp());
            y[person * n_items + item] = f64::from(u8::from(rng.next_f64() < p));
        }
    }
    (y, vec![true; n_persons * n_items])
}

/// The reflection `(a, theta) -> (-a, -theta)` holds the likelihood fixed, so
/// the solution's overall sign is not identified by the data. It must be pinned
/// by a rule rather than left to initialization, because the two backends do
/// not share one. The anchor convention: the largest-magnitude slope is
/// positive.
#[test]
fn the_largest_magnitude_slope_is_returned_positive() {
    // Every item keyed AGAINST the trait. The likelihood is identical to the
    // all-positive mirror, so only the convention decides what comes back.
    let truth = [-1.40, -1.10, -0.90, -1.25];
    let (y, observed) = simulate_2pl(&truth, 900, 4242);
    let res = fit_mmle_2pl(&y, &observed, 900, truth.len(), &MmleConfig::default());

    let anchor = res
        .a
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .expect("one slope per item");
    assert!(
        res.a[anchor] > 0.0,
        "the largest-magnitude slope must be returned positive: {:?}",
        res.a
    );
    // The data are all-reverse-keyed, so the canonical solution is the mirror:
    // every slope positive, and theta negatively related to the raw score.
    assert!(
        res.a.iter().all(|value| *value > 0.0),
        "the whole slope vector must be on one orientation: {:?}",
        res.a
    );
}

/// The rule must be a no-op on ordinary data, or it would silently rewrite
/// every fit that was already correct.
#[test]
fn ordinary_data_is_returned_unflipped() {
    let truth = [1.40, 1.10, -0.90, 1.25];
    let (y, observed) = simulate_2pl(&truth, 900, 4242);
    let res = fit_mmle_2pl(&y, &observed, 900, truth.len(), &MmleConfig::default());

    for (item, (&estimated, &expected)) in res.a.iter().zip(truth.iter()).enumerate() {
        assert_eq!(
            estimated < 0.0,
            expected < 0.0,
            "item {item} came back on the wrong side of zero ({estimated:.4} for {expected:.2}); \
             the anchor is positive here so no flip should have happened: {:?}",
            res.a
        );
    }
}

/// The point of removing the positivity floor: a reverse-keyed item comes back
/// with a negative slope of roughly the right size, instead of at the floor.
#[test]
fn a_reverse_keyed_item_recovers_its_negative_slope() {
    let truth = [1.40, 1.10, -0.90, 1.25];
    let (y, observed) = simulate_2pl(&truth, 1500, 4242);
    let res = fit_mmle_2pl(&y, &observed, 1500, truth.len(), &MmleConfig::default());
    for (item, (&estimated, &expected)) in res.a.iter().zip(truth.iter()).enumerate() {
        assert!(
            (estimated - expected).abs() < 0.35,
            "item {item} recovered {estimated:.4} for true {expected:.2}: {:?}",
            res.a
        );
    }
}
