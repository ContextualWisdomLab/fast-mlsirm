use super::*;

#[test]
fn vuong_favors_the_better_model() {
    // model A consistently better by 0.2 per case, with case noise
    let mut state = 5u64;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n = 400;
    let la: Vec<f64> = (0..n).map(|_| -1.0 + 0.1 * unif()).collect();
    let lb: Vec<f64> = la.iter().map(|&v| v - 0.2 - 0.3 * (unif() - 0.5)).collect();
    let res = vuong_nonnested(&la, &lb, 10, 10, false).unwrap();
    assert!(
        res.z > 2.0,
        "A must be significantly favored: z = {}",
        res.z
    );
    assert!(res.p_two_sided < 0.05);
    // BIC correction penalizes the bigger model
    let res_pen = vuong_nonnested(&la, &lb, 40, 10, true).unwrap();
    assert!(res_pen.z < res.z);
    // identical models are rejected as indistinguishable
    assert!(vuong_nonnested(&la, &la, 10, 10, false).is_err());
}

#[test]
fn vuong_rejects_nonfinite_likelihoods_and_differences() {
    assert!(vuong_nonnested(&[0.0, f64::NAN], &[0.0, 0.0], 1, 1, false).is_err());
    assert!(vuong_nonnested(&[0.0, f64::INFINITY], &[0.0, 0.0], 1, 1, false).is_err());
    assert!(vuong_nonnested(&[f64::MAX, 0.0], &[-f64::MAX, 1.0], 1, 1, false).is_err());
}

#[test]
fn erfc_reference_values() {
    assert!((erfc(0.0) - 1.0).abs() < 1e-7);
    assert!((erfc(1.959963984540054 / std::f64::consts::SQRT_2) - 0.05).abs() < 1e-4);
}

#[test]
fn q3_detects_locally_dependent_pair() {
    // residuals: items 0 and 1 share an extra common factor
    let mut state = 11u64;
    let mut norm = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (((state >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let (n_persons, n_items) = (600, 6);
    let mut resid = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let shared = norm();
        for i in 0..n_items {
            resid[p * n_items + i] = norm() * 0.4 + if i < 2 { 0.6 * shared } else { 0.0 };
        }
    }
    let out = dimensionality_residuals(&resid, n_persons, n_items).unwrap();
    assert!(
        out.q3[0] > 0.5,
        "dependent pair must show high Q3: {}",
        out.q3[0]
    );
    assert!(out.q3_max_abs >= out.q3[0].abs());
    assert!(out.gddm > 0.0);
    assert_eq!(out.gddm, out.mean_abs_residual_cross_product);
}
