//! Unit tests for OLS + HC0–HC3 sandwich covariance and distribution tails.

use super::*;

fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b}, got {a} (tol={tol}, |diff|={})",
        (a - b).abs()
    );
}

#[test]
fn ols_recovers_exact_plane() {
    // y = 1 + 2 x1 + 3 x2 on a small design.
    let n = 4usize;
    let k = 3usize;
    let x = vec![
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0,
    ];
    let y = vec![1.0, 3.0, 4.0, 6.0];
    let fit = fit_ols(&x, &y, n, k).expect("OLS");
    assert_close(fit.beta[0], 1.0, 1e-10);
    assert_close(fit.beta[1], 2.0, 1e-10);
    assert_close(fit.beta[2], 3.0, 1e-10);
    assert!(fit.residuals.iter().all(|e| e.abs() < 1e-12));
}

#[test]
fn hc3_matches_hand_sandwich_on_synthetic() {
    // Deterministic heteroskedastic design (n=8, k=2).
    let n = 8usize;
    let k = 2usize;
    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) - 3.5;
        x.push(1.0);
        x.push(xi);
        y.push(2.0 + 0.5 * xi + if i % 2 == 0 { 0.8 } else { -0.3 } * (1.0 + xi.abs()));
    }
    let (fit, vcov) = fit_ols_hc(&x, &y, n, k, HcType::HC3).expect("HC3");

    // Hand HC3: bread * X' diag((e/(1-h))^2) X * bread
    let bread = &fit.xtx_inv;
    let mut meat = vec![0.0_f64; k * k];
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        let w = (fit.residuals[i] / (1.0 - fit.hat_diagonal[i])).powi(2);
        for a in 0..k {
            for b in 0..k {
                meat[a * k + b] += w * row[a] * row[b];
            }
        }
    }
    let mut tmp = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            tmp[a * k + b] = (0..k).map(|c| bread[a * k + c] * meat[c * k + b]).sum();
        }
    }
    let mut expected = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            expected[a * k + b] = (0..k).map(|c| tmp[a * k + c] * bread[c * k + b]).sum();
        }
    }
    for i in 0..k * k {
        assert_close(vcov[i], expected[i], 1e-12);
    }

    // HC0/HC1/HC2 ordering sanity: HC1 = n/(n-k) HC0.
    let v0 = sandwich_vcov(&fit, &x, HcType::HC0).unwrap();
    let v1 = sandwich_vcov(&fit, &x, HcType::HC1).unwrap();
    let scale = (n as f64) / ((n - k) as f64);
    for i in 0..k * k {
        assert_close(v1[i], scale * v0[i], 1e-12);
    }
}

#[test]
fn linear_contrast_wald_matches_manual() {
    let n = 10usize;
    let k = 2usize;
    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64;
        x.extend_from_slice(&[1.0, xi]);
        y.push(1.0 + 2.0 * xi + 0.1 * ((i % 3) as f64 - 1.0));
    }
    let (fit, vcov) = fit_ols_hc(&x, &y, n, k, HcType::HC3).unwrap();
    let c = vec![0.0, 1.0];
    let out = linear_contrast(&fit.beta, &vcov, &c, (n - k) as f64).unwrap();
    assert_close(out.estimate, fit.beta[1], 1e-12);
    assert_close(out.se, vcov[1 * k + 1].sqrt(), 1e-12);
    assert_close(out.wald_chi2, out.estimate.powi(2) / out.se.powi(2), 1e-12);
    assert_close(out.p_chi2, chi2_sf_df1(out.wald_chi2), 1e-12);
    assert_close(out.f_stat, out.t_stat.powi(2), 1e-12);
    assert_close(out.p_f, f_sf(out.f_stat, 1.0, (n - k) as f64), 1e-12);
}

#[test]
fn distribution_tails_known_values() {
    // χ²(1): P(Z² > 3.8414588) ≈ 0.05
    assert_close(chi2_sf_df1(3.841458820694124), 0.05, 1e-6);
    // Alias of fitstats::chi2_sf(., 1)
    let q = 2.0_f64;
    assert_close(chi2_sf_df1(q), crate::fitstats::chi2_sf(q, 1.0), 1e-14);

    // F(1, ∞) ≈ χ²(1); large df2
    assert_close(f_sf(3.841458820694124, 1.0, 1.0e8), 0.05, 5e-4);

    // t(∞) ≈ Normal: P(Z > 1.64485362695) ≈ 0.05
    assert_close(t_sf(1.6448536269514722, 1.0e8), 0.05, 5e-4);
    // Symmetric: P(T > large negative) → 1
    assert_close(t_sf(-100.0, 10.0), 1.0, 1e-10);
}

#[test]
fn rejects_rank_deficient_and_bad_hat() {
    let x = vec![1.0, 2.0, 1.0, 2.0];
    let y = vec![1.0, 2.0];
    assert!(fit_ols(&x, &y, 2, 2).is_err());
    assert!(HcType::parse("HC9").is_err());
}
