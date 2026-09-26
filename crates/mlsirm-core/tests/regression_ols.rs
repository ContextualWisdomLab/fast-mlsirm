//! Integration tests for OLS + HC sandwich (links the already-built library).

use mlsirm_core::regression::{
    chi2_sf_df1, f_sf, fit_ols, fit_ols_hc, linear_contrast, nested_ols_column_drop,
    sandwich_vcov, t_sf, HcType,
};

fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b}, got {a} (tol={tol}, |diff|={})",
        (a - b).abs()
    );
}

#[test]
fn ols_recovers_exact_plane() {
    let n = 4usize;
    let k = 3usize;
    let x = vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let y = vec![1.0, 3.0, 4.0, 6.0];
    let fit = fit_ols(&x, &y, n, k).expect("OLS");
    assert_close(fit.beta[0], 1.0, 1e-10);
    assert_close(fit.beta[1], 2.0, 1e-10);
    assert_close(fit.beta[2], 3.0, 1e-10);
}

#[test]
fn nested_column_drop_reports_classical_fit_statistics() {
    let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let y = vec![1.0, 2.0, 2.0, 4.0, 5.0];
    let result = nested_ols_column_drop(&x, &y, 5, 2, &[1]).unwrap();
    assert_close(result.sst, 10.8, 1e-12);
    assert_close(result.sse_full, 0.8, 1e-12);
    assert_close(result.sse_reduced, 10.8, 1e-12);
    assert_close(result.r2_full, 25.0 / 27.0, 1e-12);
    assert_close(result.adjusted_r2_full, 73.0 / 81.0, 1e-12);
    assert_close(result.r2_reduced, 0.0, 1e-12);
    assert_close(result.adjusted_r2_reduced, 0.0, 1e-12);
    assert_close(result.delta_r2, 25.0 / 27.0, 1e-12);
    assert_close(result.f_stat, 37.5, 1e-12);
    assert_eq!((result.df1, result.df2), (1, 3));
    assert!(result.p_f > 0.0 && result.p_f < 0.01);

    assert!(nested_ols_column_drop(&x, &y, 5, 2, &[]).is_err());
    assert!(nested_ols_column_drop(&x, &y, 5, 2, &[2]).is_err());
    assert!(nested_ols_column_drop(&x, &[2.0; 5], 5, 2, &[1]).is_err());
    let no_intercept = vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0, 4.0, 16.0];
    assert!(nested_ols_column_drop(&no_intercept, &y, 5, 2, &[1]).is_err());
}

#[test]
fn hc3_matches_hand_sandwich_on_synthetic() {
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
    let v0 = sandwich_vcov(&fit, &x, HcType::HC0).unwrap();
    let v1 = sandwich_vcov(&fit, &x, HcType::HC1).unwrap();
    let scale = (n as f64) / ((n - k) as f64);
    for i in 0..k * k {
        assert_close(v1[i], scale * v0[i], 1e-12);
    }
}

#[test]
fn linear_contrast_and_tails() {
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
    assert_close(out.wald_chi2, out.estimate.powi(2) / out.se.powi(2), 1e-12);
    assert_close(chi2_sf_df1(3.841458820694124), 0.05, 1e-6);
    assert_close(f_sf(3.841458820694124, 1.0, 1.0e8), 0.05, 5e-4);
    assert_close(t_sf(1.6448536269514722, 1.0e8), 0.05, 5e-4);
}
