#![cfg(feature = "gpu")]
use mlsirm_core::samefit_fifa::{gpu::evaluate, infer, transform};
use mlsirm_core::samefit_grm::PackedArchive;
#[test]
fn free_item_gpu_all_rows_jacobian_covariance() {
    let a = PackedArchive::new(
        vec![false],
        2,
        3,
        vec![2., 1., 2f64.ln(), 1.7, 0.8, 1.2f64.ln(), 0.5, 1.5f64.ln()],
    )
    .unwrap();
    let t = transform(&a).unwrap();
    let p = t.n_parameters;
    let mut h = vec![0.; p * p];
    for i in 0..p {
        h[i * p + i] = 2.;
    }
    h[1] = 0.3;
    h[p] = 0.3;
    let cpu = infer(&t, &h).unwrap();
    let gpu = evaluate(&a, &cpu.parameter_covariance).unwrap();
    assert!(gpu.dispatches >= 3);
    assert_ne!(gpu.backend, "CPU");
    for (x, y) in gpu.estimates.iter().zip(&t.rows) {
        assert!((x - y.estimate).abs() < 2e-5);
    }
    for (x, y) in gpu.jacobian.iter().zip(&t.jacobian) {
        assert!((x - y).abs() < 2e-5);
    }
    for (x, y) in gpu.covariance.iter().zip(&cpu.transformed_covariance) {
        assert!((x - y).abs() < 2e-5);
    }
    let x: Vec<_> = a.packed_bits().iter().map(|&b| f64::from_bits(b)).collect();
    for c in 0..p {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[c] += 1e-5;
        xm[c] -= 1e-5;
        let tp = transform(&PackedArchive::new(vec![false], 2, 3, xp).unwrap()).unwrap();
        let tm = transform(&PackedArchive::new(vec![false], 2, 3, xm).unwrap()).unwrap();
        for r in 0..t.rows.len() {
            assert!(
                ((tp.rows[r].estimate - tm.rows[r].estimate) / 2e-5 - t.jacobian[r * p + c]).abs()
                    < 2e-8
            );
        }
    }
    println!(
        "device={} backend={} dispatches={}",
        gpu.device, gpu.backend, gpu.dispatches
    );
}

#[test]
fn anchored_negative_slope_gpu_and_invalid_covariance() {
    let a = PackedArchive::new(
        vec![true],
        3,
        3,
        vec![-2., 1., 2f64.ln(), 0.5, 1.5f64.ln(), -0.3, 0.8f64.ln()],
    )
    .unwrap();
    let t = transform(&a).unwrap();
    let p = t.n_parameters;
    let mut h = vec![0.; p * p];
    for i in 0..p {
        h[i * p + i] = 2.;
    }
    let cpu = infer(&t, &h).unwrap();
    let gpu = evaluate(&a, &cpu.parameter_covariance).unwrap();
    assert_eq!(gpu.estimates.len(), t.rows.len());
    assert_eq!(gpu.jacobian.len(), t.jacobian.len());
    assert_eq!(gpu.covariance.len(), cpu.transformed_covariance.len());
    assert_eq!(gpu.dispatches, 3);
    for (x, y) in gpu.estimates.iter().zip(&t.rows) {
        assert!((x - y.estimate).abs() < 2e-5);
    }
    for (x, y) in gpu.jacobian.iter().zip(&t.jacobian) {
        assert!((x - y).abs() < 2e-5);
    }
    for (x, y) in gpu.covariance.iter().zip(&cpu.transformed_covariance) {
        assert!((x - y).abs() < 2e-5);
    }
    assert!(evaluate(&a, &[]).is_err());
    let mut bad = cpu.parameter_covariance.clone();
    bad[0] = f64::NAN;
    assert!(evaluate(&a, &bad).is_err());
    let zero = PackedArchive::new(vec![true], 2, 2, vec![0., 1., 0., 0.]).unwrap();
    assert!(evaluate(&zero, &[0.; 16]).is_err());
}
