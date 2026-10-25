#![cfg(feature = "gpu")]
use mlsirm_core::samefit_fifa::{gpu::evaluate, transform};
use mlsirm_core::samefit_grm::PackedArchive;
#[test]
fn mixed_anchor_every_jacobian_and_covariance_cell() {
    let mask = vec![true, false];
    let x = vec![1.4, 0.7, 0.2, -1.2, 0.4, -0.1, 1.8, 0.9, 0.3, 0.2, 0.4];
    let a = PackedArchive::new(mask.clone(), 2, 3, x.clone()).unwrap();
    let t = transform(&a).unwrap();
    let p = x.len();
    let n = t.rows.len();
    let mut cov = vec![0.; p * p];
    for i in 0..p {
        for j in 0..p {
            cov[i * p + j] = if i == j { 1.0 } else { 0.03 };
        }
    }
    let gpu = evaluate(&a, &cov).unwrap();
    assert_eq!(gpu.estimates.len(), n);
    assert_eq!(gpu.jacobian.len(), n * p);
    assert_eq!(gpu.covariance.len(), n * n);
    let mut fd = vec![0.; n * p];
    for k in 0..p {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[k] += 1e-5;
        minus[k] -= 1e-5;
        let tp = transform(&PackedArchive::new(mask.clone(), 2, 3, plus).unwrap()).unwrap();
        let tm = transform(&PackedArchive::new(mask.clone(), 2, 3, minus).unwrap()).unwrap();
        for i in 0..n {
            fd[i * p + k] = (tp.rows[i].estimate - tm.rows[i].estimate) / 2e-5;
            assert!((gpu.jacobian[i * p + k] - fd[i * p + k]).abs() < 2e-5);
        }
    }
    for i in 0..n {
        assert!((gpu.estimates[i] - t.rows[i].estimate).abs() < 2e-5);
        for j in 0..n {
            let mut expected = 0.;
            for k in 0..p {
                for l in 0..p {
                    expected += fd[i * p + k] * cov[k * p + l] * fd[j * p + l];
                }
            }
            assert!(
                (gpu.covariance[i * n + j] - expected).abs() / (1. + expected.abs()) < 2e-5,
                "cell {i},{j}"
            );
        }
    }
    assert_eq!(gpu.dispatches, 3);
}
