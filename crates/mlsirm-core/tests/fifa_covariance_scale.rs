#![cfg(feature = "gpu")]
use mlsirm_core::samefit_fifa::gpu::evaluate;
use mlsirm_core::samefit_grm::PackedArchive;

fn archive() -> PackedArchive {
    PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., 0.]).unwrap()
}
fn matrix(scale: f64, off: f64) -> Vec<f64> {
    vec![
        scale,
        off * scale,
        0.,
        0.,
        off * scale,
        scale,
        0.,
        0.,
        0.,
        0.,
        scale,
        0.,
        0.,
        0.,
        0.,
        scale,
    ]
}
fn rounded_indefinite(scale: f64) -> Vec<f64> {
    let mut covariance = matrix(1.0, 1.0 + 7e-8);
    covariance[5] = covariance[1] * covariance[1] + 1e-12;
    covariance.iter().map(|v| v * scale).collect()
}
#[test]
fn rounded_covariance_is_rejected_across_scales() {
    for exponent in [-68, -100, -48, 0, 48, 100] {
        let covariance = rounded_indefinite(2.0_f64.powi(exponent));
        // The f64 2x2 determinant is positive; the actual f32 rounding is indefinite.
        assert!(covariance[0] * covariance[5] - covariance[1] * covariance[4] > 0.0);
        let rounded: Vec<f64> = covariance.iter().map(|v| f64::from(*v as f32)).collect();
        assert!(rounded[0] * rounded[5] - rounded[1] * rounded[4] < 0.0);
        match evaluate(&archive(), &covariance) {
            Err(error) => assert_eq!(
                error, "nonpositive f32 parameter covariance",
                "2^{exponent}"
            ),
            Ok(output) => panic!(
                "2^{exponent}: f32-indefinite covariance admitted; device={} dispatches={}",
                output.device, output.dispatches
            ),
        }
    }
}
#[test]
fn positive_covariance_preserves_gpu_variances_across_scales() {
    for exponent in [-100, -68, -48, 0, 48, 100] {
        let scale = 2.0_f64.powi(exponent);
        let output = evaluate(&archive(), &matrix(scale, 0.25)).unwrap();
        assert_eq!(output.dispatches, 3);
        let n = output.estimates.len();
        assert!(output.covariance.iter().all(|v| v.is_finite()));
        for i in 0..n {
            assert!(output.covariance[i * n + i] >= 0.0);
        }
        assert!((output.covariance[5 * n + 5] / scale - 1.5).abs() < 2e-5);
    }
}
#[test]
fn singular_covariance_is_rejected_across_scales() {
    for exponent in [-100, -68, 0, 100] {
        match evaluate(&archive(), &matrix(2.0_f64.powi(exponent), 1.0)) {
            Err(error) => assert_eq!(error, "nonpositive parameter covariance"),
            Ok(_) => panic!("singular covariance admitted"),
        }
    }
}
#[test]
fn indefinite_covariance_is_rejected_across_scales() {
    for exponent in [-68, -100, -48, 0, 48, 100] {
        let scale = 2.0_f64.powi(exponent);
        let result = evaluate(&archive(), &matrix(scale, 2.));
        match result {
            Err(error) => assert_eq!(error, "nonpositive parameter covariance", "2^{exponent}"),
            Ok(output) => panic!("2^{exponent}: indefinite covariance admitted; device={} backend={} dispatches={} difficulty_variance={}", output.device, output.backend, output.dispatches, output.covariance[5 * output.estimates.len() + 5]),
        }
    }
}
