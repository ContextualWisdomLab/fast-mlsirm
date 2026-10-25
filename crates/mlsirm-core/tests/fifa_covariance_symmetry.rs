#![cfg(feature = "gpu")]
use mlsirm_core::samefit_fifa::gpu::evaluate;
use mlsirm_core::samefit_grm::PackedArchive;

fn archive() -> PackedArchive {
    PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., 0.]).unwrap()
}

fn covariance(scale: f64, asymmetry: f64) -> Vec<f64> {
    let mut c = vec![0.; 16];
    for k in 0..4 {
        c[k * 4 + k] = scale;
    }
    c[1] = 0.25 * scale;
    c[4] = (0.25 + asymmetry) * scale;
    c
}

fn reject_asymmetry(exponents: &[i32]) {
    for &exponent in exponents {
        for transpose in [false, true] {
            let mut c = covariance(2.0_f64.powi(exponent), 0.04);
            if transpose {
                c.swap(1, 4);
            }
            let original = c.clone();
            match evaluate(&archive(), &c) {
                Err(error) => assert_eq!(error, "asymmetric parameter covariance", "2^{exponent}"),
                Ok(output) => panic!(
                    "2^{exponent}: asymmetric covariance admitted; device={} dispatches={}",
                    output.device, output.dispatches
                ),
            }
            assert_eq!(c, original, "caller covariance must remain unchanged");
        }
    }
}

#[test]
fn tiny_asymmetric_covariance_is_rejected_before_gpu() {
    reject_asymmetry(&[-100, -68, -48]);
}

#[test]
fn large_asymmetric_covariance_is_rejected_before_gpu() {
    reject_asymmetry(&[0, 48, 100]);
}

#[test]
fn just_over_relative_symmetry_tolerance_is_rejected_at_every_scale() {
    for exponent in [-100, -68, -48, 0, 48, 100] {
        let c = covariance(2.0_f64.powi(exponent), 1.01e-7);
        assert_eq!(
            evaluate(&archive(), &c).err().unwrap(),
            "asymmetric parameter covariance"
        );
    }
}

#[path = "support/fifa_hardware.rs"]
mod hardware;

#[test]
fn hardware_cases_require_explicit_selection() {
    hardware::assert_hardware_cases_are_ignored(&[
        "tiny_roundoff_uses_the_checked_symmetric_f32_matrix",
        "large_roundoff_uses_the_checked_symmetric_f32_matrix",
    ]);
}

fn canonical_roundoff(exponents: &[i32]) {
    for &exponent in exponents {
        let scale = 2.0_f64.powi(exponent);
        let c = covariance(scale, 4e-8);
        let original = c.clone();
        let mut canonical = c.clone();
        let mean = 0.5 * c[1] + 0.5 * c[4];
        canonical[1] = mean;
        canonical[4] = mean;
        // This fixture distinguishes the original f32 upload from its mean.
        assert_ne!((c[1] as f32).to_bits(), (c[4] as f32).to_bits());
        let expected = evaluate(&archive(), &canonical).expect("physical GPU canonical positive");
        for transpose in [false, true] {
            let mut input = c.clone();
            if transpose {
                input.swap(1, 4);
            }
            let actual = evaluate(&archive(), &input).expect("accepted relative roundoff");
            assert_eq!(actual.dispatches, 3);
            assert_eq!(actual.estimates, expected.estimates);
            assert_eq!(actual.jacobian, expected.jacobian);
            let n = actual.estimates.len();
            // These two rows retain the packed slope and first intercept, so
            // their covariance cells expose the actual f32 upload unchanged.
            assert_eq!(actual.covariance[2 * n + 4], f64::from(mean as f32));
            assert_eq!(actual.covariance[4 * n + 2], f64::from(mean as f32));
            assert_eq!(
                actual.covariance, expected.covariance,
                "2^{exponent}: the checked mean must be the f32 matrix uploaded"
            );
        }
        assert_eq!(c, original);
        println!(
            "FIFA_SYMMETRY scale=2^{exponent} device={} backend={} dispatches={}",
            expected.device, expected.backend, expected.dispatches
        );
    }
}

#[test]
#[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
fn tiny_roundoff_uses_the_checked_symmetric_f32_matrix() {
    canonical_roundoff(&[-100, -68, -48]);
}

#[test]
#[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
fn large_roundoff_uses_the_checked_symmetric_f32_matrix() {
    canonical_roundoff(&[0, 48, 100]);
}
