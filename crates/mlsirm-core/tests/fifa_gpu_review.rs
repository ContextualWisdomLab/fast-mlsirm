#![cfg(feature = "gpu")]
use mlsirm_core::samefit_fifa::gpu::evaluate;
use mlsirm_core::samefit_grm::PackedArchive;

fn identity(p: usize) -> Vec<f64> {
    let mut cov = vec![0.; p * p];
    for i in 0..p {
        cov[i * p + i] = 1.;
    }
    cov
}
#[test]
fn rejects_indefinite_covariance() {
    let a = PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., 0.]).unwrap();
    let mut cov = identity(4);
    cov[0] = -1.;
    assert!(
        evaluate(&a, &cov).is_err(),
        "indefinite covariance must not propagate"
    );
}
#[test]
fn rejects_finite_but_inaccurate_f32_transform() {
    let a = PackedArchive::new(vec![true], 2, 3, vec![1., 1e10, 0., 0., 0.]).unwrap();
    let result = evaluate(&a, &identity(5));
    assert!(result.is_err(), "distinct intercepts collapse in f32");
}
#[test]
fn rejects_f32_scale_underflow() {
    let a = PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., -120.]).unwrap();
    assert!(
        evaluate(&a, &identity(4)).is_err(),
        "f32 SD underflows to zero"
    );
}

mod independent_boundaries {
    use mlsirm_core::samefit_fifa::{gpu::evaluate, infer, transform, Row, Transformation};
    use mlsirm_core::samefit_grm::PackedArchive;

    fn identity(p: usize) -> Vec<f64> {
        let mut v = vec![0.; p * p];
        for k in 0..p {
            v[k * p + k] = 1.;
        }
        v
    }
    fn public_t(estimate: f64, gradient: Vec<f64>) -> Transformation {
        Transformation {
            rows: vec![Row {
                name: "boundary".into(),
                estimate,
            }],
            n_parameters: gradient.len(),
            jacobian: gradient,
        }
    }
    #[test]
    fn cpu_finite_and_overflow_refusal_controls() {
        let good = infer(&public_t(3., vec![1., 2.]), &[2., 1., 1., 2.]).unwrap();
        assert!((good.rows[0].se - 2f64.sqrt()).abs() < 1e-12);
        assert!(good.transformed_covariance.iter().all(|v| v.is_finite()));
        let huge = infer(&public_t(0., vec![1e200]), &[1.]);
        println!("variance_overflow={huge:?}");
        assert!(huge.is_err());
        let offdiag = infer(&public_t(0., vec![1., -1.]), &[2., 1., 1., 2.]).unwrap();
        assert!(offdiag.transformed_covariance.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn cpu_extreme_upper_ci_is_finite() {
        let result = infer(&public_t(f64::MAX, vec![1e154]), &[1.]).unwrap();
        println!("upper_extreme={result:?}");
        assert!(result.rows[0].ci95_upper.is_finite());
        assert!(result.rows[0].ci95_lower.is_finite());
    }
    #[test]
    fn cpu_extreme_lower_ci_is_finite() {
        let result = infer(&public_t(-f64::MAX, vec![1e154]), &[1.]).unwrap();
        println!("lower_extreme={result:?}");
        assert!(result.rows[0].ci95_lower.is_finite());
        assert!(result.rows[0].ci95_upper.is_finite());
    }
    #[test]
    fn cpu_empty_rows_nonfinite_covariance_must_reject() {
        let t = Transformation {
            rows: vec![],
            jacobian: vec![],
            n_parameters: 2,
        };
        let result = infer(&t, &[1e-309, 0., 0., 1e-309]);
        println!("empty_nonfinite_covariance={result:?}");
        assert!(
            result.is_err(),
            "nonfinite inverse covariance must fail even for no display rows"
        );
    }
    #[test]
    fn gpu_f32_rounded_indefinite_covariance_must_reject() {
        let a = PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., 0.]).unwrap();
        let mut c = identity(4);
        c[1] = 1. + 7e-8;
        c[4] = c[1];
        c[5] = c[1] * c[1] + 1e-12;
        let pd = mlsirm_core::inference::second_order_test(&c, 4, 0.).unwrap();
        assert!(pd.0);
        let cast: Vec<_> = c.iter().map(|&x| (x as f32) as f64).collect();
        let cast_pd = mlsirm_core::inference::second_order_test(&cast, 4, 0.).unwrap();
        println!("f64_pd={pd:?} rounded_pd={cast_pd:?}");
        let result = evaluate(&a, &c);
        match &result {
            Ok(o) => println!(
                "rounded_covariance device={} backend={} dispatches={} diag={:?}",
                o.device,
                o.backend,
                o.dispatches,
                (0..o.estimates.len())
                    .map(|i| o.covariance[i * o.estimates.len() + i])
                    .collect::<Vec<_>>()
            ),
            Err(e) => println!("rounded_covariance Err={e}"),
        }
        assert!(
            result.is_err(),
            "an f32-indefinite covariance must not be propagated as valid output"
        );
    }
    #[test]
    fn gpu_covariance_cast_underflow_must_reject() {
        let a = PackedArchive::new(vec![true], 2, 2, vec![1., 1., 0., 0.]).unwrap();
        let mut c = identity(4);
        c[0] = 1e-50;
        let result = evaluate(&a, &c);
        match &result {
            Ok(o) => println!(
                "cast_underflow device={} backend={} dispatches={} slope_variance={:?}",
                o.device,
                o.backend,
                o.dispatches,
                o.covariance[2 * o.estimates.len() + 2]
            ),
            Err(e) => println!("cast_underflow Err={e}"),
        }
        assert!(
            result.is_err(),
            "positive f64 variance lost to zero in f32 must be refused"
        );
    }
    #[test]
    fn gpu_large_slope_norm_overflow_observation() {
        for slope in [1e10, 1e15, 1e20, 1e30] {
            let a = PackedArchive::new(vec![true], 2, 2, vec![slope, 0., 0., 0.]).unwrap();
            let cpu = transform(&a).unwrap();
            let result = evaluate(&a, &identity(4));
            match result {
                Ok(o) => {
                    let idx = cpu
                        .rows
                        .iter()
                        .position(|r| r.name == "g0.i0.lambda_logit_latent_response")
                        .unwrap();
                    println!("large_slope={slope} device={} backend={} dispatches={} lambda_cpu={} lambda_gpu={}",o.device,o.backend,o.dispatches,cpu.rows[idx].estimate,o.estimates[idx]);
                    assert!(
                        (cpu.rows[idx].estimate - o.estimates[idx]).abs() < 2e-5,
                        "overflowed normalization must refuse or preserve the bounded loading"
                    );
                }
                Err(e) => println!("large_slope={slope} Err={e}"),
            }
        }
    }
    #[test]
    fn gpu_invalid_input_reasons_are_not_device_absence() {
        for (x, reason) in [
            (
                vec![1., 1., 0., -120.],
                "f32 exponential outside normal range",
            ),
            (vec![1., 1e10, 0., 0., 0.], "f32 intercept order collapsed"),
        ] {
            let categories = if x.len() == 5 { 3 } else { 2 };
            let p = x.len();
            let a = PackedArchive::new(vec![true], 2, categories, x).unwrap();
            let err = evaluate(&a, &identity(p)).err().unwrap();
            assert_eq!(err, reason);
        }
    }
}
