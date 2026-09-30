use crate::two_tier_recursion::{
    two_tier_expected_raw, two_tier_expected_raw_on, TwoTierItemParams,
};

#[test]
fn unsafe_f32_predictors_use_the_exact_cpu_result() {
    // First case loses a non-representable slope; second loses the central
    // unit during a dot product even though every input is exactly f32.
    for slopes in [
        vec![16_777_217.0, -16_777_216.0],
        vec![16_777_216.0, 1.0, -16_777_216.0],
    ] {
        let params = TwoTierItemParams {
            n_primary: slopes.len(),
            a_primary: slopes,
            a_specific: vec![0.0],
            thresholds: vec![0.0],
            specific_map: vec![-1],
            n_specific: 0,
            n_cat: 2,
        };
        let theta = vec![1.0; params.n_primary];
        let cpu = two_tier_expected_raw(&params, &theta, &[0.0], &[1.0]).unwrap();
        assert!((cpu[0] - 0.7310585786300049).abs() < 1e-15);
        for device in [crate::Device::Gpu, crate::Device::Auto] {
            let (actual, used_gpu) =
                two_tier_expected_raw_on(&params, &theta, &[0.0], &[1.0], device).unwrap();
            assert!(!used_gpu, "unsafe predictors must fall back to f64");
            assert_eq!(actual, cpu);
        }
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
#[test]
fn precision_admission_checks_thresholds_and_specific_terms() {
    use crate::gpu_two_tier::predictor_precision_is_safe;
    let mut params = TwoTierItemParams {
        a_primary: vec![1.0],
        a_specific: vec![0.5],
        thresholds: vec![0.0],
        specific_map: vec![0],
        n_primary: 1,
        n_specific: 1,
        n_cat: 2,
    };
    assert!(predictor_precision_is_safe(&params, &[0.0], &[0.0]));
    params.thresholds[0] = 16_777_217.0;
    assert!(!predictor_precision_is_safe(&params, &[0.0], &[0.0]));
    params.thresholds[0] = 0.0;
    params.a_specific[0] = 16_777_217.0;
    assert!(!predictor_precision_is_safe(&params, &[0.0], &[1.0]));
    params.a_specific[0] = 0.0;
    params.a_primary[0] = f64::MAX;
    assert!(!predictor_precision_is_safe(&params, &[2.0], &[0.0]));
}

#[test]
fn specific_and_threshold_cancellation_fall_back() {
    for (a, specific, threshold, node) in [
        (16_777_216.0, 1.0, -16_777_216.0, 1.0),
        (16_777_217.0, 0.0, -16_777_216.0, 0.0),
    ] {
        let params = TwoTierItemParams {
            a_primary: vec![a],
            a_specific: vec![specific],
            thresholds: vec![threshold],
            specific_map: vec![0],
            n_primary: 1,
            n_specific: 1,
            n_cat: 2,
        };
        let cpu = two_tier_expected_raw(&params, &[1.0], &[node], &[1.0]).unwrap();
        assert!((cpu[0] - 0.7310585786300049).abs() < 1e-15);
        let (actual, used_gpu) =
            two_tier_expected_raw_on(&params, &[1.0], &[node], &[1.0], crate::Device::Auto)
                .unwrap();
        assert!(!used_gpu);
        assert_eq!(actual, cpu);
    }
}
