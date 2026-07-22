use super::*;

#[test]
fn empirical_reliability_tracks_signal_to_noise() {
    // wide score spread + small SEs -> high rho; flat scores -> low rho
    let n = 200usize;
    let eap: Vec<f64> = (0..n).map(|p| -2.0 + 4.0 * p as f64 / n as f64).collect();
    let sd_small = vec![0.3_f64; n];
    let sd_large = vec![1.5_f64; n];
    let hi = empirical_reliability(&eap, &sd_small, n, 1).unwrap()[0];
    let lo = empirical_reliability(&eap, &sd_large, n, 1).unwrap()[0];
    assert!(hi > 0.85, "high-information scale must be reliable: {hi}");
    assert!(
        lo < hi - 0.2,
        "noisier scale must be less reliable: {lo} vs {hi}"
    );
    assert!(empirical_reliability(&eap, &sd_small, 3, 1).is_err());
    assert!(empirical_reliability(&[], &[], 2, 0).is_err());
    assert!(empirical_reliability(&[0.0, f64::NAN], &[0.3, 0.3], 2, 1).is_err());
    assert!(empirical_reliability(&[0.0, 1.0], &[-0.3, 0.3], 2, 1).is_err());
    assert!(empirical_reliability(&[0.0, 1.0], &[0.3, f64::INFINITY], 2, 1).is_err());
}

#[test]
fn empirical_reliability_default_matches_cpu_reference() {
    let n_persons = 257usize;
    let n_dims = 3usize;
    let theta: Vec<f64> = (0..n_persons)
        .flat_map(|person| {
            let x = person as f64 / n_persons as f64;
            [x.sin(), 2.0 * x - 1.0, (3.0 * x).cos()]
        })
        .collect();
    let sd: Vec<f64> = (0..n_persons)
        .flat_map(|person| {
            let x = person as f64 / n_persons as f64;
            [0.2 + x / 10.0, 0.4, 0.1 + x / 20.0]
        })
        .collect();

    let reference =
        empirical_reliability_device(&theta, &sd, n_persons, n_dims, crate::Device::Cpu).unwrap();
    let default = empirical_reliability(&theta, &sd, n_persons, n_dims).unwrap();
    for (actual, expected) in default.iter().zip(&reference) {
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
#[test]
fn empirical_reliability_explicit_gpu_matches_cpu_reference() {
    let n_persons = 513usize;
    let n_dims = 2usize;
    let theta: Vec<f64> = (0..n_persons)
        .flat_map(|person| {
            let x = person as f64 / n_persons as f64;
            [4.0 * x - 2.0, (6.0 * x).sin()]
        })
        .collect();
    let sd: Vec<f64> = (0..n_persons)
        .flat_map(|person| {
            let x = person as f64 / n_persons as f64;
            [0.3 + 0.1 * x, 0.5 - 0.1 * x]
        })
        .collect();

    let cpu =
        empirical_reliability_device(&theta, &sd, n_persons, n_dims, crate::Device::Cpu).unwrap();
    let gpu = crate::gpu_scoring::empirical_reliability_gpu(&theta, &sd, n_persons, n_dims);
    if std::env::var("WGPU_BACKEND").is_ok_and(|backend| backend.eq_ignore_ascii_case("metal")) {
        assert!(
            gpu.is_some(),
            "WGPU_BACKEND=metal was explicit, but no usable Metal adapter was selected"
        );
    }
    let Some(gpu) = gpu else {
        eprintln!("no GPU adapter present; skipping empirical-reliability GPU parity check");
        return;
    };
    for (actual, expected) in gpu.iter().zip(&cpu) {
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }
}
