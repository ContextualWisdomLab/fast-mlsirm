use super::*;
use crate::nodes::XiRule;

#[test]
fn gpu_eap_matches_cpu_reduction() {
    let (n_items, n_persons, latent_dim) = (6usize, 40usize, 1usize);
    let alpha: Vec<f64> = (0..n_items).map(|i| 0.1 * i as f64 - 0.2).collect();
    let b: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.2 * i as f64).collect();
    let zeta: Vec<f64> = (0..n_items * latent_dim)
        .map(|i| 0.3 * (i % 3) as f64 - 0.3)
        .collect();
    let fid = vec![0usize; n_items];
    let mut st = 12345u64;
    let mut u = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut y = vec![0.0_f64; n_persons * n_items];
    for v in y.iter_mut() {
        *v = if u() < 0.5 { 1.0 } else { 0.0 };
    }
    let observed = vec![true; n_persons * n_items];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -0.3,
        factor_id: &fid,
        model_type: crate::ModelType::Mls2plm,
        n_dims: 1,
        latent_dim,
        eps_distance: 1e-8,
    };
    let prior = PriorSpec::standard(1);
    let grids = scoring_grids(&bank, 21, XiRule::GaussHermite { q_xi: 11 }).unwrap();
    let ctx = prior_contexts(&prior);
    let config = bank_model_config(&bank, n_persons, n_items);
    let tables = build_tables(
        bank.alpha,
        bank.b,
        bank.zeta,
        bank.tau,
        &config,
        bank.factor_id,
        &ctx,
        &grids,
    );
    let resp = index_responses(&y, &observed, n_persons, n_items);
    let cpu = score_eap_cpu_reduce(&bank, &prior, &grids, &tables, &resp, n_persons, n_items);
    let gpu = try_score_eap_gpu(&bank, &prior, &grids, &tables, &resp, n_persons, n_items);
    if std::env::var("WGPU_BACKEND").is_ok_and(|backend| backend.eq_ignore_ascii_case("metal")) {
        assert!(
            gpu.is_some(),
            "WGPU_BACKEND=metal was explicit, but no usable Metal adapter was selected"
        );
    }
    match gpu {
        None => eprintln!("no GPU adapter present; skipping GPU EAP parity check"),
        Some(gpu) => {
            let mut max_abs = [0.0_f64; 4];
            for p in 0..n_persons {
                max_abs[0] = max_abs[0].max((gpu.loglik[p] - cpu.loglik[p]).abs());
                max_abs[1] = max_abs[1].max((gpu.theta_eap[p] - cpu.theta_eap[p]).abs());
                max_abs[2] = max_abs[2].max((gpu.theta_sd[p] - cpu.theta_sd[p]).abs());
                max_abs[3] = max_abs[3].max((gpu.xi_eap[p] - cpu.xi_eap[p]).abs());
                assert!(
                    (gpu.loglik[p] - cpu.loglik[p]).abs() < 2e-3,
                    "loglik p={p}: gpu {} vs cpu {}",
                    gpu.loglik[p],
                    cpu.loglik[p]
                );
                assert!((gpu.theta_eap[p] - cpu.theta_eap[p]).abs() < 2e-3);
                assert!((gpu.theta_sd[p] - cpu.theta_sd[p]).abs() < 2e-3);
                assert!((gpu.xi_eap[p] - cpu.xi_eap[p]).abs() < 2e-3);
            }
            eprintln!(
                "GPU EAP parity max abs: loglik={:.3e}, theta={:.3e}, theta_sd={:.3e}, xi={:.3e}; tolerance=2e-3",
                max_abs[0], max_abs[1], max_abs[2], max_abs[3]
            );
        }
    }
}

#[test]
fn gpu_bank_information_matches_cpu_reduction() {
    let n_items = 8usize;
    let n_points = 5usize;
    let n_dims = 2usize;
    let latent_dim = 2usize;
    let alpha = vec![0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.25];
    let b = vec![0.5, -0.5, 0.0, 1.0, -1.0, 0.3, -0.3, 0.8];
    let zeta = vec![
        -0.4, 0.2, 0.1, -0.2, 0.3, 0.5, -0.1, 0.4, 0.6, -0.3, -0.2, -0.5, 0.2, 0.1, -0.5, 0.3,
    ];
    let factor_id = vec![0, 1, 0, 1, 0, 1, 0, 1];
    let theta = vec![-1.2, 0.7, -0.5, 0.3, 0.0, 0.0, 0.6, -0.4, 1.1, -0.8];
    let xi = vec![-0.7, 0.2, -0.2, 0.6, 0.0, 0.0, 0.4, -0.5, 0.8, 0.3];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -0.25,
        factor_id: &factor_id,
        model_type: crate::ModelType::Mls2plm,
        n_dims,
        latent_dim,
        eps_distance: 1e-8,
    };
    let cpu = bank_information_cpu_reduce(&bank, &theta, &xi, n_points, n_items);
    let gpu = try_bank_information_gpu(&bank, &theta, &xi, n_points, n_items);
    if std::env::var("WGPU_BACKEND").is_ok_and(|backend| backend.eq_ignore_ascii_case("metal")) {
        assert!(
            gpu.is_some(),
            "WGPU_BACKEND=metal was explicit, but no usable Metal adapter was selected"
        );
    }
    match gpu {
        None => eprintln!("no GPU adapter present; skipping GPU bank-information parity check"),
        Some((gpu_item, gpu_test)) => {
            let max_item = gpu_item
                .iter()
                .zip(&cpu.0)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0_f64, f64::max);
            let max_test = gpu_test
                .iter()
                .zip(&cpu.1)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_item < 2e-4, "item information max abs={max_item}");
            assert!(max_test < 5e-4, "test information max abs={max_test}");
            eprintln!(
                "GPU bank-information parity max abs: item={max_item:.3e}, test={max_test:.3e}; tolerances=2e-4/5e-4"
            );
        }
    }

    // A log discrimination of 100 is valid in the f64 serving contract but
    // exp(100) overflows f32. The GPU attempt must be discarded instead of
    // leaking NaN information, and the public device path must return the
    // finite CPU reference.
    let extreme_alpha = vec![100.0; n_items];
    let extreme_bank = ItemBank {
        alpha: &extreme_alpha,
        b: &b,
        zeta: &zeta,
        tau: -0.25,
        factor_id: &factor_id,
        model_type: crate::ModelType::Mls2plm,
        n_dims,
        latent_dim,
        eps_distance: 1e-8,
    };
    let extreme_cpu = bank_information_cpu_reduce(&extreme_bank, &theta, &xi, n_points, n_items);
    assert!(extreme_cpu
        .0
        .iter()
        .chain(&extreme_cpu.1)
        .all(|value| value.is_finite()));
    assert!(
        try_bank_information_gpu(&extreme_bank, &theta, &xi, n_points, n_items).is_none(),
        "non-finite f32 information must trigger CPU fallback"
    );
    let extreme_device =
        bank_information_device(&extreme_bank, &theta, &xi, n_points, crate::Device::Gpu).unwrap();
    assert_eq!(extreme_device, extreme_cpu);
}
