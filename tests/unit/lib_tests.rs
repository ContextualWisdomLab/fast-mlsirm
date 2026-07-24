use super::*;

fn config() -> ModelConfig {
    ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 2,
        model_type: ModelType::Mls2plm,
        eps_distance: 1e-8,
    }
}

fn params() -> Params {
    Params {
        theta: vec![0.2, -0.4],
        alpha: vec![0.1, -0.2],
        b: vec![0.3, -0.1],
        xi: vec![0.1, 0.2, -0.2, 0.4],
        zeta: vec![0.0, -0.1, 0.3, -0.4],
        tau: 0.2,
    }
}

#[test]
fn single_item_matches_manual_nll() {
    let cfg = ModelConfig {
        n_persons: 1,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mls2plm,
        eps_distance: 1e-8,
    };
    let p = Params {
        theta: vec![0.5],
        alpha: vec![0.0],
        b: vec![0.1],
        xi: vec![0.2],
        zeta: vec![-0.3],
        tau: 0.0,
    };
    let penalty = PenaltyConfig {
        lambda_theta: 0.0,
        lambda_xi: 0.0,
        lambda_zeta: 0.0,
        lambda_b: 0.0,
        lambda_alpha: 0.0,
        lambda_tau: 0.0,
        mu_alpha: 0.0,
        mu_tau: 0.0,
    };
    let (got, _, _) = neg_loglik_and_grad(&[1.0], None, &[0], &p, &cfg, &penalty);
    let r = ((0.2_f64 - -0.3_f64).powi(2) + 1e-8).sqrt();
    let eta = 0.5 + 0.1 - r;
    let expected = softplus(eta) - eta;
    assert!((got - expected).abs() < 1e-12);
}

#[test]
fn gradient_matches_finite_difference_for_tau() {
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];
    let (base, grad, _) = neg_loglik_and_grad(&y, None, &[0, 0], &p, &cfg, &penalty);

    let h = 1e-6;
    let mut plus = p.clone();
    plus.tau += h;
    let (obj_plus, _, _) = neg_loglik_and_grad(&y, None, &[0, 0], &plus, &cfg, &penalty);
    let finite_diff = (obj_plus - base) / h;
    assert!((finite_diff - grad.tau).abs() < 1e-5);
}

#[test]
fn mask_excludes_entries() {
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];
    let mask = vec![true, true, true, false];

    let (objective, grad, loglik) =
        neg_loglik_and_grad(&y, Some(&mask), &[0, 0], &p, &cfg, &penalty);

    assert!(objective.is_finite());
    assert!(loglik.is_finite());
    assert_eq!(grad.theta.len(), cfg.n_persons * cfg.n_dims);
}

#[test]
fn device_parse_accepts_known_names() {
    assert_eq!(Device::parse("cpu"), Some(Device::Cpu));
    assert_eq!(Device::parse("GPU"), Some(Device::Gpu));
    assert_eq!(Device::parse(" Auto "), Some(Device::Auto));
    assert_eq!(Device::parse("cuda"), None);
}

#[test]
fn device_auto_matches_cpu_on_fallback() {
    // On machines/CI without a GPU adapter, Auto/Gpu must fall back to the
    // CPU path and reproduce it bit-for-bit. When a GPU is present the f32
    // kernels are exercised instead and this asserts close (not exact)
    // agreement, which is the guarantee we make for the GPGPU path.
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];
    let mask = vec![true, true, true, false];

    let (cpu_obj, cpu_grad, cpu_ll) =
        neg_loglik_and_grad_device(Device::Cpu, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
    for device in [Device::Auto, Device::Gpu] {
        let (obj, grad, ll) =
            neg_loglik_and_grad_device(device, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
        assert!(
            (obj - cpu_obj).abs() < 1e-4,
            "objective mismatch for {device:?}"
        );
        assert!((ll - cpu_ll).abs() < 1e-4, "loglik mismatch for {device:?}");
        assert!((grad.tau - cpu_grad.tau).abs() < 1e-4);
        for (a, b) in grad.theta.iter().zip(&cpu_grad.theta) {
            assert!((a - b).abs() < 1e-4);
        }
        for (a, b) in grad.xi.iter().zip(&cpu_grad.xi) {
            assert!((a - b).abs() < 1e-4);
        }
        for (a, b) in grad.zeta.iter().zip(&cpu_grad.zeta) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}

#[test]
fn device_gpu_handles_absent_mask() {
    // Exercises the `mask: None` host-side path (dense all-observed matrix)
    // through the device entry point. On a GPU-equipped host this drives the
    // GPGPU kernels with the `None` mask branch; on a GPU-less host it is the
    // CPU fallback. Either way the device result must match the CPU reference
    // computed the same way.
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];

    let (cpu_obj, cpu_grad, cpu_ll) =
        neg_loglik_and_grad_device(Device::Cpu, &y, None, &[0, 0], &p, &cfg, &penalty);
    for device in [Device::Auto, Device::Gpu] {
        let (obj, grad, ll) =
            neg_loglik_and_grad_device(device, &y, None, &[0, 0], &p, &cfg, &penalty);
        assert!(
            (obj - cpu_obj).abs() < 1e-4,
            "objective mismatch for {device:?}"
        );
        assert!((ll - cpu_ll).abs() < 1e-4, "loglik mismatch for {device:?}");
        assert!((grad.tau - cpu_grad.tau).abs() < 1e-4);
        assert_eq!(grad.b.len(), cpu_grad.b.len());
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
#[test]
fn finish_device_prefers_gpu_result_when_present() {
    // When the GPU adapter produced a result, `finish_device` must return it
    // verbatim without invoking the CPU fallback, for both Gpu and Auto.
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];
    let sentinel = Gradients {
        theta: vec![1.0, 2.0],
        alpha: vec![3.0, 4.0],
        b: vec![5.0, 6.0],
        xi: vec![7.0, 8.0, 9.0, 10.0],
        zeta: vec![11.0, 12.0, 13.0, 14.0],
        tau: 42.0,
    };
    for device in [Device::Gpu, Device::Auto] {
        let (obj, grad, ll) = finish_device(
            device,
            Some((123.0, sentinel.clone(), -7.0)),
            &y,
            None,
            &[0, 0],
            &p,
            &cfg,
            &penalty,
        );
        assert_eq!(obj, 123.0);
        assert_eq!(ll, -7.0);
        assert_eq!(grad.tau, 42.0);
        assert_eq!(grad.b, vec![5.0, 6.0]);
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
#[test]
fn finish_device_falls_back_to_cpu_when_gpu_absent() {
    // When the GPU produced no result, `finish_device` must reproduce the CPU
    // reference. `Device::Gpu` additionally emits the fallback warning while
    // `Device::Auto` stays silent; both must return the CPU numbers.
    let cfg = config();
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];
    let mask = vec![true, true, true, false];
    let expected = neg_loglik_and_grad(&y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
    for device in [Device::Gpu, Device::Auto] {
        let (obj, grad, ll) =
            finish_device(device, None, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
        assert_eq!(obj, expected.0);
        assert_eq!(ll, expected.2);
        assert_eq!(grad.tau, expected.1.tau);
    }
}

#[test]
fn mirt_ignores_latent_space_terms() {
    let mut cfg = config();
    cfg.model_type = ModelType::Mirt;
    let p = params();
    let penalty = PenaltyConfig::default();
    let y = vec![1.0, 0.0, 0.0, 1.0];

    let (objective, grad, loglik) = neg_loglik_and_grad(&y, None, &[0, 0], &p, &cfg, &penalty);

    assert!(objective.is_finite());
    assert!(loglik.is_finite());
    assert_eq!(grad.tau, 0.0);
    assert!(grad.xi.iter().all(|value| *value == 0.0));
    assert!(grad.zeta.iter().all(|value| *value == 0.0));
}
