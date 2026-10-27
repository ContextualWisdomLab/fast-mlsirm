//! Synthetic execution observers, never numerical substitutes.
#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn scoring_completion_kernel_parity_and_refusals() {
    leased();
    use crate::gpu_bifactor_scoring::{eap, tables};
    let pars = vec![vec![
        ItemParams {
            a_g: -0.7,
            a_s: Some(0.4),
            d: vec![1.2, 0.1, -1.3],
        },
        ItemParams {
            a_g: 1.1,
            a_s: None,
            d: vec![1.0, 0.1, -1.2],
        },
    ]];
    let tg = vec![vec![-1.0, 0.2, 1.7]];
    let ts = vec![vec![vec![-0.8, 0.6]]];
    let block = [Some(0), None];
    let actual = tables(&pars, &block, &tg, &ts, 4).unwrap();
    for i in 0..2 {
        for t in 0..3 {
            for h in 0..if i == 0 { 2 } else { 1 } {
                let eta = pars[0][i].a_g * tg[0][t] + if i == 0 { 0.4 * ts[0][0][h] } else { 0.0 };
                let expected = grm_logprobs(eta, &pars[0][i].d);
                let offset = (t * if i == 0 { 2 } else { 1 } + h) * 4;
                for k in 0..4 {
                    assert!(
                        (actual[0][i][offset + k] - expected[k]).abs() < 1e-9,
                        "native logprob parity item={i} node={t}/{h} category={k} error={}",
                        (actual[0][i][offset + k] - expected[k]).abs()
                    );
                }
                let mass: f64 = actual[0][i][offset..offset + 4]
                    .iter()
                    .map(|x| x.exp())
                    .sum();
                assert!((mass - 1.0).abs() < 1e-9);
            }
        }
    }
    let post = [0.2, 0.3, 0.5, 0.0, 1.0, 0.0];
    let scores = eap(&post, &[0, 0], &tg).unwrap();
    for p in 0..2 {
        let row = &post[p * 3..p * 3 + 3];
        let mean: f64 = row.iter().zip(&tg[0]).map(|(w, x)| w * x).sum();
        let second: f64 = row.iter().zip(&tg[0]).map(|(w, x)| w * x * x).sum();
        assert!((scores[p][0] - mean).abs() < 1e-9);
        assert!((scores[p][1] - (second - mean * mean).max(0.0).sqrt()).abs() < 1e-9);
    }
    assert!(eap(&[0.0, 0.0, 0.0], &[0], &tg).is_err());
    assert!(eap(&[-1.0, 1.0, 1.0], &[0], &tg).is_err());
    assert!(eap(&post, &[0, 2], &tg).is_err());
    assert!(eap(&post[..2], &[0], &tg).is_err());
    assert!(eap(&[1.0], &[0], &[vec![f64::NAN]]).is_err());
    let mut bad = pars.clone();
    bad[0][0].d[1] = bad[0][0].d[0];
    assert!(tables(&bad, &block, &tg, &ts, 4).is_err());
    assert!(tables(&pars, &[Some(2), None], &tg, &ts, 4).is_err());
    assert!(tables(&pars, &block, &[vec![f64::NAN]], &ts, 4).is_err());
    assert!(
        tables(&pars, &block, &[vec![1000.0]], &ts, 4).is_err(),
        "device arithmetic-range flag must reject"
    );
    let r =
        crate::gpu_group_update::update_selected(&[[2.0, 4.0, 0.0]], &[true], &[false]).unwrap();
    assert_eq!(r, vec![[2.0, 1.0]]);
    assert!(
        crate::gpu_group_update::update_selected(&[[0.0, 1.0, 2.0]], &[true], &[false]).is_err()
    );
    assert!(crate::gpu_group_update::update_selected(&[[1.0, 1.0, 2.0]], &[], &[false]).is_err());
    assert!(
        crate::gpu_group_update::update_selected(&[[1e-300, 1.0, 2.0]], &[true], &[false]).is_err()
    );
}

#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn scoring_precision_original_input_narrow_probabilities() {
    leased();
    for upper in [1.0_f64] {
        for gap in [1e-7, 1e-8] {
            for neighbor in [-1i64, 0, 1] {
                let middle = upper - gap;
                let lower = f64::from_bits((middle.to_bits() as i64 + neighbor) as u64);
                let params = vec![vec![ItemParams {
                    a_g: 1.1,
                    a_s: None,
                    d: vec![upper, lower, -2.0],
                }]];
                let actual = crate::gpu_bifactor_scoring::tables(
                    &params,
                    &[None],
                    &[vec![-1.0]],
                    &[vec![]],
                    4,
                )
                .unwrap();
                let expected = grm_logprobs(-1.1, &params[0][0].d);
                let mut mass = 0.0;
                for k in 0..4 {
                    let log_error = (actual[0][0][k] - expected[k]).abs();
                    let probability = actual[0][0][k].exp();
                    let probability_error = (probability - expected[k].exp()).abs();
                    mass += probability;
                    eprintln!("NARROW_ORIGINAL upper={upper} gap={gap:e} neighbor={neighbor} category={k} log_error={log_error:e} probability_error={probability_error:e}");
                    assert!(
                        log_error < 1e-9,
                        "original-input narrow log probability error={log_error:e}"
                    );
                    assert!(probability_error < 1e-9);
                }
                assert!((mass - 1.0).abs() < 1e-9);
            }
        }
    }
}

#[test]
#[ignore = "native-f64 logit-rounding counterexample; unchanged strict NONPASS retained"]
fn scoring_precision_negative_logit_rounding_nonpass() {
    leased();
    let params = vec![vec![ItemParams {
        a_g: 1.1,
        a_s: None,
        d: vec![-1.0, -1.0 - 1e-7, -2.0],
    }]];
    let actual =
        crate::gpu_bifactor_scoring::tables(&params, &[None], &[vec![-1.0]], &[vec![]], 4).unwrap();
    let expected = grm_logprobs(-1.1, &params[0][0].d);
    let error = (actual[0][0][1] - expected[1]).abs();
    assert!(error < 1e-9, "native rounded-logit gap error={error:e}");
}

#[test]
#[ignore = "strict native-f64 narrow-gap oracle; known NONPASS retained"]
fn scoring_completion_narrow_native_logprob_nonpass() {
    leased();
    let params = vec![vec![ItemParams {
        a_g: 1.1,
        a_s: None,
        d: vec![1.0, 1.0 - 1e-8, -1.2],
    }]];
    let actual =
        crate::gpu_bifactor_scoring::tables(&params, &[None], &[vec![-1.0]], &[vec![]], 4).unwrap();
    let expected = grm_logprobs(-1.1, &params[0][0].d);
    assert!(
        (actual[0][0][1] - expected[1]).abs() < 1e-9,
        "strict native narrow category parity error={}",
        (actual[0][0][1] - expected[1]).abs()
    );
}

#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn scoring_completion_asymmetric_high_grid_native_parity() {
    leased();
    for (qg, qs) in [(121, 241), (241, 121)] {
        let (x, w) = gh_rule(qg).unwrap();
        let (z, _) = gh_rule(qs).unwrap();
        let params = vec![vec![ItemParams {
            a_g: 0.35,
            a_s: Some(-0.25),
            d: vec![1.1, 0.1, -1.0],
        }]];
        let actual = crate::gpu_bifactor_scoring::tables(
            &params,
            &[Some(0)],
            &[x.to_vec()],
            &[vec![z.to_vec()]],
            4,
        )
        .unwrap();
        let mut max_log_error = 0.0_f64;
        for t in 0..qg {
            for h in 0..qs {
                let expected = grm_logprobs(0.35 * x[t] - 0.25 * z[h], &params[0][0].d);
                for k in 0..4 {
                    max_log_error =
                        max_log_error.max((actual[0][0][(t * qs + h) * 4 + k] - expected[k]).abs());
                }
            }
        }
        assert!(
            max_log_error < 1e-9,
            "high-grid native table parity error={max_log_error}"
        );
        // Production reduced posterior is f32; use that exact transport here.
        let post: Vec<f64> = w.iter().map(|w| *w as f32 as f64).collect();
        let score = crate::gpu_bifactor_scoring::eap(&post, &[0], &[x.to_vec()]).unwrap();
        let mass: f64 = post.iter().sum();
        let mean: f64 = post.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() / mass;
        let second: f64 = post.iter().zip(x).map(|(w, x)| w * x * x).sum::<f64>() / mass;
        assert!(
            (score[0][0] - mean).abs() < 1e-9
                && (score[0][1] - (second - mean * mean).sqrt()).abs() < 1e-9
        );
        eprintln!("asymmetric-grid {qg}/{qs}: native table max_abs_error={max_log_error:e}; EAP parity passed");
    }
}

use super::*;
use std::cell::Cell;
thread_local! {
    static HOST_TABLE_CELLS: Cell<usize> = const { Cell::new(0) };
    static HOST_EAP_ROWS: Cell<usize> = const { Cell::new(0) };
    static DEVICE_TABLE_CALLS: Cell<usize> = const { Cell::new(0) };
    static DEVICE_EAP_CALLS: Cell<usize> = const { Cell::new(0) };
    static DEVICE_MEAN_ONLY_ROWS: Cell<usize> = const { Cell::new(0) };
}
pub(super) fn host_table() {
    HOST_TABLE_CELLS.with(|x| x.set(x.get() + 1));
}
pub(super) fn host_eap() {
    HOST_EAP_ROWS.with(|x| x.set(x.get() + 1));
}
pub(crate) fn device_table() {
    DEVICE_TABLE_CALLS.with(|x| x.set(x.get() + 1));
}
pub(crate) fn device_eap() {
    DEVICE_EAP_CALLS.with(|x| x.set(x.get() + 1));
}
pub(crate) fn device_mean_only(n: usize) {
    DEVICE_MEAN_ONLY_ROWS.with(|x| x.set(x.get() + n));
}
fn reset() {
    for key in [
        &HOST_TABLE_CELLS,
        &HOST_EAP_ROWS,
        &DEVICE_TABLE_CALLS,
        &DEVICE_EAP_CALLS,
        &DEVICE_MEAN_ONLY_ROWS,
    ] {
        key.with(|x| x.set(0));
    }
}
fn leased() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
}
fn fixture() -> (
    Vec<usize>,
    Vec<bool>,
    Vec<usize>,
    Vec<i32>,
    BifactorMultigroupConfig,
) {
    let y: Vec<usize> = (0..12)
        .flat_map(|p| (0..7).map(move |i| (p + i) % 3))
        .collect();
    let mut obs = vec![true; y.len()];
    obs[11 * 7 + 2] = false;
    obs[11 * 7 + 3] = false;
    obs[8 * 7] = false;
    let cfg = BifactorMultigroupConfig {
        q_general: 3,
        q_specific: 3,
        max_iter: 1,
        newton_iter: 1,
        estimate_specific_means: true,
        estimate_specific_vars: false,
        seed: 812,
        n_starts: 1,
        ..Default::default()
    };
    (
        y,
        obs,
        (0..12).map(|p| usize::from(p >= 4)).collect(),
        vec![0, 0, 1, 1, 2, 2, -1],
        cfg,
    )
}
#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn scoring_completion_multigroup_tables_and_eap_are_device_consumed() {
    leased();
    reset();
    let y = vec![0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0];
    let groups = [0, 0, 0, 0, 1, 1, 1, 1];
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 1,
        max_iter: 1,
        newton_iter: 20,
        device: crate::Device::Gpu,
        ..Default::default()
    };
    let gpu = fit_bifactor_grm_multigroup(&y, None, &groups, 2, &[-1, -1], 8, 2, 0, 2, None, &cfg)
        .unwrap();
    assert!(gpu.scoring_posterior_gpu);
    assert_eq!(
        HOST_TABLE_CELLS.with(Cell::get),
        0,
        "explicit GPU still builds host probability cells"
    );
    assert_eq!(
        HOST_EAP_ROWS.with(Cell::get),
        0,
        "explicit GPU still reduces EAP moments on host"
    );
    assert!(
        DEVICE_TABLE_CALLS.with(Cell::get) >= 3,
        "no completed device table call consumed"
    );
    assert_eq!(
        DEVICE_EAP_CALLS.with(Cell::get),
        1,
        "no completed final EAP reduction consumed"
    );
    let cpu = fit_bifactor_grm_multigroup(
        &y,
        None,
        &groups,
        2,
        &[-1, -1],
        8,
        2,
        0,
        2,
        None,
        &BifactorMultigroupConfig {
            device: crate::Device::Cpu,
            ..cfg
        },
    )
    .unwrap();
    assert!(
        HOST_TABLE_CELLS.with(Cell::get) > 0 && HOST_EAP_ROWS.with(Cell::get) > 0,
        "CPU control did not exercise the observing hooks"
    );
    for (a, b) in gpu.theta_g_eap.iter().zip(&cpu.theta_g_eap) {
        assert!((a - b).abs() < 1e-5);
    }
    for (a, b) in gpu.theta_g_sd.iter().zip(&cpu.theta_g_sd) {
        assert!((a - b).abs() < 1e-5);
    }
}
#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn scoring_completion_specific_mean_only_real_consumer() {
    leased();
    reset();
    let (y, obs, groups, map, cfg) = fixture();
    let cpu =
        fit_bifactor_grm_multigroup(&y, Some(&obs), &groups, 2, &map, 12, 7, 3, 3, None, &cfg)
            .unwrap();
    reset();
    let gpu = fit_bifactor_grm_multigroup(
        &y,
        Some(&obs),
        &groups,
        2,
        &map,
        12,
        7,
        3,
        3,
        None,
        &BifactorMultigroupConfig {
            device: crate::Device::Gpu,
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(HOST_TABLE_CELLS.with(Cell::get), 0);
    assert_eq!(HOST_EAP_ROWS.with(Cell::get), 0);
    assert_eq!(
        DEVICE_MEAN_ONLY_ROWS.with(Cell::get),
        3,
        "mean-only update remains host arithmetic"
    );
    assert_eq!(gpu.specific_sd, vec![vec![1.0; 3]; 2]);
    assert!(gpu
        .dispatch
        .iter()
        .all(|d| d.gpu_used && d.fallback_reason.is_none()));
    for (a, b) in gpu
        .specific_mean
        .iter()
        .flatten()
        .zip(cpu.specific_mean.iter().flatten())
    {
        assert!((a - b).abs() < 2e-5, "specific-mean CPU/GPU parity");
    }
    // Same returned bank/prior oracle separates scoring from optimizer trajectory.
    let (x, w) = gh_rule(3).unwrap();
    for p in 0..12 {
        let g = groups[p];
        let mut den = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        for t in 0..3 {
            for h0 in 0..3 {
                for h1 in 0..3 {
                    for h2 in 0..3 {
                        let hs = [h0, h1, h2];
                        let theta = gpu.general_mean[g] + gpu.general_sd[g] * x[t];
                        let mut probability = w[t] * w[h0] * w[h1] * w[h2];
                        for i in 0..7 {
                            if obs[p * 7 + i] {
                                let eta = gpu.a_general[g][i] * theta
                                    + if i < 6 {
                                        gpu.a_specific[g][i]
                                            * (gpu.specific_mean[g][i / 2]
                                                + gpu.specific_sd[g][i / 2] * x[hs[i / 2]])
                                    } else {
                                        0.0
                                    };
                                probability *=
                                    grm_logprobs(eta, &gpu.threshold[g][i * 2..i * 2 + 2])
                                        [y[p * 7 + i]]
                                        .exp();
                            }
                        }
                        den += probability;
                        first += probability * theta;
                        second += probability * theta * theta;
                    }
                }
            }
        }
        let mean = first / den;
        let sd = (second / den - mean * mean).max(0.0).sqrt();
        assert!(
            (gpu.theta_g_eap[p] - mean).abs() < 2e-5,
            "same-fit EAP parity"
        );
        assert!((gpu.theta_g_sd[p] - sd).abs() < 2e-5, "same-fit SD parity");
    }
    eprintln!("same-fit full-product EAP/SD oracle passed for all 12 synthetic rows");
}

#[test]
#[ignore = "separate one-iteration CPU/GPU fit trajectory; known NONPASS retained"]
fn scoring_completion_separate_fit_trajectory_nonpass() {
    leased();
    let (y, obs, groups, map, cfg) = fixture();
    let cpu =
        fit_bifactor_grm_multigroup(&y, Some(&obs), &groups, 2, &map, 12, 7, 3, 3, None, &cfg)
            .unwrap();
    let gpu = fit_bifactor_grm_multigroup(
        &y,
        Some(&obs),
        &groups,
        2,
        &map,
        12,
        7,
        3,
        3,
        None,
        &BifactorMultigroupConfig {
            device: crate::Device::Gpu,
            ..cfg
        },
    )
    .unwrap();
    for (a, b) in gpu.theta_g_eap.iter().zip(&cpu.theta_g_eap) {
        assert!(
            (a - b).abs() < 2e-5,
            "separate-fit EAP parity abs_error={}",
            (a - b).abs()
        );
    }
    for (a, b) in gpu.theta_g_sd.iter().zip(&cpu.theta_g_sd) {
        assert!(
            (a - b).abs() < 2e-5,
            "separate-fit SD parity abs_error={}",
            (a - b).abs()
        );
    }
}


#[path = "signed_boundary_reference.rs"]
mod signed_boundary_reference;

#[test]
#[ignore = "canonical owner-held GPU lease required"]
fn signed_boundary_device_exact_parameter_precision() {
    leased();
    let mut max_error = 0.0_f64;
    let mut max_native_error = 0.0_f64;
    let mut max_mass_error = 0.0_f64;
    for &(base, thresholds, exact, _) in signed_boundary_reference::CASES {
        let params = vec![vec![ItemParams { a_g: base, a_s: None, d: thresholds.to_vec() }]];
        let actual = crate::gpu_bifactor_scoring::tables(&params, &[None], &[vec![1.0]], &[vec![]], 4).unwrap();
        let native = grm_logprobs(base, &thresholds);
        let mut mass = 0.0;
        for k in 0..4 {
            let error = (actual[0][0][k] - exact[k]).abs();
            let native_error = (actual[0][0][k] - native[k]).abs();
            max_error = max_error.max(error);
            max_native_error = max_native_error.max(native_error);
            mass += actual[0][0][k].exp();
            eprintln!("SIGNED_CELL base={base:e} upper={:e} lower={:e} category={k} device={:.17e} native={:.17e} reference={:.17e} error={error:e} native_error={native_error:e}", thresholds[0], thresholds[1], actual[0][0][k], native[k], exact[k]);
            assert!(error < 1e-9, "exact-parameter device error={error:e}");
            assert!(native_error < 1e-9, "exact-parameter device/native error={native_error:e}");
        }
        max_mass_error = max_mass_error.max((mass-1.0).abs());
        assert!((mass-1.0).abs() < 1e-9);
    }
    eprintln!("DEVICE_EXACT cases={} max_log_error={max_error:e} max_native_error={max_native_error:e} max_mass_error={max_mass_error:e}", signed_boundary_reference::CASES.len());
}

#[test]
#[ignore = "rounded-full-logit target differs from exact parameters; strict NONPASS retained"]
fn signed_boundary_rounded_logits_nonpass() {
    leased();
    let base = -1.1;
    let params = vec![vec![ItemParams { a_g: base, a_s: None, d: vec![-1.0, -1.0-1e-7, -2.0] }]];
    let actual = crate::gpu_bifactor_scoring::tables(&params, &[None], &[vec![1.0]], &[vec![]], 4).unwrap();
    let rounded: Vec<f64> = params[0][0].d.iter().map(|d| base + d).collect();
    let expected = grm_logprobs(0.0, &rounded);
    let error = (actual[0][0][1]-expected[1]).abs();
    assert!(error < 1e-9, "explicit rounded-full-logit error={error:e}");
}
