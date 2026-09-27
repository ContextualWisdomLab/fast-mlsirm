//! Exact no-prior regression against main 99c228a8 on fixed synthetic inputs.
//! The pinned bits were generated on s1 (x86_64 Linux, Rust 1.97.1);
//! transcendental functions may round differently on other platforms.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm, fit_bifactor_grm_multigroup, BifactorGrmConfig, BifactorMultigroupConfig,
    SlopePrior,
};

fn feed(h: &mut u64, word: u64) {
    *h = (*h ^ word).wrapping_mul(0x100000001b3);
}

fn floats(h: &mut u64, values: &[f64]) {
    for &v in values {
        feed(h, v.to_bits());
    }
}

fn rows(h: &mut u64, values: &[Vec<f64>]) {
    for row in values {
        floats(h, row);
    }
}

fn metadata(
    h: &mut u64,
    n_iter: usize,
    converged: bool,
    reason: &str,
    change: f64,
    best_start: usize,
    n_parameters: usize,
) {
    feed(h, n_iter as u64);
    feed(h, converged as u64);
    for byte in reason.bytes() {
        feed(h, byte as u64);
    }
    feed(h, change.to_bits());
    feed(h, best_start as u64);
    feed(h, n_parameters as u64);
}

fn data() -> (Vec<usize>, Vec<usize>) {
    let mut state = 20_260_921u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    };
    let mut y = Vec::with_capacity(2 * 200 * 6);
    let mut groups = Vec::with_capacity(400);
    let ag = [1.3, 1.1, 0.9, 1.2, 1.0, 0.8];
    let as_ = [1.0, 0.9, 1.1, 0.8, 1.0, 0.9];
    let d = [
        1.2, -0.8, 1.0, -1.0, 1.3, -0.7, 1.1, -0.9, 0.9, -1.1, 1.2, -0.8,
    ];
    for group in 0..2 {
        for _ in 0..200 {
            let (tg, ts) = {
                let mut normal = || {
                    let u1 = uniform().clamp(1e-12, 1.0 - 1e-12);
                    let u2 = uniform();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                (normal() + 0.5 * group as f64, [normal(), normal()])
            };
            for i in 0..6 {
                let base = ag[i] * tg + as_[i] * ts[i / 3];
                let u = uniform();
                let mut cat = 0;
                for k in 0..2 {
                    if u < 1.0 / (1.0 + (-(base + d[i * 2 + k])).exp()) {
                        cat += 1;
                    } else {
                        break;
                    }
                }
                y.push(cat);
            }
            groups.push(group);
        }
    }
    (y, groups)
}

#[test]
fn none_path_matches_main_bits_single_and_multigroup() {
    let (y, groups) = data();
    let map = [0, 0, 0, 1, 1, 1];
    let single = fit_bifactor_grm(
        &y[..200 * 6],
        None,
        &map,
        200,
        6,
        2,
        3,
        &BifactorGrmConfig {
            q_general: 7,
            q_specific: 7,
            max_iter: 500,
            tol: 1e-4,
            n_starts: 1,
            seed: 42,
            newton_iter: 10,
            ridge: 1e-8,
            slope_prior: SlopePrior::None,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("single-group reference fit");
    let mut hs = 0xcbf29ce484222325u64;
    floats(&mut hs, &single.a_general);
    floats(&mut hs, &single.a_specific);
    floats(&mut hs, &single.threshold);
    floats(&mut hs, &single.theta_g_eap);
    floats(&mut hs, &single.theta_g_sd);
    for &count in &single.category_counts {
        feed(&mut hs, count as u64);
    }
    floats(&mut hs, &single.loglik_trace);
    metadata(
        &mut hs,
        single.n_iter,
        single.converged,
        &single.termination_reason,
        single.final_loglik_change,
        single.best_start,
        single.n_parameters,
    );
    assert_eq!(
        hs, 0xfa2246fadb9c8984,
        "single-group None path diverged from main 99c228a8"
    );

    let mg = fit_bifactor_grm_multigroup(
        &y,
        None,
        &groups,
        2,
        &map,
        400,
        6,
        2,
        3,
        Some(&[true, true, false, true, true, false]),
        &BifactorMultigroupConfig {
            q_general: 7,
            q_specific: 7,
            max_iter: 500,
            tol: 1e-4,
            n_starts: 1,
            seed: 42,
            newton_iter: 10,
            ridge: 1e-8,
            slope_prior: SlopePrior::None,
            estimate_specific_vars: false,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("multigroup reference fit");
    let mut hm = 0xcbf29ce484222325u64;
    rows(&mut hm, &mg.a_general);
    rows(&mut hm, &mg.a_specific);
    rows(&mut hm, &mg.threshold);
    floats(&mut hm, &mg.general_mean);
    floats(&mut hm, &mg.general_sd);
    rows(&mut hm, &mg.specific_sd);
    floats(&mut hm, &mg.theta_g_eap);
    floats(&mut hm, &mg.theta_g_sd);
    for row in &mg.group_category_counts {
        for &count in row {
            feed(&mut hm, count as u64);
        }
    }
    floats(&mut hm, &mg.loglik_trace);
    metadata(
        &mut hm,
        mg.n_iter,
        mg.converged,
        &mg.termination_reason,
        mg.final_loglik_change,
        mg.best_start,
        mg.n_parameters,
    );
    assert_eq!(
        hm, 0x4f3d57b5c4061d3c,
        "multigroup None path diverged from main 99c228a8"
    );
}
