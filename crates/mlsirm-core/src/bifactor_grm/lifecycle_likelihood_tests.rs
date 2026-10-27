//! Synthetic same-node diagnostic only. CPU oracles never feed fitting.
use super::*;
thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ROWS: std::cell::RefCell<Vec<serde_json::Value>> = const { std::cell::RefCell::new(Vec::new()) };
}
pub(super) fn seed_retained(pars: &mut [Vec<ItemParams>], mus: &mut [f64], sigmas: &mut [f64]) {
    if !ACTIVE.with(|v| v.get()) {
        return;
    }
    // Exact retained initial winner; no fitted-state/person payload required.
    let packed = [
        0.5446694403732497,
        -0.38974921816525665,
        0.9721333888238428,
        -0.1019296939950246,
        1.329977046485114,
        0.42271968187964903,
        2.4084463378708403,
        0.4705526858115061,
    ];
    for row in pars {
        for i in 0..4 {
            row[i].a_g = packed[2 * i];
            row[i].d = vec![packed[2 * i + 1]];
        }
    }
    mus[1] = -0.06302244961261749;
    sigmas[1] = 1.028186321258545;
}
#[allow(clippy::too_many_arguments)]
pub(super) fn observe(
    stage: &str,
    iteration: usize,
    mut gpu: f64,
    v: &Validated,
    y: &[usize],
    obs: Option<&[bool]>,
    gid: &[usize],
    pars: &[Vec<ItemParams>],
    tg: &[Vec<f64>],
    ts: &[Vec<Vec<f64>>],
    lw: &[f64],
    lv: &[f64],
    qg: usize,
    qs: usize,
) {
    if !ACTIVE.with(|v| v.get()) {
        return;
    }
    if !gpu.is_finite() {
        gpu = e_step_multigroup(
            v,
            y,
            obs,
            gid,
            2,
            pars,
            tg,
            ts,
            lw,
            lv,
            qg,
            qs,
            crate::Device::Gpu,
        )
        .0;
    }
    let receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
    assert!(receipt.used && receipt.fallback_reason.is_none());
    let cpu = e_step_multigroup(
        v,
        y,
        obs,
        gid,
        2,
        pars,
        tg,
        ts,
        lw,
        lv,
        qg,
        qs,
        crate::Device::Cpu,
    )
    .0;
    let mut sensitivity = Vec::new();
    if iteration >= 10 {
        for n in [21, 121] {
            let (x, w) = gh_rule(n).unwrap();
            let (old_x, _) = gh_rule(qg).unwrap();
            let sigma = (tg[1][qg - 1] - tg[1][0]) / (old_x[qg - 1] - old_x[0]);
            let mu = tg[1][0] - sigma * old_x[0];
            let alt_tg = vec![x.to_vec(), x.iter().map(|x| mu + sigma * x).collect()];
            let alt_lw: Vec<_> = w.iter().map(|w| w.ln()).collect();
            let ll = e_step_multigroup(
                v,
                y,
                obs,
                gid,
                2,
                pars,
                &alt_tg,
                ts,
                &alt_lw,
                lv,
                n,
                qs,
                crate::Device::Cpu,
            )
            .0;
            sensitivity.push(serde_json::json!({"quadpts":n,"cpu_fixed_parameters":ll}));
        }
    }
    let row = serde_json::json!({"stage":stage,"iteration":iteration,"gpu":gpu,"cpu_same_nodes":cpu,
        "gpu_minus_cpu":gpu-cpu,"backend":receipt.backend,"device":receipt.device_name,"quadrature_sensitivity":sensitivity});
    eprintln!("LIKELIHOOD_DISCRIMINATOR {row}");
    ROWS.with(|v| v.borrow_mut().push(row));
}
fn fixture() -> (Vec<usize>, Vec<usize>) {
    let mut seed = 98231u64;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / 9007199254740992.0
    };
    let mut y = Vec::new();
    let mut group = Vec::new();
    for g in 0..2 {
        for _ in 0..128 {
            let theta = (uniform() + uniform() + uniform() + uniform() + uniform() + uniform()
                - 3.)
                * 2f64.sqrt();
            group.push(g);
            for i in 0..4 {
                let eta = (0.8 + 0.2 * i as f64) * theta + (-0.6 + 0.4 * i as f64);
                y.push(usize::from(uniform() < 1. / (1. + (-eta).exp())));
            }
        }
    }
    (y, group)
}
#[test]
#[ignore = "canonical hardware GPU lease required"]
fn retained_embedded_item2_decrease() {
    run_retained(7, 100, false);
}
#[test]
#[ignore = "canonical hardware GPU lease required"]
fn retained_embedded_item2_q21() {
    run_retained(21, 100, false);
}
#[test]
#[ignore = "canonical hardware GPU lease required"]
fn retained_embedded_item2_finite_objective() {
    run_retained(7, 100, false);
}
fn run_retained(q: usize, iterations: usize, expect_refusal: bool) {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
    let (y, gid) = fixture();
    let cfg = BifactorMultigroupConfig {
        q_general: q,
        q_specific: 1,
        max_iter: iterations,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        estimate_specific_vars: false,
        estimate_specific_means: false,
        device: crate::Device::Gpu,
    };
    let single = BifactorGrmConfig {
        q_general: q,
        q_specific: 1,
        max_iter: iterations,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        device: crate::Device::Gpu,
    };
    let v = validate(&y, None, &[-1; 4], 256, 4, 0, 2, &single).unwrap();
    let (x, w) = gh_rule(q).unwrap();
    let (z, u) = gh_rule(1).unwrap();
    let lw: Vec<_> = w.iter().map(|w| w.ln()).collect();
    let lv: Vec<_> = u.iter().map(|w| w.ln()).collect();
    let mut dispatch = Vec::new();
    let mut m = 0;
    ACTIVE.with(|v| v.set(true));
    ROWS.with(|v| v.borrow_mut().clear());
    let out = run_single_start_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &[true, true, false, true],
        &cfg,
        x,
        z,
        &lw,
        &lv,
        q,
        1,
        0,
        None,
        &mut dispatch,
        &mut m,
    );
    ACTIVE.with(|v| v.set(false));
    if expect_refusal {
        let error = out.err().expect("retained native refusal must reproduce");
        eprintln!(
            "RETAINED_REFUSAL {error} m_dispatch={m} sweeps={}",
            dispatch.len()
        );
        assert!(error.contains("decreased at iteration 11"), "{error}");
    } else {
        let r = out.unwrap();
        eprintln!(
            "RETAINED_Q{q} converged={} iterations={} m_dispatch={m}",
            r.converged, r.n_iter
        );
        assert!(r.converged);
        ROWS.with(|rows| {
            let rows=rows.borrow();
            let sweeps:Vec<_>=rows.iter().filter(|r|r["stage"]=="sweep").collect();
            for row in &sweeps {
                assert!(row["gpu_minus_cpu"].as_f64().unwrap().abs()<1e-9,"strict device total likelihood parity");
            }
            for pair in sweeps.windows(2) {
                let old=pair[0]["cpu_same_nodes"].as_f64().unwrap();
                let new=pair[1]["cpu_same_nodes"].as_f64().unwrap();
                assert!(new-old>=-32.0*f64::EPSILON*(1.0+old.abs()),"native finite objective must be monotone");
            }
            let previous=sweeps[sweeps.len()-2]["cpu_same_nodes"].as_f64().unwrap();
            let current=sweeps[sweeps.len()-1]["cpu_same_nodes"].as_f64().unwrap();
            let delta=current-previous;let threshold=cfg.tol*(1.0+previous.abs());
            eprintln!("NATIVE_STOP_CHECK delta={delta:.16e} threshold={threshold:.16e} iterations={}",r.n_iter);
            assert!(delta<=threshold,"premature convergence: native delta={delta:.16e} > unchanged threshold={threshold:.16e}");
        });
    }
    assert!(dispatch
        .iter()
        .all(|r| r.gpu_used && r.fallback_reason.is_none()));
}

#[test]
#[ignore = "canonical hardware GPU lease required"]
fn strict_same_node_counts_moments_and_likelihood() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
    // Odd persons, four categories, one specific block, individual missing items,
    // entirely missing block, and high quadrature zero/underflow weights.
    for (qg, qs) in [(7, 5), (21, 7), (121, 1), (481, 1)] {
        let n = 17;
        let ni = 4;
        let nc = 4;
        let gid: Vec<_> = (0..n).map(|p| p % 2).collect();
        let y: Vec<_> = (0..n * ni).map(|i| (i / ni + i % ni) % nc).collect();
        let mut obs = vec![true; n * ni];
        obs[0] = false;
        obs[ni] = false;
        obs[ni + 1] = false;
        let cfg = BifactorGrmConfig {
            q_general: qg,
            q_specific: qs,
            max_iter: 1,
            tol: 1e-8,
            n_starts: 1,
            seed: 42,
            newton_iter: 20,
            ridge: 1e-6,
            slope_prior: SlopePrior::None,
            device: crate::Device::Gpu,
        };
        let v = validate(&y, Some(&obs), &[0, 0, -1, -1], n, ni, 1, nc, &cfg).unwrap();
        let mut pars = vec![initial_params(&v, &y, Some(&obs), 42, 0); 2];
        for (g, row) in pars.iter_mut().enumerate() {
            for (i, p) in row.iter_mut().enumerate() {
                p.a_g = 0.3 + 0.12 * i as f64;
                p.a_s = p.a_s.map(|_| 0.2 + 0.07 * g as f64);
                p.d = vec![0.9, 0.1, -0.7];
            }
        }
        let (x, w) = gh_rule(qg).unwrap();
        let (z, u) = gh_rule(qs).unwrap();
        let tg = vec![x.to_vec(), x.iter().map(|v| 0.17 + 1.13 * v).collect()];
        let ts = vec![
            vec![z.to_vec()],
            vec![z.iter().map(|v| -0.12 + 0.93 * v).collect()],
        ];
        let lw: Vec<_> = w.iter().map(|v| v.ln()).collect();
        let ls: Vec<_> = u.iter().map(|v| v.ln()).collect();
        let cpu = e_step_multigroup(
            &v,
            &y,
            Some(&obs),
            &gid,
            2,
            &pars,
            &tg,
            &ts,
            &lw,
            &ls,
            qg,
            qs,
            crate::Device::Cpu,
        );
        let gpu = e_step_multigroup(
            &v,
            &y,
            Some(&obs),
            &gid,
            2,
            &pars,
            &tg,
            &ts,
            &lw,
            &ls,
            qg,
            qs,
            crate::Device::Gpu,
        );
        let receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
        assert!(
            receipt.used && receipt.fallback_reason.is_none(),
            "{receipt:?}"
        );
        let flatten = |r: &(
            f64,
            Vec<Vec<Vec<Vec<f64>>>>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
        )| {
            let mut v = vec![r.0];
            v.extend(r.1.iter().flatten().flatten().flatten().copied());
            v.extend_from_slice(&r.2);
            v.extend_from_slice(&r.3);
            v.extend_from_slice(&r.4);
            v.extend(r.5.iter().flatten().copied());
            v.extend(r.6.iter().flatten().copied());
            v.extend(r.7.iter().flatten().copied());
            v
        };
        let c = flatten(&cpu);
        let g = flatten(&gpu);
        assert_eq!(c.len(), g.len());
        let max = c
            .iter()
            .zip(&g)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("STRICT_E_PARITY qg={qg} qs={qs} n={n} cells={} max_abs={max:.16e} likelihood_abs={:.16e}",c.len(),(cpu.0-gpu.0).abs());
        assert!(g.iter().all(|v| v.is_finite()));
        assert!(max < 1e-9, "native strict E parity {max}");
    }
}

pub(super) fn deny_cpu_item_fit() {
    assert!(
        !ACTIVE.with(|v| v.get()),
        "CPU fitting forbidden in retained GPU fixture"
    );
}
#[test]
#[ignore = "canonical hardware GPU lease required"]
fn retained_q7_returned_bank_strict_final_posterior() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
    let (y, gid) = fixture();
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 1,
        max_iter: 100,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        estimate_specific_vars: false,
        estimate_specific_means: false,
        device: crate::Device::Gpu,
    };
    ACTIVE.with(|v| v.set(true));
    ROWS.with(|v| v.borrow_mut().clear());
    let out = fit_bifactor_grm_multigroup(
        &y,
        None,
        &gid,
        2,
        &[-1; 4],
        256,
        4,
        0,
        2,
        Some(&[true, true, false, true]),
        &cfg,
    );
    ACTIVE.with(|v| v.set(false));
    let r = out.unwrap();
    assert!(r.converged);
    r.validate_gpu_evidence(
        &y,
        None,
        &gid,
        2,
        &[-1; 4],
        &[true, true, false, true],
        &cfg,
        false,
    )
    .unwrap();
    let (x, w) = gh_rule(7).unwrap();
    let (z, u) = gh_rule(1).unwrap();
    let lw: Vec<_> = w.iter().map(|w| w.ln()).collect();
    let ls: Vec<_> = u.iter().map(|w| w.ln()).collect();
    let cfg1 = BifactorGrmConfig {
        q_general: 7,
        q_specific: 1,
        max_iter: 100,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        device: crate::Device::Gpu,
    };
    let v = validate(&y, None, &[-1; 4], 256, 4, 0, 2, &cfg1).unwrap();
    let pars: Vec<_> = (0..2)
        .map(|g| pack_params(&v, &r.a_general[g], &r.a_specific[g], &r.threshold[g]))
        .collect();
    let tg: Vec<Vec<f64>> = (0..2)
        .map(|g| {
            x.iter()
                .map(|x| r.general_mean[g] + r.general_sd[g] * x)
                .collect()
        })
        .collect();
    let ts = vec![Vec::<Vec<f64>>::new(); 2];
    let cpu = e_step_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &pars,
        &tg,
        &ts,
        &lw,
        &ls,
        7,
        1,
        crate::Device::Cpu,
    )
    .0;
    let gpu = e_step_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &pars,
        &tg,
        &ts,
        &lw,
        &ls,
        7,
        1,
        crate::Device::Gpu,
    )
    .0;
    assert!((cpu - gpu).abs() < 1e-9);
    assert!((cpu - r.loglik_trace.last().unwrap()).abs() < 1e-9);
    eprintln!("RETURNED_BANK qg=7 iterations={} loglik={:.16e} native_abs_error={:.16e} sweeps={} item_receipts={} full_gpu={}",r.n_iter,gpu,(cpu-gpu).abs(),r.dispatch.len(),r.dispatch.iter().map(|r|r.items.len()).sum::<usize>(),r.execution_evidence.as_ref().unwrap().full_gpu);
    let _ = (z, u);
}

#[test]
#[ignore = "canonical hardware GPU lease required"]
fn conditional_group_m_step_ascends_same_counts_mixed() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
    let n = 40;
    let ni = 4;
    let nc = 4;
    let gid: Vec<_> = (0..n).map(|p| p % 2).collect();
    let y: Vec<_> = (0..n * ni).map(|i| (i / ni + i % ni) % nc).collect();
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 5,
        max_iter: 1,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        estimate_specific_means: true,
        estimate_specific_vars: true,
        device: crate::Device::Gpu,
    };
    let single = BifactorGrmConfig {
        q_general: 7,
        q_specific: 5,
        max_iter: 1,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        device: crate::Device::Gpu,
    };
    let v = validate(&y, None, &[0, 0, -1, -1], n, ni, 1, nc, &single).unwrap();
    let mut pars = vec![initial_params(&v, &y, None, 42, 0); 2];
    for row in &mut pars {
        for (i, p) in row.iter_mut().enumerate() {
            p.a_g = 0.4 + 0.16 * i as f64;
            p.a_s = p.a_s.map(|_| 0.45 + 0.2 * i as f64);
            p.d = vec![1.1, 0.1, -0.9];
        }
    }
    let (x, w) = gh_rule(7).unwrap();
    let (z, u) = gh_rule(5).unwrap();
    let lw: Vec<_> = w.iter().map(|v| v.ln()).collect();
    let ls: Vec<_> = u.iter().map(|v| v.ln()).collect();
    let mut mus = vec![0.0, 0.2];
    let mut sigmas = vec![1.0, 1.2];
    let mut sm = vec![vec![0.0], vec![-0.1]];
    let mut ss = vec![vec![1.0], vec![0.9]];
    let make_nodes = |m: &[f64],
                      sd: &[f64],
                      sm: &[Vec<f64>],
                      ss: &[Vec<f64>]|
     -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        (
            (0..2)
                .map(|g| x.iter().map(|x| m[g] + sd[g] * x).collect())
                .collect(),
            (0..2)
                .map(|g| vec![z.iter().map(|z| sm[g][0] + ss[g][0] * z).collect()])
                .collect(),
        )
    };
    let (tg, ts) = make_nodes(&mus, &sigmas, &sm, &ss);
    let before = e_step_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &pars,
        &tg,
        &ts,
        &lw,
        &ls,
        7,
        5,
        crate::Device::Gpu,
    );
    let (limited, backend, device) = group_em::update(
        &v,
        &pars,
        &before.1,
        &cfg,
        x,
        z,
        &mut mus,
        &mut sigmas,
        &mut sm,
        &mut ss,
    )
    .unwrap();
    let (tg2, ts2) = make_nodes(&mus, &sigmas, &sm, &ss);
    let after = e_step_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &pars,
        &tg2,
        &ts2,
        &lw,
        &ls,
        7,
        5,
        crate::Device::Gpu,
    )
    .0;
    let native = e_step_multigroup(
        &v,
        &y,
        None,
        &gid,
        2,
        &pars,
        &tg2,
        &ts2,
        &lw,
        &ls,
        7,
        5,
        crate::Device::Cpu,
    )
    .0;
    assert!(after >= before.0);
    assert!((after - native).abs() < 1e-9);
    assert_eq!(
        (mus[0], sigmas[0], sm[0][0], ss[0][0]),
        (0.0, 1.0, 0.0, 1.0)
    );
    eprintln!("MIXED_GROUP_CONDITIONAL qg=7 qs=5 gain={:.16e} native_abs_error={:.16e} limited={limited} backend={backend} device={device}",after-before.0,(native-after).abs());
}
