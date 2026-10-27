//! Retained generic pooled-item line-search regression, no study records.

#[test]
#[ignore = "requires actual GPU lease"]
fn narrow_gap_subtraction_preserves_represented_words() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let values = [0.8000001_f64, 0.8];
    let packed: Vec<f32> = values
        .iter()
        .flat_map(|&v| {
            let h = v as f32;
            [h, (v - h as f64) as f32]
        })
        .collect();
    let a = packed[0] as f64 + packed[1] as f64;
    let b = packed[2] as f64 + packed[3] as f64;
    let gap = a - b;
    let prefix = include_str!("gpu_mstep_ds.wgsl")
        .split("@compute @workgroup_size(1)")
        .next()
        .unwrap();
    let shader = format!("{prefix}\n@compute @workgroup_size(1) fn probe() {{ let g=sub(data[0],data[1]); result[0]=g; result[1]=gap_mass(g); result[2]=lg(gap_mass(g)); }}");
    let raw = dispatch(&shader, &packed, 6, 1, "probe").unwrap();
    let got: Vec<f64> = raw.chunks(2).map(|x| x[0] as f64 + x[1] as f64).collect();
    eprintln!(
        "NARROW_PRIMITIVES packed={packed:?} gap={gap:.17e} got={got:?} mass={:.17e} log={:.17e}",
        -(-gap).exp_m1(),
        (-(-gap).exp_m1()).ln()
    );
    assert!((got[0] - gap).abs() < 1e-21);
    assert!((got[1] - (-(-gap).exp_m1())).abs() < 1e-21);
    assert!((got[2] - (-(-gap).exp_m1()).ln()).abs() < 1e-12);
}

use super::*;

// Transport-only probe executes the production difference arithmetic on Metal.
fn device_delta(
    start: &[f64],
    end: &[f64],
    off: usize,
    nodes: &[f64],
    counts: &[Vec<f64>],
    prior: SlopePrior,
) -> f64 {
    let p = start.len();
    let mut data = vec![p as f64, off as f64, counts.len() as f64];
    data.extend_from_slice(start);
    for (n, row) in counts.iter().enumerate() {
        data.extend_from_slice(&nodes[n * off..(n + 1) * off]);
        data.extend_from_slice(row);
    }
    data.extend(match prior {
        SlopePrior::None => [0., 0., 1.],
        SlopePrior::Lognormal { mu, sd } => [1., mu, sd],
    });
    data.extend([0.; 7]);
    let candidate = data.len();
    data.extend_from_slice(end);
    let packed: Vec<f32> = data
        .into_iter()
        .flat_map(|v| {
            let h = v as f32;
            [h, (v - h as f64) as f32]
        })
        .collect();
    let prefix = include_str!("gpu_mstep_ds.wgsl")
        .split("@compute @workgroup_size(1)")
        .next()
        .unwrap();
    let shader=format!("{prefix}\n@compute @workgroup_size(1) fn probe() {{ arithmetic_range_failure=false; let w=width(); for(var j=0u;j<np();j++) {{ result[w+j]=data[3u+j]; result[w+np()+j]=data[{candidate}u+j]; }} result[0]=objective_delta(); result[1]=ds(select(0.0,1.0,arithmetic_range_failure)); }}");
    let raw = dispatch(
        &shader,
        &packed,
        2 * (1 + p + p * p + 3 * p + 9),
        1,
        "probe",
    )
    .unwrap();
    assert_eq!(raw[2], 0.0, "difference arithmetic range refusal");
    raw[0] as f64 + raw[1] as f64
}

#[test]
#[ignore = "requires actual GPU lease"]
fn objective_delta_matches_native_with_signed_prior_and_zero_step() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let nodes = [-1., 0.4, 0.5, -0.7];
    let counts = vec![vec![2., 3., 1.], vec![1., 2., 4.]];
    for prior in [SlopePrior::None, SlopePrior::Lognormal { mu: 0.2, sd: 0.8 }] {
        let start = [-1.1, 0.6, 0.8, -0.7];
        let represented_nodes: Vec<_> = nodes
            .iter()
            .map(|&v| {
                let hi = v as f32;
                hi as f64 + (v - hi as f64) as f32 as f64
            })
            .collect();
        let represented_counts: Vec<Vec<f64>> = counts
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&v| {
                        let hi = v as f32;
                        hi as f64 + (v - hi as f64) as f32 as f64
                    })
                    .collect()
            })
            .collect();
        let ng = [represented_nodes[0], represented_nodes[2]];
        let ns = [represented_nodes[1], represented_nodes[3]];
        let native = |p: &[f64]| {
            let (mut f, _, _) = crate::bifactor_grm::item_neg_ll_grad_hess(
                p,
                true,
                &ng,
                &ns,
                &represented_counts,
                false,
            );
            if let SlopePrior::Lognormal { mu, sd } = prior {
                for a in &p[..2] {
                    let mut g = 0.;
                    crate::bifactor_grm::add_lnorm_abs_slope_prior(*a, mu, sd, &mut f, &mut g);
                }
            }
            f
        };
        assert_eq!(device_delta(&start, &start, 2, &nodes, &counts, prior), 0.);
        for end in [
            [-1.09, 0.61, 0.81, -0.69],
            [-0.8, 0.4, 1.2, -0.6],
            [-1.2, 0.7, 0.8000001, 0.8],
            [-1.2, 0.7, 0.5 + 2.0_f64.powi(-20), 0.5],
        ] {
            let forward = device_delta(&start, &end, 2, &nodes, &counts, prior);
            // The device receives two f32 words, not arbitrary binary64.
            // Narrow gaps amplify transport error: bind this arithmetic oracle
            // to the exact represented inputs while retaining original-f64
            // stationarity checks in the pooled regression below.
            let represented = |v: f64| {
                let hi = v as f32;
                hi as f64 + (v - hi as f64) as f32 as f64
            };
            let exact_start: Vec<_> = start.iter().map(|&v| represented(v)).collect();
            let exact_end: Vec<_> = end.iter().map(|&v| represented(v)).collect();
            let legacy_expected = native(&exact_end) - native(&exact_start);
            // Independent binary64 log-probability oracle: form the threshold
            // gap before adding eta. The ordinary CPU helper subtracts
            // (eta+b0)-(eta+b1), which loses digits for a narrow category.
            let stable = |p: &[f64]| {
                let logsig = |x: f64| {
                    if x >= 0. {
                        -(-x).exp().ln_1p()
                    } else {
                        x - x.exp().ln_1p()
                    }
                };
                let mut f = 0.;
                for (n, row) in represented_counts.iter().enumerate() {
                    let eta = p[0] * ng[n] + p[1] * ns[n];
                    let lp = [
                        logsig(-eta - p[2]),
                        logsig(eta + p[2])
                            + logsig(-eta - p[3])
                            + (-(-(p[2] - p[3])).exp_m1()).ln(),
                        logsig(eta + p[3]),
                    ];
                    f -= row.iter().zip(lp).map(|(c, l)| c * l).sum::<f64>();
                }
                if let SlopePrior::Lognormal { mu, sd } = prior {
                    for a in &p[..2] {
                        let mut g = 0.;
                        crate::bifactor_grm::add_lnorm_abs_slope_prior(*a, mu, sd, &mut f, &mut g);
                    }
                }
                f
            };
            let expected = stable(&exact_end) - stable(&exact_start);
            eprintln!("NATIVE_ORACLE_COMPARISON ordinary={legacy_expected:e} stable={expected:e}");
            if end[2] - end[3] > 1e-5 {
                assert!((legacy_expected - expected).abs() < 1e-10);
            }
            let reverse = device_delta(&end, &start, 2, &nodes, &counts, prior);
            eprintln!("DELTA_PARITY prior={prior:?} expected={expected:e} actual={forward:e} reverse={reverse:e}");
            assert!((forward - expected).abs() < 1e-10);
            assert!((forward + reverse).abs() < 1e-10);
        }
    }
}

#[test]
#[ignore = "requires actual GPU lease"]
fn pooled_item_objective_cancellation_does_not_block_stationarity() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let nodes: Vec<f64> = vec![
        -1.7320508075688772,
        -1.7320508075688772,
        -1.7320508075688772,
        0.0,
        -1.7320508075688772,
        1.7320508075688772,
        0.0,
        -1.7320508075688772,
        0.0,
        0.0,
        0.0,
        1.7320508075688772,
        1.7320508075688772,
        -1.7320508075688772,
        1.7320508075688772,
        0.0,
        1.7320508075688772,
        1.7320508075688772,
    ]
    .repeat(3);
    let counts: Vec<Vec<f64>> = vec![
        vec![
            0.019447116181254387,
            0.0005885573918931186,
            0.0015209810808300972,
        ],
        vec![
            0.1887645274400711,
            0.026671970263123512,
            0.018362026661634445,
        ],
        vec![
            0.04769324138760567,
            0.038630977272987366,
            0.008273828774690628,
        ],
        vec![0.6518723368644714, 0.11482878774404526, 0.3395843505859375],
        vec![1.9553325176239014, 2.064967155456543, 2.0367679595947266],
        vec![0.10873226076364517, 0.6884223818778992, 0.3395843505859375],
        vec![
            0.016507845371961594,
            0.012280276045203209,
            0.07519713044166565,
        ],
        vec![
            0.011397501453757286,
            0.04860395938158035,
            0.16688430309295654,
        ],
        vec![
            0.00025150334113277495,
            0.005007338710129261,
            0.013823515735566616,
        ],
    ]
    .into_iter()
    .cycle()
    .take(27)
    .collect();
    let start = [
        1.0861524476821254,
        0.32379614043445226,
        0.706647017430674,
        -0.7631493107054155,
    ];
    let ctrl = NewtonControl {
        iterations: 20,
        backtracks: 25,
        ridge: 1e-8,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    };
    let ng: Vec<_> = nodes.chunks(2).map(|x| x[0]).collect();
    let ns: Vec<_> = nodes.chunks(2).map(|x| x[1]).collect();
    let full = [
        1.0861529278757835,
        0.3237962736351748,
        0.7066471271371881,
        -0.7631494155777236,
    ];
    let (f0, g0, _) =
        crate::bifactor_grm::item_neg_ll_grad_hess(&start, true, &ng, &ns, &counts, false);
    let (f1, g1, _) =
        crate::bifactor_grm::item_neg_ll_grad_hess(&full, true, &ng, &ns, &counts, false);
    eprintln!(
        "POOLED_NATIVE_ORACLE f0={f0:.17} f1={f1:.17} delta={:e} gradient0={g0:?} gradient1={g1:?}",
        f1 - f0
    );
    assert!(
        f1 < f0,
        "native f64 objective must decrease on retained full step"
    );
    assert!(g0.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-9);
    assert!(g1.iter().map(|v| v.abs()).fold(0.0, f64::max) <= 1e-9);
    let delta = device_delta(&start, &full, 2, &nodes, &counts, SlopePrior::None);
    let reverse = device_delta(&full, &start, 2, &nodes, &counts, SlopePrior::None);
    eprintln!("POOLED_DEVICE_DELTA forward={delta:e} reverse={reverse:e}");
    assert!(delta < 0.0 && reverse > 0.0);
    assert!((delta + reverse).abs() < 1e-18);
    assert!((delta - (f1 - f0)).abs() < 1e-14);
    let (updated, receipt) = update_grm(&start, 2, &nodes, &counts, SlopePrior::None, &ctrl)
        .expect("full Newton correction reaches strict stationarity");
    let ng: Vec<_> = nodes.chunks(2).map(|x| x[0]).collect();
    let ns: Vec<_> = nodes.chunks(2).map(|x| x[1]).collect();
    let (_, g, _) =
        crate::bifactor_grm::item_neg_ll_grad_hess(&updated, true, &ng, &ns, &counts, false);
    let norm = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
    eprintln!("POOLED_NATIVE_GRADIENT {norm:e} receipt={receipt:?}");
    assert_eq!(receipt.cpu_numerical_fallbacks, 0);
    assert!(norm <= 1e-9, "{norm:e}");
}
