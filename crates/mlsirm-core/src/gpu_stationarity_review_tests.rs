//! Independent native controls for the unchanged Newton stopping contract.
//! The raw helper copies transport only and executes the production WGSL body.
use super::*;

fn control() -> NewtonControl {
    NewtonControl {
        iterations: 20,
        backtracks: 25,
        ridge: 1e-8,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    }
}

fn fixture() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let coords = vec![-1., -0.5, -1., 0.8, 0., -0.5, 0., 0.8, 1., -0.5, 1., 0.8];
    let ts = vec![-0.8, 0.6];
    let truth = [0.7, -0.4, 0.5, 0.8, -0.7];
    let nodes = crate::two_tier_gpu_item::pack_nodes(&[0, 1], true, &coords, &ts, 2, 6, 2).unwrap();
    let counts = nodes
        .chunks(3)
        .map(|x| {
            let eta = x.iter().zip(&truth).map(|(x, a)| x * a).sum::<f64>();
            crate::poly::grm_logprobs(eta, &truth[3..])
                .iter()
                .map(|p| 100. * p.exp())
                .collect()
        })
        .collect();
    (coords, ts, counts)
}

fn native_norm(params: &[f64], coords: &[f64], ts: &[f64], counts: &[Vec<f64>]) -> f64 {
    let (_, g, _) = crate::two_tier_grm::item_neg_ll_grad_hess(
        params,
        &[0, 1],
        true,
        coords,
        ts,
        2,
        6,
        2,
        counts,
    );
    g.iter().map(|x| x.abs()).fold(0., f64::max)
}

fn raw(
    params: &[f64],
    off: usize,
    nodes: &[f64],
    counts: &[Vec<f64>],
    ctrl: &NewtonControl,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    ctrl.validate().unwrap();
    let (data, width) = pack(params, off, nodes, counts, SlopePrior::None).unwrap();
    let ctl = data.len() - 7;
    let mut precise: Vec<f64> = data.iter().map(|x| *x as f64).collect();
    precise[3..3 + params.len()].copy_from_slice(params);
    for (n, row) in counts.iter().enumerate() {
        let start = 3 + params.len() + n * (params.len() + 1);
        precise[start..start + off].copy_from_slice(&nodes[n * off..(n + 1) * off]);
        precise[start + off..start + off + row.len()].copy_from_slice(row);
    }
    precise[ctl..].copy_from_slice(&[
        ctrl.ridge,
        ctrl.iterations as f64,
        ctrl.backtracks as f64,
        ctrl.gradient_tolerance,
        ctrl.step_tolerance,
        ctrl.pivot_tolerance,
        ctrl.armijo,
    ]);
    let packed: Vec<f32> = precise
        .iter()
        .flat_map(|&x| {
            let hi = x as f32;
            [hi, (x - hi as f64) as f32]
        })
        .collect();
    let meta = width + 3 * params.len();
    let values: Vec<f64> = dispatch(
        include_str!("gpu_mstep_ds.wgsl"),
        &packed,
        2 * (meta + 9),
        1,
        "main",
    )
    .unwrap()
    .chunks_exact(2)
    .map(|x| x[0] as f64 + x[1] as f64)
    .collect();
    (
        values[width..width + params.len()].to_vec(),
        values[meta..meta + 9].to_vec(),
    )
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn review_small_nonstationary_step_is_not_status_one() {
    let (coords, ts, counts) = fixture();
    let nodes = crate::two_tier_gpu_item::pack_nodes(&[0, 1], true, &coords, &ts, 2, 6, 2).unwrap();
    let start = [0.9, -0.2, 0.7, 0.6, -0.5];
    let mut ctrl = control();
    ctrl.ridge = 1e12;
    ctrl.iterations = 1;
    let (updated, meta) = raw(&start, 3, &nodes, &counts, &ctrl);
    let change = updated
        .iter()
        .zip(start)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let norm = native_norm(&updated, &coords, &ts, &counts);
    eprintln!("REVIEW_SMALL_STEP change={change:e} native_norm={norm:e} device_meta={meta:?}");
    assert!(change > 0. && change <= ctrl.step_tolerance);
    assert!(norm > ctrl.gradient_tolerance);
    assert_eq!(meta[7], 1.);
    assert_eq!(
        meta[0], 5.,
        "small nonstationary step must not earn converged status1"
    );
    // The inherited host API accepts bounded status5; it is not a convergence flag.
    let (public, receipt) = crate::two_tier_gpu_item::update_item(
        &start,
        &[0, 1],
        true,
        &coords,
        &ts,
        2,
        6,
        2,
        &counts,
        3,
        &ctrl,
        crate::Device::Gpu,
    )
    .unwrap();
    assert_eq!(public, updated);
    assert_eq!(receipt.cpu_numerical_fallbacks, 0);
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn review_local_correction_requires_next_device_evaluation() {
    let (coords, ts, counts) = fixture();
    let nodes = crate::two_tier_gpu_item::pack_nodes(&[0, 1], true, &coords, &ts, 2, 6, 2).unwrap();
    let start = [
        0.6999999999530928,
        -0.3999999999755537,
        0.49999999996858635,
        0.7999999999137817,
        -0.6999999999200837,
    ];
    let ctrl = control();
    let before = native_norm(&start, &coords, &ts, &counts);
    assert!(before > ctrl.gradient_tolerance);
    let (updated, meta) = raw(&start, 3, &nodes, &counts, &ctrl);
    let norm = native_norm(&updated, &coords, &ts, &counts);
    eprintln!("REVIEW_LOCAL before={before:e} native_norm={norm:e} device_meta={meta:?}");
    assert!(norm <= ctrl.gradient_tolerance);
    assert_eq!(meta[0], 1.);
    assert_eq!(meta[7], 1.);
    assert_eq!(
        meta[3], 2.,
        "must evaluate Hessian at the updated point in the next iteration"
    );
    assert_eq!(
        meta[2], 3.,
        "initial, trial and next-iteration gradient evaluations"
    );
    let mut one = ctrl;
    one.iterations = 1;
    let (limited, limited_meta) = raw(&start, 3, &nodes, &counts, &one);
    assert_eq!(limited, updated);
    assert_eq!(
        limited_meta[0], 5.,
        "accepted correction alone is not converged status1"
    );
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn review_adapter_propagates_criterion_solve_search_and_range_failures() {
    use crate::two_tier_gpu_item::update_item;
    let (coords, ts, counts) = fixture();
    let start = [0.9, -0.2, 0.7, 0.6, -0.5];
    for (field, expected) in [(0, "linear solve failure"), (1, "line-search failure")] {
        let mut ctrl = control();
        if field == 0 {
            ctrl.pivot_tolerance = 1e30;
        } else {
            ctrl.armijo = 0.999999;
            ctrl.backtracks = 1;
        }
        let error = update_item(
            &start,
            &[0, 1],
            true,
            &coords,
            &ts,
            2,
            6,
            2,
            &counts,
            3,
            &ctrl,
            crate::Device::Gpu,
        )
        .unwrap_err();
        eprintln!("REVIEW_PROPAGATED {error}");
        assert!(error.contains(expected) && error.contains("CPU numerical fallback forbidden"));
    }
    for (x, c, expected) in [
        (
            90.,
            vec![vec![0., 1e38]],
            "two-word arithmetic range exceeded",
        ),
        (0., vec![vec![3e38, 3e38]], "nonfinite criterion"),
    ] {
        let error = update_item(
            &[1., 0.],
            &[0],
            false,
            &[x],
            &[],
            1,
            1,
            0,
            &c,
            2,
            &control(),
            crate::Device::Gpu,
        )
        .unwrap_err();
        eprintln!("REVIEW_PROPAGATED {error}");
        assert!(error.contains(expected) && error.contains("CPU numerical fallback forbidden"));
    }
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn review_adapter_reversed_free_dimensions_preserve_native_stationarity() {
    let (coords, ts, counts) = fixture();
    let (updated, receipt) = crate::two_tier_gpu_item::update_item(
        &[-0.2, 0.9, 0.7, 0.6, -0.5],
        &[1, 0],
        true,
        &coords,
        &ts,
        2,
        6,
        2,
        &counts,
        3,
        &control(),
        crate::Device::Gpu,
    )
    .unwrap();
    let (_, g, _) = crate::two_tier_grm::item_neg_ll_grad_hess(
        &updated,
        &[1, 0],
        true,
        &coords,
        &ts,
        2,
        6,
        2,
        &counts,
    );
    let norm = g.iter().map(|x| x.abs()).fold(0., f64::max);
    eprintln!("REVIEW_REVERSED native_norm={norm:e} receipt={receipt:?}");
    assert!(norm <= control().gradient_tolerance);
    assert_eq!(receipt.cpu_numerical_fallbacks, 0);
}
