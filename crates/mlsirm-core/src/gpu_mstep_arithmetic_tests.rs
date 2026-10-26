//! Native compiler tests of the exact production arithmetic, not CPU emulation.
use super::*;

fn scalar(op: &str, values: &[f64]) -> Vec<f64> {
    let prefix = include_str!("gpu_mstep_ds.wgsl")
        .split("fn np()")
        .next()
        .unwrap();
    let source = format!("{prefix}\n@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{ result[id.x]={op}(data[id.x]); }}");
    let data: Vec<f32> = values
        .iter()
        .flat_map(|&v| {
            let hi = v as f32;
            [hi, (v - hi as f64) as f32]
        })
        .collect();
    dispatch(
        &source,
        &data,
        values.len() * 2,
        values.len() as u32,
        "probe",
    )
    .unwrap()
    .chunks_exact(2)
    .map(|v| v[0] as f64 + v[1] as f64)
    .collect()
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn gpu_mstep_arithmetic_rejects_unrepresentable_nonzero_count() {
    let result = pack(
        &[1.0, 0.5, -0.5],
        1,
        &[0.0],
        &[vec![1e-50, 1.0, 1.0]],
        SlopePrior::None,
    );
    assert!(
        result.is_err(),
        "nonzero f64 count becomes zero in two-f32 transport"
    );
    assert!(pack(
        &[1.0, 0.5, -0.5],
        1,
        &[0.0],
        &[vec![0.0, 1.0, 1.0]],
        SlopePrior::None
    )
    .is_ok());
    for bad in [f64::INFINITY, f64::NAN, -1.0, f64::MAX] {
        assert!(pack(
            &[1.0, 0.5, -0.5],
            1,
            &[0.0],
            &[vec![bad, 1.0, 1.0]],
            SlopePrior::None
        )
        .is_err());
    }
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn gpu_mstep_arithmetic_large_count_tail_must_not_fake_stationarity() {
    let p = [1.0, 0.0];
    let nodes = [90.0];
    let counts = vec![vec![0.0, 1e38]];
    let ctrl = NewtonControl {
        iterations: 20,
        backtracks: 25,
        ridge: 1e-8,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    };
    // Independent native f64 stable endpoint gradient (not fitter fallback).
    let native_gradient = |p: &[f64]| {
        let tail = (-p[0] * nodes[0] - p[1]).exp();
        let db = -counts[0][1] * tail / (1.0 + tail);
        (db * nodes[0]).abs().max(db.abs())
    };
    assert!(native_gradient(&p) > 1.0);
    let result = update_grm(&p, 1, &nodes, &counts, SlopePrior::None, &ctrl);
    match result {
        Ok((updated, receipt)) => {
            let norm = native_gradient(&updated);
            eprintln!("LARGE_COUNT_TAIL norm={norm} receipt={receipt:?}");
            assert!(
                norm <= 1e-9,
                "GPU Ok with native gradient {norm}, tail mass lost"
            );
        }
        Err(error) => eprintln!("LARGE_COUNT_TAIL_REJECTION {error}"),
    }
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn gpu_mstep_arithmetic_exp_tails_and_overflow() {
    let values = [-70.0, -80.0, -90.0, -100.0, -104.0, -120.0];
    let got = scalar("ex", &values);
    for (&x, y) in values.iter().zip(got) {
        let expected = x.exp();
        eprintln!(
            "EXP_TAIL x={x} gpu={y:e} native={expected:e} relative={:e}",
            (y / expected - 1.0).abs()
        );
        assert!(y.is_finite() && (y - expected).abs() <= 1e-9);
    }
    let got = scalar("ex", &[90.0, 1000.0]);
    for y in got {
        assert!(!y.is_finite(), "overflow silently returns finite {y}");
    }
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn gpu_mstep_arithmetic_log_upper_range() {
    let values = [1.0, 2.0, 0.5, 1e-20, 1e20, f32::MAX as f64];
    let got = scalar("lg", &values);
    for (&x, y) in values.iter().zip(got) {
        let expected = x.ln();
        eprintln!("LOG_RANGE x={x:e} gpu={y:e} native={expected:e}");
        assert!(
            y.is_finite() && (y - expected).abs() <= 1e-9,
            "log range x={x:e} gpu={y:e} native={expected:e}"
        );
    }
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn gpu_mstep_arithmetic_exp_upper_range() {
    let values = [-60.0, -20.0, -0.7, 0.0, 0.7, 20.0, 60.0, 88.0, 88.7];
    let got = scalar("ex", &values);
    for (&x, y) in values.iter().zip(got) {
        let expected = x.exp();
        eprintln!("EXP_RANGE x={x:e} gpu={y:e} native={expected:e}");
        assert!(
            y.is_finite() && (y / expected - 1.0).abs() <= 1e-9,
            "exp range x={x:e} gpu={y:e} native={expected:e}"
        );
    }
}
