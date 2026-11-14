//! Native compiler tests of the exact production arithmetic, not CPU emulation.
use super::*;

fn scalar_probe(op: &str, values: &[f64], report_flag: &str) -> Vec<(f64, bool)> {
    let prefix = include_str!("gpu_mstep_ds.wgsl")
        .split("fn np()")
        .next()
        .unwrap();
    // Each invocation owns its private flag and two output pairs. Read the
    // production flag independently of the exponential's rounded value.
    let source = format!("{prefix}\n@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{ arithmetic_range_failure=false; result[2u*id.x]={op}(data[id.x]); result[2u*id.x+1u]=ds(select(0.0,1.0,{report_flag})); }}");
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
        values.len() * 4,
        values.len() as u32,
        "probe",
    )
    .unwrap()
    .chunks_exact(4)
    .map(|v| {
        assert!(v[2] == 0.0 || v[2] == 1.0, "invalid range flag {}", v[2]);
        assert_eq!(v[3], 0.0);
        (v[0] as f64 + v[1] as f64, v[2] == 1.0)
    })
    .collect()
}

fn scalar_with_range(op: &str, values: &[f64]) -> Vec<(f64, bool)> {
    scalar_probe(op, values, "arithmetic_range_failure")
}

fn assert_exp_range(x: f64, y: f64, range_failure: bool) {
    if x < -66.0 {
        assert!(range_failure, "missing arithmetic range refusal at {x}");
    } else {
        let expected = x.exp();
        assert!(!range_failure, "in-range exponential refused at {x}");
        assert!(
            y.is_finite() && y > 0.0 && (y / expected - 1.0).abs() <= 1e-9,
            "in-range exp x={x} gpu={y:e} native={expected:e}"
        );
    }
}

#[test]
#[ignore = "requires owner-held physical GPU lease"]
#[should_panic(expected = "missing arithmetic range refusal at -90")]
fn gpu_mstep_arithmetic_range_oracle_rejects_suppressed_flag() {
    // Test-only readback mutant: the production arithmetic prefix is unchanged.
    // The old absolute tail assertion accepts the rounded zero; this oracle
    // must reject the missing refusal even though the value is unchanged.
    let (value, flag) = scalar_probe("ex", &[-90.0], "false")[0];
    assert_exp_range(-90.0, value, flag);
}

fn scalar(op: &str, values: &[f64]) -> Vec<f64> {
    scalar_with_range(op, values)
        .into_iter()
        .map(|(value, range_failure)| {
            assert!(!range_failure, "unexpected arithmetic range refusal");
            value
        })
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
    // Both neighbors round to the same high word as -66. The low word must
    // decide refusal, and one invocation's flag must not contaminate another.
    let epsilon = 2.0_f64.powi(-30);
    let values = [
        -70.0,
        -66.0,
        -80.0,
        -66.0 + epsilon,
        -90.0,
        0.0,
        -100.0,
        -66.0 - epsilon,
        -104.0,
        -120.0,
        -60.0,
    ];
    assert_eq!((-66.0 - epsilon) as f32, -66.0_f32);
    assert_eq!((-66.0 + epsilon) as f32, -66.0_f32);
    let got = scalar_with_range("ex", &values);
    assert_eq!(got.len(), values.len());
    for (&x, (y, range_failure)) in values.iter().zip(got) {
        let expected = x.exp();
        eprintln!("EXP_TAIL x={x:.12} gpu={y:e} native={expected:e} range_failure={range_failure}");
        assert_exp_range(x, y, range_failure);
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
