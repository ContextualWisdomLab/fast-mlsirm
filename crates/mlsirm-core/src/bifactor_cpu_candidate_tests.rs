use crate::bifactor_grm::{
    ac_reset_calls, e_step_multigroup, ItemParams, Validated, AC_BLOCK_LSE, AC_CACHE_F64,
    AC_COUNT_BUFFERS,
};
fn reset_calls() {
    ac_reset_calls();
}
fn calls() -> usize {
    AC_BLOCK_LSE.with(|n| n.get())
}
pub(super) fn cache_f64() -> usize {
    AC_CACHE_F64.with(|n| n.get())
}
pub(super) fn count_buffers() -> usize {
    AC_COUNT_BUFFERS.with(|n| n.get())
}
// Test-only fixed inputs and complete returned-array projection.
// Shared fixture bytes are included inside each private module.
pub(crate) fn fixture(
    q: usize,
    masked: bool,
    shift: f64,
    zero_weight: bool,
) -> (Vec<u64>, usize, usize) {
    assert!(q >= 121);
    let np = 12;
    let ni = 5;
    let ns = 2;
    let nc = 4;
    let v = Validated {
        n_persons: np,
        n_items: ni,
        n_specific: ns,
        n_cat: nc,
        m1: nc - 1,
        blocks: vec![vec![0, 1], vec![2, 3]],
        general_only: vec![4],
        item_block: vec![Some(0), Some(0), Some(1), Some(1), None],
    };
    let group: Vec<usize> = (0..np).map(|p| p / 6).collect();
    let mut y = vec![0usize; np * ni];
    let mut mask = vec![true; np * ni];
    for p in 0..np {
        let pattern = p % 3;
        for i in 0..ni {
            y[p * ni + i] = (pattern + i) % nc;
            if masked && ((pattern == 1 && i < 2) || (pattern == 2 && (i == 0 || i == 3))) {
                mask[p * ni + i] = false;
                y[p * ni + i] = usize::MAX; // Unobserved payload must never index tables.
            }
        }
    }
    let observed = if masked { Some(mask.as_slice()) } else { None };
    let mut patterns = std::collections::HashSet::new();
    for p in 0..np {
        for (block, members) in v.blocks.iter().enumerate() {
            let values: Vec<Option<usize>> = members
                .iter()
                .map(|&i| {
                    if mask[p * ni + i] {
                        Some(y[p * ni + i])
                    } else {
                        None
                    }
                })
                .collect();
            patterns.insert((group[p], block, values));
        }
    }
    let expected = patterns.len() * q;
    let params: Vec<Vec<ItemParams>> = (0..2)
        .map(|g| {
            (0..ni)
                .map(|i| ItemParams {
                    a_g: (if i == 3 { -0.9 } else { 1.1 }) + 0.08 * (g as f64),
                    a_s: if i == 4 {
                        None
                    } else {
                        Some(0.7 + 0.05 * (g as f64))
                    },
                    d: vec![1.2 + shift + 0.03 * (i as f64), 0.1 + shift, -1.1 + shift],
                })
                .collect()
        })
        .collect();
    let (x, w) = crate::quadrature::gh_rule(q).expect("caller quadrature");
    let mut log_wg: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let mut log_ws = log_wg.clone();
    if zero_weight {
        log_wg[0] = f64::NEG_INFINITY;
        log_ws[0] = f64::NEG_INFINITY;
    }
    let tg: Vec<Vec<f64>> = (0..2)
        .map(|g| {
            x.iter()
                .map(|&n| if g == 0 { n } else { 0.4 + 1.2 * n })
                .collect()
        })
        .collect();
    let ts: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|g| {
            (0..ns)
                .map(|s| {
                    x.iter()
                        .map(|&n| (1.0 + 0.1 * (g + s) as f64) * n)
                        .collect()
                })
                .collect()
        })
        .collect();
    reset_calls();
    let (ll, counts, w_acc, s1_g, s2_g, _s1_spec, s2_spec, w_spec) = e_step_multigroup(
        &v,
        &y,
        observed,
        &group,
        2,
        &params,
        &tg,
        &ts,
        &log_wg,
        &log_ws,
        q,
        q,
        crate::Device::Cpu,
    );
    let calls = calls();
    let mut output = vec![ll];
    for group in counts {
        for item in group {
            for node in item {
                output.extend(node);
            }
        }
    }
    output.extend(w_acc);
    output.extend(s1_g);
    output.extend(s2_g);
    for row in s2_spec {
        output.extend(row);
    }
    for row in w_spec {
        output.extend(row);
    }
    assert!(
        output.iter().all(|v| v.is_finite()),
        "Nonfinite returned component"
    );
    (
        output.into_iter().map(f64::to_bits).collect(),
        calls,
        expected,
    )
}
