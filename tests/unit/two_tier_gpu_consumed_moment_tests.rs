use super::*;
/// Moving fixed-specific focal calls do not consume specific moments.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn focal_fixed_specific_caller_omits_only_unused_gpu_moments() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 6,
        n_items: 4,
        n_primary: 1,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: q,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1], vec![2, 3]],
        specific_free: vec![],
        item_block: vec![Some(0), Some(0), Some(1), Some(1)],
    };
    let y: Vec<usize> = (0..24).map(|i| (i / 4 + i % 4) % 3).collect();
    let observed: Vec<bool> = (0..24).map(|i| i % 7 != 0).collect();
    let params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![1.0 + 0.1 * i as f64],
            a_s: Some(0.8),
            d: vec![0.8, -0.8],
        })
        .collect();
    let lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let lws = vec![lw.clone(); 2];
    let ts = vec![x.to_vec(); 2];
    crate::gpu_bifactor::reset_gpu_dispatch_receipt();
    let Some(full) = e_step_fipc_gpu(&v, &y, Some(&observed), &params, &lw, &lws, x, &ts, q, q)
    else {
        assert!(crate::gpu_bifactor::gpu_dispatch_receipt()
            .fallback_reason
            .is_some());
        return;
    };
    crate::gpu_bifactor::reset_specific_moment_dispatch_count();
    let projected = e_step_fipc_anchored_moments(
        &v,
        &y,
        Some(&observed),
        &params,
        &lw,
        &lws,
        x,
        &ts,
        q,
        q,
        crate::Device::Gpu,
        &[true, true, false, true],
        false,
    );
    assert_eq!(full.0.to_bits(), projected.0.to_bits());
    assert_eq!(full.1, projected.1);
    for (a, b) in [
        (&full.2, &projected.2),
        (&full.3, &projected.3),
        (&full.6, &projected.6),
        (&full.7, &projected.7),
    ] {
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(
        crate::gpu_bifactor::specific_moment_dispatch_count(),
        0,
        "actual focal caller dispatched unconsumed specific moments"
    );
    let retained = e_step_fipc_anchored_moments(
        &v,
        &y,
        Some(&observed),
        &params,
        &lw,
        &lws,
        x,
        &ts,
        q,
        q,
        crate::Device::Gpu,
        &[true, true, false, true],
        true,
    );
    assert_eq!(
        crate::gpu_bifactor::specific_moment_dispatch_count(),
        1,
        "estimated specific path must retain moment reduction"
    );
    assert_eq!(full.4, retained.4);
    assert_eq!(full.5, retained.5);
}
