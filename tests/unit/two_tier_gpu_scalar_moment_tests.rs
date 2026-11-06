use super::*;
/// Scalar GPU likelihood does not consume specific moments.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn scalar_gpu_likelihood_omits_unconsumed_specific_moments() {
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
    let actual = focal_loglik(
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
    );
    assert_eq!(actual.to_bits(), full.0.to_bits());
    assert_eq!(
        crate::gpu_bifactor::specific_moment_dispatch_count(),
        0,
        "scalar GPU likelihood still dispatched unused specific moments"
    );
    let retained =
        e_step_fipc_gpu(&v, &y, Some(&observed), &params, &lw, &lws, x, &ts, q, q).unwrap();
    assert_eq!(
        crate::gpu_bifactor::specific_moment_dispatch_count(),
        1,
        "default full GPU helper must still dispatch specific moments"
    );
    assert_eq!(retained.4, full.4);
    assert_eq!(retained.5, full.5);
}
