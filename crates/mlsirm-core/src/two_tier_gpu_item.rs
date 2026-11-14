//! Two-tier free-primary coordinate adapter for the strict GRM GPU optimizer.
//! This preserves the native two-tier item design x=[free primary..., specific?].
//! The optimizer's shared arithmetic and caller stopping controls are unchanged.
use crate::gpu_mstep::{NewtonControl, StageReceipt};

/// Node transport, not a statistical reduction. Primary grid row varies slowly,
/// specific node varies fastest, matching two_tier_grm::item_neg_ll_grad_hess.
pub fn pack_nodes(
    free: &[usize],
    has_specific: bool,
    coords: &[f64],
    ts: &[f64],
    n_primary: usize,
    n_grid: usize,
    qs: usize,
) -> Result<Vec<f64>, String> {
    let bad = || "invalid two-tier GPU node layout".to_string();
    if n_primary == 0
        || n_grid == 0
        || free.is_empty()
        || n_primary.checked_mul(n_grid) != Some(coords.len())
        || free
            .iter()
            .enumerate()
            .any(|(i, d)| *d >= n_primary || free[..i].contains(d))
        || coords.iter().any(|x| !x.is_finite())
        || (has_specific && (qs == 0 || ts.len() != qs || ts.iter().any(|x| !x.is_finite())))
    {
        return Err(bad());
    }
    let repeats = if has_specific { qs } else { 1 };
    let off = free
        .len()
        .checked_add(usize::from(has_specific))
        .ok_or_else(bad)?;
    let cells = n_grid
        .checked_mul(repeats)
        .and_then(|n| n.checked_mul(off))
        .ok_or_else(bad)?;
    // Reject unrepresentable device index geometry before host packing allocation.
    if cells > u32::MAX as usize {
        return Err(bad());
    }
    let mut nodes = Vec::new();
    nodes.try_reserve_exact(cells).map_err(|_| bad())?;
    for g in 0..n_grid {
        for h in 0..repeats {
            for &d in free {
                nodes.push(coords[g * n_primary + d]);
            }
            if has_specific {
                nodes.push(ts[h]);
            }
        }
    }
    Ok(nodes)
}

/// Validate a completed device update without requiring an update at stationarity.
pub fn require_dispatch(receipt: &StageReceipt) -> Result<(), String> {
    if receipt.cpu_numerical_fallbacks != 0
        || receipt.objective == 0
        || receipt.gradient == 0
        || receipt.hessian == 0
        || receipt.prior == 0
        || (receipt.parameter_update > 0
            && (receipt.solve == 0
                || receipt.line_search == 0
                || receipt.m_step_gpu_dispatches == 0))
    {
        return Err(
            "two-tier item requires completed GPU criterion stages without CPU fallback".into(),
        );
    }
    Ok(())
}

/// One free-primary item update. Only an explicit GPU request is admissible.
/// No CPU optimizer is reachable from this API.
#[allow(clippy::too_many_arguments)]
pub fn update_item(
    params: &[f64],
    free: &[usize],
    has_specific: bool,
    coords: &[f64],
    ts: &[f64],
    n_primary: usize,
    n_grid: usize,
    qs: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
    control: &NewtonControl,
    device: crate::Device,
) -> Result<(Vec<f64>, StageReceipt), String> {
    if device != crate::Device::Gpu {
        return Err("two-tier item adapter requires explicit GPU; CPU/Auto forbidden".into());
    }
    control.validate()?;
    let off = free
        .len()
        .checked_add(usize::from(has_specific))
        .ok_or("item dimension overflow")?;
    let rows = n_grid
        .checked_mul(if has_specific { qs } else { 1 })
        .ok_or("item rows overflow")?;
    if n_cat < 2
        || off.checked_add(n_cat - 1) != Some(params.len())
        || counts.len() != rows
        || counts
            .iter()
            .any(|r| r.len() != n_cat || r.iter().any(|v| !v.is_finite() || *v < 0.))
    {
        return Err("invalid two-tier GPU item parameter/count shape".into());
    }
    let (limit, _) = crate::gpu_mstep::limits()?;
    let cells = rows.checked_mul(off).ok_or("item node overflow")?;
    crate::gpu_mstep::checked_buffer(cells.checked_mul(2).ok_or("item node overflow")?, limit)?;
    let nodes = pack_nodes(free, has_specific, coords, ts, n_primary, n_grid, qs)?;
    let (updated, receipt) = crate::gpu_mstep::update_grm(
        params,
        off,
        &nodes,
        counts,
        crate::bifactor_grm::SlopePrior::None,
        control,
    )?;
    require_dispatch(&receipt)?;
    Ok((updated, receipt))
}

#[cfg(test)]
mod tests {
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
    #[test]
    fn strict_update_refuses_cpu_and_bad_count_shape() {
        for device in [crate::Device::Cpu, crate::Device::Auto] {
            assert!(update_item(
                &[1., 0.],
                &[0],
                false,
                &[0.],
                &[],
                1,
                1,
                0,
                &[vec![1., 1.]],
                2,
                &control(),
                device
            )
            .is_err());
        }
        assert!(update_item(
            &[1., 0.],
            &[0],
            false,
            &[0.],
            &[],
            1,
            1,
            0,
            &[],
            2,
            &control(),
            crate::Device::Gpu
        )
        .is_err());
    }
    #[test]
    #[ignore = "requires actual GPU under the common host lease"]
    fn matched_single_primary_is_gpu_dispatch_control() {
        assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
        let params = [0.7, 0.5, -0.6];
        let coords = [-1., 0., 1.];
        let counts = vec![vec![1., 2., 3.], vec![1., 3., 2.], vec![3., 2., 1.]];
        let (a, receipt) = update_item(
            &params,
            &[0],
            false,
            &coords,
            &[],
            1,
            3,
            0,
            &counts,
            3,
            &control(),
            crate::Device::Gpu,
        )
        .unwrap();
        require_dispatch(&receipt).unwrap();
        let direct = crate::gpu_mstep::update_grm(
            &params,
            1,
            &coords,
            &counts,
            crate::bifactor_grm::SlopePrior::None,
            &control(),
        )
        .unwrap();
        assert_eq!(a, direct.0);
        assert_eq!(receipt.objective, direct.1.objective);
    }
    #[test]
    #[ignore = "requires actual GPU under the common host lease"]
    fn vector_gpu_strict_update_matches_native_stationarity() {
        assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
        let coords = vec![-1., -0.5, -1., 0.8, 0., -0.5, 0., 0.8, 1., -0.5, 1., 0.8];
        let specific = [-0.8, 0.6];
        let truth = [0.7, -0.4, 0.5, 0.8, -0.7];
        let nodes = pack_nodes(&[0, 1], true, &coords, &specific, 2, 6, 2).unwrap();
        let counts: Vec<Vec<f64>> = nodes
            .chunks(3)
            .map(|x| {
                let eta = x.iter().zip(&truth).map(|(x, a)| x * a).sum::<f64>();
                crate::poly::grm_logprobs(eta, &truth[3..])
                    .iter()
                    .map(|p| 100. * p.exp())
                    .collect()
            })
            .collect();
        let start = [0.9, -0.2, 0.7, 0.6, -0.5];
        let (updated, receipt) = update_item(
            &start,
            &[0, 1],
            true,
            &coords,
            &specific,
            2,
            6,
            2,
            &counts,
            3,
            &control(),
            crate::Device::Gpu,
        )
        .unwrap();
        require_dispatch(&receipt).unwrap();
        assert!(receipt.parameter_update > 0 && receipt.m_step_gpu_dispatches > 0);
        let (_, gradient, _) = crate::two_tier_grm::item_neg_ll_grad_hess(
            &updated,
            &[0, 1],
            true,
            &coords,
            &specific,
            2,
            6,
            2,
            &counts,
        );
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        println!("E_VECTOR_GPU stationarity={norm:e} receipt={receipt:?} params={updated:?}");
        assert!(
            norm < control().gradient_tolerance,
            "native analytic gradient={gradient:?}"
        );
        for (a, b) in updated.iter().zip(truth) {
            assert!((a - b).abs() < 1e-8);
        }
    }
    #[test]
    fn vector_order_is_primary_slow_specific_fast() {
        assert_eq!(
            pack_nodes(
                &[2, 0],
                true,
                &[1., 2., 3., 4., 5., 6.],
                &[-1., 1.],
                3,
                2,
                2
            )
            .unwrap(),
            vec![3., 1., -1., 3., 1., 1., 6., 4., -1., 6., 4., 1.]
        );
        assert_eq!(
            pack_nodes(&[1], false, &[1., 2., 3., 4.], &[], 2, 2, 0).unwrap(),
            vec![2., 4.]
        );
    }
    #[test]
    fn invalid_shapes_reject_without_device() {
        for free in [vec![0, 0], vec![2], vec![]] {
            assert!(pack_nodes(&free, false, &[0., 1.], &[], 2, 1, 0).is_err());
        }
        assert!(pack_nodes(&[0], true, &[0., 1.], &[], 2, 1, 0).is_err());
        assert!(pack_nodes(&[0], false, &[], &[], usize::MAX, 2, 0).is_err());
        assert!(pack_nodes(&[0], false, &[f64::NAN], &[], 1, 1, 0).is_err());
    }
    #[test]
    fn strict_receipt_rejects_cpu_or_missing_criterion() {
        assert!(require_dispatch(&StageReceipt::default()).is_err());
        let mut r = StageReceipt {
            objective: 1,
            gradient: 1,
            hessian: 1,
            prior: 1,
            ..Default::default()
        };
        assert!(require_dispatch(&r).is_ok()); // stationary call still dispatches criterion
        r.cpu_numerical_fallbacks = 1;
        assert!(require_dispatch(&r).is_err());
        r.cpu_numerical_fallbacks = 0;
        r.parameter_update = 1;
        assert!(require_dispatch(&r).is_err());
    }
}
