//! Native GPU E[T|G] for two-primary orthogonal two-tier GRM banks.
//! Integrates W and the item's independent specific factor, without fixing W EAP.
//! Basis: Cai (2010, p. 589, Eqs. 11–12), cumulative GRM probabilities;
//! Cai (2015, pp. 542–543, Eqs. 14–17), independent-specific integration.
//! E[T|G] is the sum of marginal item means by linearity of expectation; this
//! function does not compute the total-score distribution or its variance.
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model with
//! applications. Psychometrika, 75, 581–612. doi:10.1007/s11336-010-9178-0.
//! Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
//! factor models with applications in test scoring, scale alignment, and model
//! fit testing. Psychometrika, 80, 535–559. doi:10.1007/s11336-014-9411-3.
use crate::two_tier_recursion::TwoTierItemParams;

#[derive(Debug)]
pub struct ExpectedRawGpu {
    pub scores: Vec<f64>,
    pub backend: String,
    pub device: String,
    pub dispatches: usize,
    pub cpu_numerical_fallbacks: usize,
    pub q_wording: usize,
    pub q_specific: usize,
}

/// W is N(mean_w,sd_w²), independent of G. Specific factors are zero-mean with
/// caller-supplied SDs. Correlated primary banks are explicitly refused.
/// G coordinates are caller-owned (e.g. G EAP); there is no W EAP argument.
/// This returns F(g)=E[T|G=g] on the zero-based category-total scale. Passing a
/// G EAP returns F(E[G|responses]), not E[F(G)|responses].
/// Uploaded f32 node transforms and absolute predictor bounds must fit within
/// f32::MAX/4. This conservative pre-dispatch refusal leaves rounding and
/// reassociation headroom; finite inputs alone do not guarantee acceptance.
#[allow(clippy::too_many_arguments)]
pub fn expected_raw_g_only(
    params: &TwoTierItemParams,
    general: &[f64],
    mean_w: f64,
    sd_w: f64,
    specific_sd: &[f64],
    primary_covariance: f64,
    q_wording: usize,
    q_specific: usize,
    device: crate::Device,
) -> Result<ExpectedRawGpu, String> {
    if device != crate::Device::Gpu {
        return Err("G-only scoring requires explicit GPU; fallback forbidden".into());
    }
    if primary_covariance != 0. {
        return Err("G-only scorer supports orthogonal primary banks only; correlated W|G integration pending".into());
    }
    let ni = params.n_items();
    let nc = params.n_cat;
    let finite_f32 = |x: f64| x.is_finite() && (x as f32).is_finite();
    if params.n_primary != 2
        || ni == 0
        || nc < 2
        || general.is_empty()
        || ni.checked_mul(2) != Some(params.a_primary.len())
        || ni.checked_mul(nc.saturating_sub(1)) != Some(params.thresholds.len())
        || params.specific_map.len() != ni
        || specific_sd.len() != params.n_specific
        || !finite_f32(mean_w)
        || !finite_f32(sd_w)
        || (sd_w as f32) <= 0.
        || specific_sd
            .iter()
            .any(|x| !finite_f32(*x) || (*x as f32) <= 0.)
        || general
            .iter()
            .chain(&params.a_primary)
            .chain(&params.a_specific)
            .chain(&params.thresholds)
            .any(|x| !finite_f32(*x))
    {
        return Err("invalid G-only scoring dimensions or nonfinite parameters".into());
    }
    for i in 0..ni {
        let s = params.specific_map[i];
        if s < -1
            || (s >= 0 && s as usize >= params.n_specific)
            || (s == -1 && params.a_specific[i] != 0.)
            || params.thresholds[i * (nc - 1)..(i + 1) * (nc - 1)]
                .windows(2)
                .any(|v| (v[0] as f32) <= (v[1] as f32))
        {
            return Err("invalid G-only scoring item support or thresholds".into());
        }
    }
    if q_wording == 0 || q_specific == 0 {
        return Err("explicit positive nuisance quadrature counts required".into());
    }
    let (limit, max_groups) = crate::gpu_mstep::limits()?;
    let input_cells = ni
        .checked_mul(nc + 3)
        .and_then(|x| x.checked_add(general.len()))
        .and_then(|x| x.checked_add(q_wording.checked_mul(2)?))
        .and_then(|x| x.checked_add(q_specific.checked_mul(2)?))
        .and_then(|x| x.checked_add(7))
        .ok_or("score size overflow")?;
    crate::gpu_mstep::checked_buffer(input_cells, limit)?;
    crate::gpu_mstep::checked_buffer(general.len(), limit)?;
    if [ni, nc, q_wording, q_specific, general.len()]
        .iter()
        .any(|n| *n > 16_777_216)
        || general.len().div_ceil(64) > max_groups
    {
        return Err("score index/dispatch limit".into());
    }
    let (wn, ww) = crate::quadrature::gh_rule(q_wording).ok_or("unsupported wording quadrature")?;
    let (sn, sw) =
        crate::quadrature::gh_rule(q_specific).ok_or("unsupported specific quadrature")?;
    // Range admission only, not host scoring: bound every shader intermediate
    // using the actual uploaded f32 values, promoted to f64 for the checks.
    // Absolute sums also protect legal backend reassociations and cancellation.
    let uploaded_abs = |x: f64| (x as f32 as f64).abs();
    let safe = f32::MAX as f64 / 4.;
    let range_error = "unsafe f32 predictor range; no GPU dispatch or CPU fallback";
    let g_bound = general.iter().copied().map(uploaded_abs).fold(0., f64::max);
    let w_node_bound = wn.iter().copied().map(uploaded_abs).fold(0., f64::max);
    let s_node_bound = sn.iter().copied().map(uploaded_abs).fold(0., f64::max);
    let w_bound = uploaded_abs(mean_w) + uploaded_abs(sd_w) * w_node_bound;
    if !w_bound.is_finite() || w_bound > safe {
        return Err(range_error.into());
    }
    for i in 0..ni {
        let mapped = params.specific_map[i];
        let s_scale = if mapped < 0 {
            0.
        } else {
            uploaded_abs(specific_sd[mapped as usize])
        };
        let s_bound = s_scale * s_node_bound;
        let intercept_bound = params.thresholds[i * (nc - 1)..(i + 1) * (nc - 1)]
            .iter()
            .copied()
            .map(uploaded_abs)
            .fold(0., f64::max);
        let predictor_bound = uploaded_abs(params.a_primary[2 * i]) * g_bound
            + uploaded_abs(params.a_primary[2 * i + 1]) * w_bound
            + uploaded_abs(params.a_specific[i]) * s_bound
            + intercept_bound;
        // Multiplication may be reassociated under backend fast math. Bound
        // both slope*SD and every two-factor subset of slope*SD*node, including
        // q=1's zero node which would otherwise conceal the first overflow.
        let w_slope = uploaded_abs(params.a_primary[2 * i + 1]);
        let s_slope = uploaded_abs(params.a_specific[i]);
        let w_scale = uploaded_abs(sd_w);
        let products = [
            w_slope * uploaded_abs(mean_w),
            w_slope * w_scale,
            w_slope * w_node_bound,
            w_scale * w_node_bound,
            s_slope * s_scale,
            s_slope * s_node_bound,
            s_scale * s_node_bound,
        ];
        if products.iter().any(|x| !x.is_finite() || *x > safe)
            || !s_bound.is_finite()
            || s_bound > safe
            || !predictor_bound.is_finite()
            || predictor_bound > safe
        {
            return Err(range_error.into());
        }
    }
    let mut data = Vec::with_capacity(input_cells);
    data.extend([
        ni as f32,
        nc as f32,
        general.len() as f32,
        q_wording as f32,
        q_specific as f32,
        mean_w as f32,
        sd_w as f32,
    ]);
    data.extend(general.iter().map(|x| *x as f32));
    for i in 0..ni {
        let s = params.specific_map[i];
        data.extend([
            params.a_primary[2 * i] as f32,
            params.a_primary[2 * i + 1] as f32,
            params.a_specific[i] as f32,
            if s < 0 {
                0.
            } else {
                specific_sd[s as usize] as f32
            },
        ]);
        data.extend(
            params.thresholds[i * (nc - 1)..(i + 1) * (nc - 1)]
                .iter()
                .map(|x| *x as f32),
        );
    }
    for (&x, &w) in wn.iter().zip(ww) {
        data.extend([x as f32, w as f32]);
    }
    for (&x, &w) in sn.iter().zip(sw) {
        data.extend([x as f32, w as f32]);
    }
    debug_assert_eq!(data.len(), input_cells);
    let values = crate::gpu_mstep::dispatch(
        include_str!("two_tier_gpu_score.wgsl"),
        &data,
        general.len(),
        general.len().div_ceil(64) as u32,
        "main",
    )?;
    if values
        .iter()
        .any(|x| !x.is_finite() || *x < 0. || *x > (ni * (nc - 1)) as f32)
    {
        return Err("invalid GPU expected raw return; no CPU fallback".into());
    }
    let context = crate::gpu::GpuContext::get().ok_or("GPU context missing after dispatch")?;
    Ok(ExpectedRawGpu {
        scores: values.iter().map(|x| *x as f64).collect(),
        backend: context.backend().into(),
        device: context.device_name().into(),
        dispatches: 1,
        cpu_numerical_fallbacks: 0,
        q_wording,
        q_specific,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn params() -> TwoTierItemParams {
        TwoTierItemParams {
            a_primary: vec![0.7, 1.2, 1.1, -0.6],
            a_specific: vec![0.4, 0.],
            thresholds: vec![0.8, -0.5, 0.4, -0.9],
            specific_map: vec![0, -1],
            n_primary: 2,
            n_specific: 1,
            n_cat: 3,
        }
    }
    #[test]
    fn rejects_correlated_primary_or_non_gpu_without_dispatch() {
        let p = params();
        for d in [crate::Device::Cpu, crate::Device::Auto] {
            assert!(expected_raw_g_only(&p, &[0.], 0., 1., &[1.], 0., 15, 15, d).is_err());
        }
        for covariance in [0.2, f64::NAN] {
            assert!(expected_raw_g_only(
                &p,
                &[0.],
                0.,
                1.,
                &[1.],
                covariance,
                15,
                15,
                crate::Device::Gpu
            )
            .is_err());
        }
        for (sd, qw, qs) in [(0., 15, 15), (1., 0, 15), (1., 15, 0)] {
            assert!(
                expected_raw_g_only(&p, &[0.], 0., sd, &[1.], 0., qw, qs, crate::Device::Gpu)
                    .is_err()
            );
        }
    }
    #[test]
    #[ignore = "requires actual GPU and common host lease"]
    fn integrates_w_and_specific_against_native_lw() {
        assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
        let p = params();
        let general = [-0.8, 0., 0.9];
        let got = expected_raw_g_only(
            &p,
            &general,
            0.3,
            1.2,
            &[0.8],
            0.,
            15,
            15,
            crate::Device::Gpu,
        )
        .unwrap();
        assert_eq!(got.dispatches, 1);
        assert_eq!(got.cpu_numerical_fallbacks, 0);
        assert!(!got.backend.is_empty() && !got.device.is_empty());
        let (wn, ww) = crate::quadrature::gh_rule(15).unwrap();
        let (sn, sw) = crate::quadrature::gh_rule(15).unwrap();
        let scaled: Vec<f64> = sn.iter().map(|x| x * 0.8).collect();
        for (i, &g) in general.iter().enumerate() {
            let coordinates: Vec<f64> = wn.iter().flat_map(|x| [g, 0.3 + 1.2 * x]).collect();
            let means =
                crate::two_tier_recursion::two_tier_expected_raw(&p, &coordinates, &scaled, sw)
                    .unwrap();
            let reference: f64 = means.iter().zip(ww).map(|(m, w)| m * w).sum();
            assert!(
                (got.scores[i] - reference).abs() < 2e-5,
                "{} vs {reference}",
                got.scores[i]
            );
        }
        let plugin =
            crate::two_tier_recursion::two_tier_expected_raw(&p, &[general[0], 0.3], &scaled, sw)
                .unwrap();
        assert!(
            (got.scores[0] - plugin[0]).abs() > 1e-3,
            "must integrate W, not plug its mean"
        );
        let shifted = expected_raw_g_only(
            &p,
            &general,
            1.,
            1.2,
            &[0.8],
            0.,
            15,
            15,
            crate::Device::Gpu,
        )
        .unwrap();
        assert!((got.scores[0] - shifted.scores[0]).abs() > 1e-3);
        println!("E_G_ONLY_GPU {got:?}");
    }
}
