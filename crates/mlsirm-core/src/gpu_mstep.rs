//! Strict device execution of the canonical GRM expected-count criterion.
//!
//! Chalmers (2012, p. 3, Eq. 2; pp. 5–6, Eqs. 6–11) supplies the unchanged
//! category-probability and EM contracts. Equations 6–11 explicitly illustrate
//! dichotomous EM; Eq. 2 supplies the ordinal category probabilities used here.
//! Device precision and Newton controls are implementation choices, not accuracy
//! guarantees from that paper. Hardware tests follow the repository's explicit
//! ignored physical-GPU gate: run `cargo test -p mlsirm-core --lib gpu_mstep
//! -- --include-ignored --nocapture --test-threads=1` on hardware. Ordinary CI
//! skips these dispatch tests; that skip is not GPU execution evidence.
//!
//! Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
//! for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
//! https://doi.org/10.18637/jss.v048.i06

use crate::bifactor_grm::SlopePrior;

#[derive(Debug)]
pub struct Evaluation {
    pub objective: f64,
    pub gradient: Vec<f64>,
    pub hessian: Vec<Vec<f64>>,
    pub backend: String,
    pub device: String,
    pub dispatches: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StageReceipt {
    pub objective: usize,
    pub gradient: usize,
    pub hessian: usize,
    // Includes explicit None-prior evaluation; not evidence of MAP fitting.
    pub prior: usize,
    pub solve: usize,
    pub line_search: usize,
    pub parameter_update: usize,
    pub e_step_gpu_dispatches: usize,
    pub m_step_gpu_dispatches: usize,
    pub cpu_numerical_fallbacks: usize,
}
impl StageReceipt {
    pub(crate) fn add(&mut self, rhs: &Self) {
        self.objective += rhs.objective;
        self.gradient += rhs.gradient;
        self.hessian += rhs.hessian;
        self.prior += rhs.prior;
        self.solve += rhs.solve;
        self.line_search += rhs.line_search;
        self.parameter_update += rhs.parameter_update;
        self.e_step_gpu_dispatches += rhs.e_step_gpu_dispatches;
        self.m_step_gpu_dispatches += rhs.m_step_gpu_dispatches;
        self.cpu_numerical_fallbacks += rhs.cpu_numerical_fallbacks;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NewtonControl {
    pub iterations: usize,
    pub backtracks: usize,
    pub ridge: f64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub pivot_tolerance: f64,
    pub armijo: f64,
}
impl NewtonControl {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.iterations == 0
            || self.backtracks == 0
            || self.iterations > 16_777_216
            || self.backtracks > 16_777_216
            || self
                .iterations
                .checked_mul(self.backtracks + 1)
                .is_none_or(|n| n > 16_777_216)
        {
            return Err("invalid GPU Newton iteration controls".into());
        }
        for x in [
            self.ridge,
            self.gradient_tolerance,
            self.step_tolerance,
            self.pivot_tolerance,
            self.armijo,
        ] {
            if !x.is_finite() || !(x as f32).is_finite() || x <= 0.0 || (x as f32) <= 0.0 {
                return Err("GPU Newton controls must be finite positive f32 values".into());
            }
        }
        if self.armijo >= 1.0 {
            return Err("Armijo constant must be below one".into());
        }
        Ok(())
    }
}

fn checked_layout(
    np: usize,
    off: usize,
    nn: usize,
    limit: usize,
    workgroups: usize,
) -> Result<(usize, usize), String> {
    let bad = || "GPU M-step resource bounds exceeded before allocation".to_string();
    if off == 0 || np <= off || nn == 0 || np > 16_777_216 || nn > 16_777_216 {
        return Err(bad());
    }
    let input = np
        .checked_add(1)
        .and_then(|v| v.checked_mul(nn))
        .and_then(|v| v.checked_add(np))
        .and_then(|v| v.checked_add(13))
        .ok_or_else(bad)?;
    let width = np
        .checked_mul(np)
        .and_then(|v| v.checked_add(np))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(bad)?;
    checked_buffer(input, limit)?;
    checked_buffer(width, limit)?;
    if width.div_ceil(64) > workgroups {
        return Err(bad());
    }
    Ok((input, width))
}

pub(crate) fn checked_buffer(cells: usize, limit: usize) -> Result<usize, String> {
    let bytes = cells
        .checked_mul(4)
        .ok_or("GPU buffer size overflow before allocation")?;
    if cells == 0 || cells > u32::MAX as usize || bytes > limit {
        return Err("GPU buffer resource bounds exceeded before allocation".into());
    }
    Ok(bytes)
}

fn hardware_adapter(kind: wgpu::DeviceType) -> bool {
    matches!(
        kind,
        wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::VirtualGpu
    )
}

pub(crate) fn limits() -> Result<(usize, usize), String> {
    let ctx =
        crate::gpu::GpuContext::get().ok_or("GPU unavailable; CPU numerical fallback forbidden")?;
    if !hardware_adapter(ctx.device_type) {
        return Err(
            "strict GPU execution requires a hardware GPU; software/CPU adapter rejected".into(),
        );
    }
    let lim = ctx.device.limits();
    Ok((
        lim.max_buffer_size
            .min(lim.max_storage_buffer_binding_size as u64) as usize,
        lim.max_compute_workgroups_per_dimension as usize,
    ))
}

// Readback and error scopes establish completion; enqueueing alone never earns a receipt.
pub(crate) fn dispatch(
    source: &str,
    data: &[f32],
    output_cells: usize,
    workgroups: u32,
    entry: &str,
) -> Result<Vec<f32>, String> {
    use wgpu::util::DeviceExt;
    let ctx =
        crate::gpu::GpuContext::get().ok_or("GPU unavailable; CPU numerical fallback forbidden")?;
    let device = &ctx.device;
    let (limit, max_groups) = limits()?;
    checked_buffer(data.len(), limit)?;
    let bytes = checked_buffer(output_cells, limit)? as u64;
    if workgroups == 0 || workgroups as usize > max_groups {
        return Err("GPU dispatch bounds exceeded".into());
    }
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("strict-grm-input"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strict-grm-output"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strict-grm-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("strict-grm"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("strict-grm"),
        layout: None,
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &read, 0, bytes);
    ctx.queue.submit([encoder.finish()]);
    // Pop all scopes even when a pipeline/dispatch is rejected.
    let validation_error = pollster::block_on(validation.pop());
    let internal_error = pollster::block_on(internal.pop());
    let oom_error = pollster::block_on(oom.pop());
    if let Some(error) = validation_error.or(internal_error).or(oom_error) {
        return Err(format!("GPU execution rejected: {error}"));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |x| {
        let _ = tx.send(x);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let view = read
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let v = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    read.unmap();
    Ok(v)
}

fn pack(
    params: &[f64],
    off: usize,
    nodes: &[f64],
    counts: &[Vec<f64>],
    prior: SlopePrior,
) -> Result<(Vec<f32>, usize), String> {
    let (limit, groups) = limits()?;
    let (input, width) = checked_layout(params.len(), off, counts.len(), limit, groups)?;
    let np = params.len();
    let node_cells = counts
        .len()
        .checked_mul(off)
        .ok_or("GPU node size overflow")?;
    if nodes.len() != node_cells || counts.iter().any(|r| r.len() != np - off + 1) {
        return Err("invalid GRM M-step shape".into());
    }
    if !params
        .iter()
        .chain(nodes)
        .all(|x| x.is_finite() && (*x as f32).is_finite())
        || params[off..]
            .windows(2)
            .any(|x| (x[0] as f32) <= (x[1] as f32))
        || counts
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || *x < 0.0 || !(*x as f32).is_finite())
    {
        return Err("nonfinite, negative count or unordered GRM boundary".into());
    }
    prior.validate()?;
    let prior_values = match prior {
        SlopePrior::None => [0.0, 0.0, 1.0],
        SlopePrior::Lognormal { mu, sd } => {
            if !(mu as f32).is_finite()
                || !(sd as f32).is_finite()
                || (sd as f32) <= 0.0
                || params[..off].iter().any(|a| (*a as f32) == 0.0)
            {
                return Err("unsupported GPU prior value".into());
            }
            [1.0, mu as f32, sd as f32]
        }
    };
    let mut data = Vec::with_capacity(input);
    data.extend([np as f32, off as f32, counts.len() as f32]);
    data.extend(params.iter().map(|x| *x as f32));
    for (n, row) in counts.iter().enumerate() {
        data.extend(nodes[n * off..(n + 1) * off].iter().map(|x| *x as f32));
        data.extend(row.iter().map(|x| *x as f32));
    }
    data.extend(prior_values);
    data.extend([0.0; 7]);
    Ok((data, width))
}

pub fn evaluate_grm(
    params: &[f64],
    has_specific: bool,
    node_g: &[f64],
    node_s: &[f64],
    counts: &[Vec<f64>],
) -> Result<Evaluation, String> {
    let off = if has_specific { 2 } else { 1 };
    // Bounds checked before even the coordinate-packing allocation.
    let (limit, groups) = limits()?;
    checked_layout(params.len(), off, counts.len(), limit, groups)?;
    if counts.len() != node_g.len() || (has_specific && node_s.len() != counts.len()) {
        return Err("invalid GRM M-step shape".into());
    }
    let mut nodes = Vec::with_capacity(counts.len() * off);
    for n in 0..counts.len() {
        nodes.push(node_g[n]);
        if has_specific {
            nodes.push(node_s[n]);
        }
    }
    evaluate_grm_prior(params, off, &nodes, counts, SlopePrior::None)
}

pub fn evaluate_grm_prior(
    params: &[f64],
    off: usize,
    nodes: &[f64],
    counts: &[Vec<f64>],
    prior: SlopePrior,
) -> Result<Evaluation, String> {
    let (data, width) = pack(params, off, nodes, counts, prior)?;
    let source = concat!(
        include_str!("gpu_mstep_common.wgsl"),
        include_str!("gpu_mstep_prior.wgsl"),
        include_str!("gpu_mstep.wgsl")
    );
    let raw = dispatch(source, &data, width, width.div_ceil(64) as u32, "main")?;
    if raw.iter().any(|x| !x.is_finite()) {
        return Err("GPU M-step nonfinite output; CPU fallback forbidden".into());
    }
    let v: Vec<f64> = raw.iter().map(|x| *x as f64).collect();
    let np = params.len();
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable")?;
    Ok(Evaluation {
        objective: v[0],
        gradient: v[1..1 + np].to_vec(),
        hessian: v[1 + np..].chunks(np).map(|r| r.to_vec()).collect(),
        backend: ctx.backend().into(),
        device: ctx.device_name().into(),
        dispatches: 1,
    })
}

pub fn update_grm(
    params: &[f64],
    off: usize,
    nodes: &[f64],
    counts: &[Vec<f64>],
    prior: SlopePrior,
    control: &NewtonControl,
) -> Result<(Vec<f64>, StageReceipt), String> {
    control.validate()?;
    let (limit, groups) = limits()?;
    let (_, width) = checked_layout(params.len(), off, counts.len(), limit, groups)?;
    let meta = width
        .checked_add(params.len().checked_mul(3).ok_or("GPU scratch overflow")?)
        .ok_or("GPU scratch overflow")?;
    let cells = meta.checked_add(9).ok_or("GPU scratch overflow")?;
    checked_buffer(cells, limits()?.0)?;
    let (mut data, width) = pack(params, off, nodes, counts, prior)?;
    let ctl = data.len() - 7;
    data[ctl..].copy_from_slice(&[
        control.ridge as f32,
        control.iterations as f32,
        control.backtracks as f32,
        control.gradient_tolerance as f32,
        control.step_tolerance as f32,
        control.pivot_tolerance as f32,
        control.armijo as f32,
    ]);

    let source = concat!(
        include_str!("gpu_mstep_common.wgsl"),
        include_str!("gpu_mstep_prior.wgsl"),
        include_str!("gpu_mstep_optimize.wgsl")
    );
    let v = dispatch(source, &data, cells, 1, "main")?;
    let status = v[meta] as u32;
    if status != 1 && status != 5 {
        let reason = match status {
            2 => "nonfinite criterion",
            3 => "linear solve failure",
            4 => "line-search failure",
            _ => "invalid completion status",
        };
        let np = params.len();
        let step = width + 2 * np;
        let f0 = v[0] as f64;
        let gnorm = v[1..1 + np]
            .iter()
            .fold(0.0f64, |m, x| m.max((*x as f64).abs()));
        let directional: f64 = (0..np).map(|j| v[1 + j] as f64 * v[step + j] as f64).sum();
        return Err(format!(
            "GPU M-step {reason}; CPU numerical fallback forbidden; completed stages={:?}; \
             last objective={f0:e} gradient_max_abs={gnorm:e} newton_directional={directional:e} \
             f32_objective_ulp={:e}",
            &v[meta..],
            f0.abs() * f32::EPSILON as f64
        ));
    }
    let p: Vec<f64> = v[width..width + params.len()]
        .iter()
        .map(|x| *x as f64)
        .collect();
    if p.iter().any(|x| !x.is_finite()) || p[off..].windows(2).any(|x| x[0] <= x[1]) {
        return Err("invalid GPU updated parameters".into());
    }
    let c = &v[meta + 1..meta + 8];
    if c.iter()
        .any(|x| !x.is_finite() || *x < 0.0 || x.fract() != 0.0)
    {
        return Err("invalid device stage receipt".into());
    }
    Ok((
        p,
        StageReceipt {
            objective: c[0] as usize,
            gradient: c[1] as usize,
            hessian: c[2] as usize,
            prior: c[3] as usize,
            solve: c[4] as usize,
            line_search: c[5] as usize,
            parameter_update: c[6] as usize,
            m_step_gpu_dispatches: usize::from(c[6] > 0.0),
            ..Default::default()
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_mstep_rejects_software_adapter_and_invalid_sizes() {
        assert!(!hardware_adapter(wgpu::DeviceType::Cpu));
        assert!(!hardware_adapter(wgpu::DeviceType::Other));
        assert!(hardware_adapter(wgpu::DeviceType::IntegratedGpu));
        assert!(checked_layout(3, 3, 1, usize::MAX, usize::MAX).is_err());
        assert!(checked_layout(3, 1, 0, usize::MAX, usize::MAX).is_err());
        assert!(checked_buffer(usize::MAX, usize::MAX).is_err());
        assert!(checked_buffer(100, 399).is_err());
    }
    #[test]
    fn gpu_mstep_checked_preallocation_overflow() {
        for (np, nn) in [(usize::MAX, 1), (3, usize::MAX), (1 << 30, 2)] {
            let r = std::panic::catch_unwind(|| checked_layout(np, 1, nn, usize::MAX, usize::MAX));
            assert!(
                r.is_ok(),
                "dimensions must reject before arithmetic/allocation"
            );
            assert!(r.unwrap().is_err());
        }
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_narrow_threshold_derivatives() {
        for gap in [1e-4, 1e-6, 1e-8] {
            let p = [1.0, gap, 0.0];
            let x = 0.7;
            let actual = evaluate_grm(&p, false, &[x], &[], &[vec![0.0, 1.0, 0.0]]).unwrap();
            let u = x + gap;
            let v = x;
            let su = 1.0 / (1.0 + (-u).exp());
            let sv = 1.0 / (1.0 + (-v).exp());
            let a = su * (1.0 - su);
            let b = sv * (1.0 - sv);
            let r = 1.0 / gap.exp_m1();
            let t = r * (1.0 + r);
            let grad = [(su + sv - 1.0) * x, su - 1.0 - r, sv + r];
            let hess = [
                [(a + b) * x * x, a * x, b * x],
                [a * x, a + t, -t],
                [b * x, -t, b + t],
            ];
            let f = -(-(-u).exp().ln_1p() - v.exp().ln_1p() + (-(-gap).exp_m1()).ln());
            assert!((actual.objective - f).abs() < 1e-5);
            for j in 0..3 {
                assert!(
                    (actual.gradient[j] - grad[j]).abs() < 2e-5 * (1.0 + grad[j].abs()),
                    "gap={gap} gradient {j}: {} vs {}",
                    actual.gradient[j],
                    grad[j]
                );
                for k in 0..3 {
                    assert!(
                        (actual.hessian[j][k] - hess[j][k]).abs() < 2e-5 * (1.0 + hess[j][k].abs()),
                        "gap={gap} hessian {j},{k}: {} vs {}",
                        actual.hessian[j][k],
                        hess[j][k]
                    );
                }
            }
        }
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_small_positive_tail_preserved() {
        let actual = evaluate_grm(&[1.0, 0.0], false, &[20.0], &[], &vec![vec![0.0, 1.0]]).unwrap();
        let expected = (-20.0f64).exp().ln_1p();
        assert!((actual.objective / expected - 1.0).abs() < 1e-5);
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_narrow_tail_is_finite() {
        let p = [1.0, 0.00000001, 0.0];
        let g = [80.0];
        let c = vec![vec![0.0, 1.0, 0.0]];
        let expected = crate::bifactor_grm::item_neg_ll_grad_hess(&p, false, &g, &[], &c, false);
        let actual = evaluate_grm(&p, false, &g, &[], &c).expect("finite narrow tail");
        assert!((actual.objective - expected.0).abs() < 0.001);
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_real_dispatch_matches_analytic_reference() {
        let p = [1.1, 0.6, 0.8, -0.7];
        let g = [-1.0, -0.2, 0.5, 1.2];
        let s = [0.4, -0.7, 0.8, -0.1];
        let c = vec![
            vec![2.0, 3.0, 1.0],
            vec![1.0, 4.0, 2.0],
            vec![0.0, 2.0, 5.0],
            vec![1.0, 1.0, 4.0],
        ];
        let expected = crate::bifactor_grm::item_neg_ll_grad_hess(&p, true, &g, &s, &c, false);
        let actual = evaluate_grm(&p, true, &g, &s, &c).expect("strict GPU evaluation");
        assert_eq!(
            actual.dispatches, 1,
            "must execute actual GPU M-step kernel"
        );
        assert!(!actual.backend.is_empty());
        assert!((actual.objective - expected.0).abs() < 2e-5);
        for j in 0..p.len() {
            assert!(
                (actual.gradient[j] - expected.1[j]).abs() < 2e-5,
                "gradient {j}"
            );
            for k in 0..p.len() {
                assert!(
                    (actual.hessian[j][k] - expected.2[j][k]).abs() < 2e-5,
                    "hessian {j},{k}"
                );
            }
        }
        println!(
            "GPU_MSTEP backend={} device={} dispatches={} objective_error={}",
            actual.backend,
            actual.device,
            actual.dispatches,
            (actual.objective - expected.0).abs()
        );
    }
    fn control() -> NewtonControl {
        NewtonControl {
            iterations: 4,
            backtracks: 25,
            ridge: 1e-3,
            gradient_tolerance: 0.005,
            step_tolerance: 1e-4,
            pivot_tolerance: 1e-10,
            armijo: 1e-4,
        }
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_signed_prior_matches_canonical_contract() {
        let p = [-1.1, 0.6, 0.8, -0.7];
        let nodes = [-1.0, 0.4, 0.5, -0.7];
        let counts = vec![vec![2.0, 3.0, 1.0], vec![1.0, 2.0, 4.0]];
        let plain = evaluate_grm_prior(&p, 2, &nodes, &counts, SlopePrior::None).unwrap();
        let with = evaluate_grm_prior(
            &p,
            2,
            &nodes,
            &counts,
            SlopePrior::Lognormal { mu: 0.2, sd: 0.8 },
        )
        .unwrap();
        let mut objective = plain.objective;
        for j in 0..2 {
            let mut g = 0.0;
            crate::bifactor_grm::add_lnorm_abs_slope_prior(p[j], 0.2, 0.8, &mut objective, &mut g);
            let h = crate::bifactor_grm::lnorm_abs_slope_prior_curvature(p[j], 0.2, 0.8);
            assert!((with.gradient[j] - plain.gradient[j] - g).abs() < 2e-5);
            assert!((with.hessian[j][j] - plain.hessian[j][j] - h).abs() < 2e-5);
        }
        assert!((with.objective - objective).abs() < 2e-5);
        assert!(evaluate_grm_prior(
            &[0.0, 0.8, -0.7],
            1,
            &[-1.0, 0.5],
            &counts,
            SlopePrior::Lognormal { mu: 0.2, sd: 0.8 }
        )
        .is_err());
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_update_and_deterministic_failures() {
        let params = [1.1, 0.8, -0.7];
        let nodes = [-1.0, -0.2, 0.5, 1.2];
        let counts = vec![
            vec![8.0, 3.0, 1.0],
            vec![3.0, 4.0, 2.0],
            vec![1.0, 2.0, 5.0],
            vec![1.0, 1.0, 9.0],
        ];
        let prior = SlopePrior::Lognormal { mu: 0.0, sd: 1.0 };
        let before = evaluate_grm_prior(&params, 1, &nodes, &counts, prior).unwrap();
        let (after, receipt) = update_grm(&params, 1, &nodes, &counts, prior, &control()).unwrap();
        let objective = evaluate_grm_prior(&after, 1, &nodes, &counts, prior)
            .unwrap()
            .objective;
        assert!(objective < before.objective);
        assert!(
            receipt.solve > 0
                && receipt.parameter_update > 0
                && receipt.line_search > 0
                && receipt.prior > 0
        );
        assert_eq!(receipt.cpu_numerical_fallbacks, 0);
        println!(
            "GPU_UPDATE_RECEIPT {}",
            serde_json::to_string(&receipt).unwrap()
        );
        let mut bad = control();
        bad.pivot_tolerance = 1e30;
        let error = update_grm(&params, 1, &nodes, &counts, prior, &bad).unwrap_err();
        assert!(error.contains("linear solve failure"));
        println!("GPU_SOLVE_REJECTION {error}");
        bad = control();
        bad.ridge = 1e30;
        let error = update_grm(&params, 1, &nodes, &counts, prior, &bad).unwrap_err();
        assert!(error.contains("line-search failure"));
        println!("GPU_LINE_SEARCH_REJECTION {error}");
    }
    fn synthetic() -> (Vec<usize>, usize, usize) {
        let n = 180;
        let items = 4;
        let mut seed = 72915u64;
        let mut y = Vec::new();
        for p in 0..n {
            let theta = (p as f64 - (n - 1) as f64 / 2.0) / (n as f64 / 5.0);
            for i in 0..items {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((seed >> 11) as f64) / ((1u64 << 53) as f64);
                let a = 0.7 + 0.2 * i as f64;
                let ge1 = 1.0 / (1.0 + (-a * theta - 0.8).exp());
                let ge2 = 1.0 / (1.0 + (-a * theta + 0.8).exp());
                y.push(if u < 1.0 - ge1 {
                    0
                } else if u < 1.0 - ge2 {
                    1
                } else {
                    2
                });
            }
        }
        (y, n, items)
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_canonical_strict_fit_converges() {
        let (y, n, items) = synthetic();
        let ctrl = control();
        let cfg = crate::grm::GrmConfig {
            max_iter: 100,
            tol: 1e-4,
            q: 121,
            ridge: ctrl.ridge,
            newton_iter: ctrl.iterations,
            xi_rule: crate::marginal::XiRuleKind::GaussHermite,
            xi_points: 121,
            xi_seed: 17,
        };
        let (fit, receipt) =
            crate::grm::fit_grm_gpu_strict(&y, None, &vec![1; items], n, items, 1, 3, &cfg, &ctrl)
                .expect("canonical strict fit");
        assert!(fit.converged && fit.n_iter > 1);
        assert_eq!(fit.termination_reason, "tolerance_met");
        assert!(fit
            .slope
            .iter()
            .chain(&fit.threshold)
            .chain(&fit.theta)
            .all(|x| x.is_finite()));
        assert!(fit.threshold.chunks(2).all(|x| x[0] > x[1]));
        assert!(
            receipt.objective > 0
                && receipt.gradient > 0
                && receipt.hessian > 0
                && receipt.prior > 0
                && receipt.solve > 0
                && receipt.line_search > 0
                && receipt.parameter_update > 0
        );
        assert_eq!(receipt.e_step_gpu_dispatches, fit.n_iter + 1);
        assert!(receipt.m_step_gpu_dispatches > 0);
        assert_eq!(receipt.cpu_numerical_fallbacks, 0);
        let cpu = crate::grm::fit_grm(&y, None, &vec![1; items], n, items, 1, 3, &cfg).unwrap();
        assert!((fit.loglik_trace.last().unwrap() - cpu.loglik_trace.last().unwrap()).abs() < 0.1);
        for (a, b) in fit.slope.iter().zip(&cpu.slope) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
        println!(
            "GPU_FIT_RECEIPT {}",
            serde_json::to_string(&receipt).unwrap()
        );
        println!(
            "GPU_FIT iterations={} loglik={} cpu_reference_loglik={}",
            fit.n_iter,
            fit.loglik_trace.last().unwrap(),
            cpu.loglik_trace.last().unwrap()
        );
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_canonical_strict_rejects_nonconvergence() {
        let (y, n, items) = synthetic();
        let ctrl = control();
        let cfg = crate::grm::GrmConfig {
            max_iter: 1,
            tol: 1e-12,
            q: 121,
            ridge: ctrl.ridge,
            newton_iter: ctrl.iterations,
            xi_rule: crate::marginal::XiRuleKind::GaussHermite,
            xi_points: 121,
            xi_seed: 17,
        };
        let err =
            crate::grm::fit_grm_gpu_strict(&y, None, &vec![1; items], n, items, 1, 3, &cfg, &ctrl)
                .unwrap_err();
        assert!(err.contains("nonconvergence"), "{err}");
        println!("GPU_NEGATIVE_REJECTION {err}");
    }
    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn gpu_mstep_estep_matches_missing_multidimensional_reference() {
        let y = [0, 1, 1, 0, 2, 2];
        let observed = [true, false, true, true, true, true];
        let nodes = [-1.0, 0.4, 0.0, -0.2, 1.0, 0.7];
        let logw = [0.25f64.ln(), 0.5f64.ln(), 0.25f64.ln()];
        let params = vec![vec![1.2, 0.3, 0.5, -0.8], vec![0.9, 0.4, -0.6]];
        let dims = vec![vec![0, 1], vec![1]];
        let actual = crate::gpu_grm_estep::expectation(
            &y,
            Some(&observed),
            3,
            2,
            2,
            3,
            &nodes,
            &logw,
            &params,
            &dims,
        )
        .unwrap();
        let mut counts = vec![vec![vec![0.0; 3]; 3]; 2];
        let mut theta = vec![0.0; 6];
        let mut ll = 0.0;
        for p in 0..3 {
            let mut l = logw;
            for nd in 0..3 {
                for i in 0..2 {
                    if observed[p * 2 + i] {
                        let base: f64 = dims[i]
                            .iter()
                            .enumerate()
                            .map(|(j, d)| params[i][j] * nodes[nd * 2 + d])
                            .sum();
                        l[nd] += crate::poly::grm_logprobs(base, &params[i][dims[i].len()..])
                            [y[p * 2 + i]];
                    }
                }
            }
            let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = l.iter().map(|v| (v - mx).exp()).sum();
            ll += mx + den.ln();
            for nd in 0..3 {
                let post = (l[nd] - mx).exp() / den;
                for i in 0..2 {
                    if observed[p * 2 + i] {
                        counts[i][nd][y[p * 2 + i]] += post;
                    }
                }
                for d in 0..2 {
                    theta[p * 2 + d] += post * nodes[nd * 2 + d];
                }
            }
        }
        assert!((actual.loglik - ll).abs() < 2e-5);
        for (a, b) in actual.theta.iter().zip(&theta) {
            assert!((a - b).abs() < 2e-5);
        }
        for (a, b) in actual
            .counts
            .iter()
            .flatten()
            .flatten()
            .zip(counts.iter().flatten().flatten())
        {
            assert!((a - b).abs() < 2e-5);
        }
        assert!(crate::gpu_grm_estep::expectation(
            &[],
            None,
            usize::MAX,
            2,
            2,
            3,
            &[],
            &[],
            &[],
            &[]
        )
        .is_err());
    }
}

// Synthetic regression controls; hardware execution is an explicit gate.
#[cfg(test)]
mod independent_review {
    use super::*;

    fn hardware_available() -> bool {
        match limits() {
            Ok(_) => true,
            Err(error) => {
                eprintln!("INDEPENDENT_GPU_NOT_EXECUTED {error}");
                false
            }
        }
    }

    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn independent_review_zero_mass_quadrature() {
        assert!(
            hardware_available(),
            "this evidence cohort requires hardware GPU"
        );
        let params = vec![vec![1.0, 0.0]];
        let dims = vec![vec![0]];
        let nodes = [-1.0, 0.0, 1.0];
        let with_zero = crate::gpu_grm_estep::expectation(
            &[0, 1],
            None,
            2,
            1,
            1,
            2,
            &nodes,
            &[f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY],
            &params,
            &dims,
        );
        assert!(
            with_zero.is_ok(),
            "valid zero-mass nodes rejected: {:?}",
            with_zero.err()
        );
        let with_zero = with_zero.unwrap();
        let positive = crate::gpu_grm_estep::expectation(
            &[0, 1],
            None,
            2,
            1,
            1,
            2,
            &[0.0],
            &[0.0],
            &params,
            &dims,
        )
        .unwrap();
        assert_eq!(with_zero.loglik, positive.loglik);
        assert_eq!(with_zero.theta, positive.theta);
        assert_eq!(with_zero.counts[0][0], vec![0.0, 0.0]);
        assert_eq!(with_zero.counts[0][1], positive.counts[0][0]);
        assert_eq!(with_zero.counts[0][2], vec![0.0, 0.0]);
        for invalid in [
            vec![f64::NEG_INFINITY; 3],
            vec![f64::NAN, 0.0, 0.0],
            vec![f64::INFINITY, 0.0, 0.0],
        ] {
            assert!(crate::gpu_grm_estep::expectation(
                &[0, 1],
                None,
                2,
                1,
                1,
                2,
                &nodes,
                &invalid,
                &params,
                &dims,
            )
            .is_err());
        }
    }

    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn independent_review_native_gh_zero_tails() {
        assert!(
            hardware_available(),
            "this evidence cohort requires hardware GPU"
        );
        let x = crate::nodes::build_xi_nodes(crate::nodes::XiRule::GaussHermite { q_xi: 481 }, 1)
            .unwrap();
        let zeros = x.logw.iter().filter(|w| **w == f64::NEG_INFINITY).count();
        println!(
            "INDEPENDENT_GH nodes={} zero_mass_nodes={zeros}",
            x.logw.len()
        );
        assert!(
            zeros > 0,
            "fixture must reach the real zero-weight producer"
        );
        let actual = crate::gpu_grm_estep::expectation(
            &[0, 1],
            None,
            2,
            1,
            1,
            2,
            &x.grid,
            &x.logw,
            &[vec![1.0, 0.0]],
            &[vec![0]],
        );
        assert!(
            actual.is_ok(),
            "native valid GH rule rejected: {:?}",
            actual.err()
        );
    }

    #[test]
    #[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
    fn independent_review_loglik_replication_stability() {
        assert!(
            hardware_available(),
            "this evidence cohort requires hardware GPU"
        );
        let items = 6;
        let n = 10_000;
        let params = vec![vec![1.0, 0.0]; items];
        let dims = vec![vec![0]; items];
        let single = crate::gpu_grm_estep::expectation(
            &vec![1; items],
            None,
            1,
            items,
            1,
            2,
            &[0.0],
            &[0.0],
            &params,
            &dims,
        )
        .unwrap();
        let bulk = crate::gpu_grm_estep::expectation(
            &vec![1; n * items],
            None,
            n,
            items,
            1,
            2,
            &[0.0],
            &[0.0],
            &params,
            &dims,
        )
        .unwrap();
        // Compare two genuine library GPU results; no independent estimator.
        // This is an implementation test bound, not a paper accuracy guarantee.
        let expected = single.loglik * n as f64;
        let error = (bulk.loglik - expected).abs();
        let bound = 4.0 * f32::EPSILON as f64 * (1.0 + expected.abs());
        println!("INDEPENDENT_LOGLIK n={n} items={items} bulk={} repeated_single={expected} error={error} bound={bound}", bulk.loglik);
        assert!(
            error <= bound,
            "serial f32 accumulation exceeds replication bound"
        );
        // The fitter uses cfg.tol * (1 + abs(previous_ll)). This fixture
        // checks that scale, not convergence of an arbitrary fitted model.
        let caller_bound = crate::grm::GrmConfig::default().tol * (1.0 + expected.abs());
        assert!(error <= caller_bound);
        for persons in [3, 127, 10_001] {
            let actual = crate::gpu_grm_estep::expectation(
                &vec![1; persons * items],
                None,
                persons,
                items,
                1,
                2,
                &[0.0],
                &[0.0],
                &params,
                &dims,
            )
            .unwrap();
            let expected = single.loglik * persons as f64;
            assert!(
                (actual.loglik - expected).abs()
                    <= crate::grm::GrmConfig::default().tol * (1.0 + expected.abs())
            );
        }
    }
}
