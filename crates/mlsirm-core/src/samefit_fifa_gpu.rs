//! Strict GPU f32 coefficient/Jacobian/delta propagation; no CPU fallback.
use crate::samefit_grm::{Coordinate, PackedArchive};
use wgpu::util::DeviceExt;
#[derive(serde::Serialize)]
pub struct Output {
    pub estimates: Vec<f64>,
    pub jacobian: Vec<f64>,
    pub covariance: Vec<f64>,
    pub backend: String,
    pub device: String,
    pub dispatches: usize,
}
fn admit_adapter(kind: wgpu::DeviceType) -> Result<(), String> {
    if matches!(kind, wgpu::DeviceType::Cpu | wgpu::DeviceType::Other) {
        return Err("hardware GPU required; software/unknown adapter prohibited".into());
    }
    Ok(())
}
fn workspace_shape(
    p: usize,
    j: usize,
    z: usize,
    g: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let per_item = z
        .checked_mul(3)
        .and_then(|v| v.checked_add(2))
        .ok_or("item row overflow")?;
    let per_group = j
        .checked_mul(per_item)
        .and_then(|v| v.checked_add(2))
        .ok_or("group row overflow")?;
    let n = g.checked_mul(per_group).ok_or("row overflow")?;
    let np = n.checked_mul(p).ok_or("Jacobian overflow")?;
    let nn = n.checked_mul(n).ok_or("covariance overflow")?;
    let total = np
        .checked_mul(2)
        .and_then(|v| v.checked_add(n))
        .and_then(|v| v.checked_add(nn))
        .ok_or("workspace overflow")?;
    if p == 0
        || j == 0
        || z == 0
        || g < 2
        || total > 16_000_000
        || p.checked_mul(p).is_none_or(|v| v > 16_000_000)
    {
        return Err("GPU workspace outside bounded range".into());
    }
    // Every uniform, storage offset, invocation count and index fits WGSL u32.
    for v in [p, j, z, g, n, np, nn, total] {
        u32::try_from(v).map_err(|_| "WGSL index overflow")?;
    }
    Ok((n, np, nn, total))
}
/// FIFA-only Cholesky admission (Higham, 2009, p. 251, Eq. 1; p. 253).
/// Divide by the largest entry before factoring to remove the covariance's
/// units from the pivot test. Require finite, strictly positive pivots; no
/// absolute eigenvalue cutoff, diagonal loading or pseudoinverse is used.
/// This floating-point test has the near-boundary limits discussed on p. 252.
///
/// Higham, N. J. (2009). Cholesky factorization. *Wiley Interdisciplinary
/// Reviews: Computational Statistics, 1*(2), 251–254.
/// https://doi.org/10.1002/wics.18
fn positive_definite_covariance(matrix: &[f64], p: usize) -> bool {
    if p == 0 || p.checked_mul(p) != Some(matrix.len()) {
        return false;
    }
    let scale = matrix.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 || matrix.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let mut lower = vec![0.0; matrix.len()];
    for i in 0..p {
        for j in 0..=i {
            let mut value = matrix[i * p + j] / scale;
            for k in 0..j {
                value -= lower[i * p + k] * lower[j * p + k];
            }
            if !value.is_finite() {
                return false;
            }
            if i == j {
                if value <= 0.0 {
                    return false;
                }
                lower[i * p + j] = value.sqrt();
            } else {
                lower[i * p + j] = value / lower[j * p + j];
                if !lower[i * p + j].is_finite() {
                    return false;
                }
            }
        }
    }
    true
}
#[cfg(test)]
mod admission_tests {
    #[test]
    fn geometry_overflow_is_error_not_panic() {
        for (p, j, z, g) in [
            (1, 1, usize::MAX, 2),
            (1, usize::MAX, 1, 2),
            (usize::MAX, 1, 1, 2),
            (1, 1, 1, usize::MAX),
        ] {
            let result = std::panic::catch_unwind(|| super::workspace_shape(p, j, z, g));
            assert!(
                result.is_ok(),
                "geometry must return Err without overflow panic"
            );
            assert!(result.unwrap().is_err());
        }
    }
    #[test]
    fn software_and_unknown_adapters_are_rejected() {
        use wgpu::DeviceType::*;
        for kind in [Cpu, Other] {
            assert!(super::admit_adapter(kind).is_err());
        }
        for kind in [IntegratedGpu, DiscreteGpu, VirtualGpu] {
            assert!(super::admit_adapter(kind).is_ok());
        }
    }
}
/// f32 implementation of the coefficient convention documented by
/// [`super::transform`] and first-order propagation `J C J^T` (Joint Committee
/// for Guides in Metrology, 2011, p. 15, §6.2.1.3, Eq. 3).
/// Covariance symmetry uses relative tolerance 1e-7 against the largest entry,
/// without an absolute floor. Accepted roundoff is replaced by the pair mean.
/// Scale-normalized Cholesky checks that symmetric f64 matrix and its exact
/// f32-rounded upload (Higham, 2009, pp. 251–253).
/// Numerical caps/guards are implementation policy, not statistical validity.
///
/// # References
/// Higham, N. J. (2009). Cholesky factorization. *Wiley Interdisciplinary
/// Reviews: Computational Statistics, 1*(2), 251–254.
/// https://doi.org/10.1002/wics.18
///
/// Joint Committee for Guides in Metrology. (2011). *Evaluation of measurement
/// data—Supplement 2 to the “Guide to the expression of uncertainty in
/// measurement”—Extension to any number of output quantities* (JCGM 102:2011).
/// https://doi.org/10.59161/JCGM102-2011
pub fn evaluate(a: &PackedArchive, cov: &[f64]) -> Result<Output, String> {
    let coords = a.coordinates()?;
    let p = coords.len();
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let g = a.n_groups();
    let (n, np, nn, total) = workspace_shape(p, j, z, g)?;
    if total > 16_000_000 || p.checked_mul(p) != Some(cov.len()) {
        return Err("GPU workspace or covariance shape invalid".into());
    }
    let x: Vec<f32> = a
        .packed_bits()
        .iter()
        .map(|&b| f64::from_bits(b) as f32)
        .collect();
    let mut c: Vec<f32> = cov.iter().map(|&v| v as f32).collect();
    if x.iter().chain(&c).any(|v| !v.is_finite()) {
        return Err("nonfinite f32 input".into());
    }
    let scale = cov.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let mut symmetric_cov = cov.to_vec();
    for i in 0..p {
        for k in 0..i {
            let left = cov[i * p + k];
            let right = cov[k * p + i];
            if scale > 0.0 && (left / scale - right / scale).abs() > 1e-7 {
                return Err("asymmetric parameter covariance".into());
            }
            // Canonicalize only admitted relative roundoff. Check and upload
            // the same matrix; casting one shared mean keeps f32 symmetry exact.
            let mean = 0.5 * left + 0.5 * right;
            symmetric_cov[i * p + k] = mean;
            symmetric_cov[k * p + i] = mean;
            c[i * p + k] = mean as f32;
            c[k * p + i] = mean as f32;
        }
    }
    if !positive_definite_covariance(&symmetric_cov, p) {
        return Err("nonpositive parameter covariance".into());
    }
    // Validate the matrix that is actually transferred to the f32 shader.
    // Rounding can destroy positive definiteness or erase a positive variance.
    let rounded_cov: Vec<f64> = c.iter().map(|&value| f64::from(value)).collect();
    if !positive_definite_covariance(&rounded_cov, p) {
        return Err("nonpositive f32 parameter covariance".into());
    }
    for (coord, &value) in coords.iter().zip(&x) {
        if matches!(coord, Coordinate::LogSd(_) | Coordinate::LogGap(_, _, _)) {
            let exponential = value.exp();
            if !exponential.is_finite() || exponential < f32::MIN_POSITIVE {
                return Err("f32 exponential outside normal range".into());
            }
        }
    }
    let mut ix = Vec::new();
    let index = |c: Coordinate| coords.iter().position(|v| *v == c).map(|i| i as u32);
    for group in 0..g {
        for item in 0..j {
            let h = if index(Coordinate::Slope(group, item)).is_some() {
                group
            } else {
                0
            };
            let ai = index(Coordinate::Slope(h, item)).ok_or("missing slope")?;
            if x[ai as usize].abs() < 1e-12 {
                return Err("difficulty undefined at zero slope".into());
            }
            ix.push(ai);
            ix.push(index(Coordinate::FirstIntercept(h, item)).ok_or("missing intercept")?);
            for k in 1..z {
                ix.push(index(Coordinate::LogGap(h, item, k)).ok_or("missing gap")?);
            }
        }
    }
    for group in 0..g {
        ix.push(index(Coordinate::Mean(group)).unwrap_or(u32::MAX));
        ix.push(index(Coordinate::LogSd(group)).unwrap_or(u32::MAX));
    }
    // Preserve ordered thresholds in the precision the shader actually uses.
    for block in ix[..g * j * (z + 1)].chunks_exact(z + 1) {
        let mut intercept = x[block[1] as usize];
        for &gap in &block[2..] {
            let next = intercept - x[gap as usize].exp();
            if !next.is_finite() || next >= intercept {
                return Err("f32 intercept order collapsed".into());
            }
            intercept = next;
        }
    }
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable; CPU fallback prohibited")?;
    admit_adapter(ctx.device_type)?;
    let device = &ctx.device;
    let shape = [p as u32, n as u32, j as u32, z as u32, g as u32, 0, 0, 0];
    let make = |data: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FIFA"),
            contents: data,
            usage,
        })
    };
    let limits = device.limits();
    for len in [x.len(), ix.len(), c.len(), total] {
        let bytes = (len as u64).checked_mul(4).ok_or("buffer byte overflow")?;
        if bytes > limits.max_storage_buffer_binding_size as u64 || bytes > limits.max_buffer_size {
            return Err("GPU buffer limit".into());
        }
    }
    for count in [n + np, np, nn] {
        if count.div_ceil(64) as u64 > limits.max_compute_workgroups_per_dimension as u64 {
            return Err("GPU dispatch limit".into());
        }
    }
    let uniform = make(bytemuck::cast_slice(&shape), wgpu::BufferUsages::UNIFORM);
    let pb = make(bytemuck::cast_slice(&x), wgpu::BufferUsages::STORAGE);
    let ib = make(bytemuck::cast_slice(&ix), wgpu::BufferUsages::STORAGE);
    let cb = make(bytemuck::cast_slice(&c), wgpu::BufferUsages::STORAGE);
    let output = crate::gpu::output_buffer(device, "FIFA output", total);
    let staging = crate::gpu::staging_buffer(device, "FIFA readback", total);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FIFA"),
        source: wgpu::ShaderSource::Wgsl(include_str!("samefit_fifa.wgsl").into()),
    });
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for i in 1..=4 {
        entries.push(crate::gpu::storage_entry(i, i < 4));
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FIFA"),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FIFA"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let buffers = [&uniform, &pb, &ib, &cb, &output];
    let be: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FIFA"),
        layout: &layout,
        entries: &be,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (entry, count) in [
        ("transform", n + np),
        ("intermediate", np),
        ("propagate", nn),
    ] {
        let groups = count.div_ceil(64) as u32;
        if groups > device.limits().max_compute_workgroups_per_dimension {
            return Err("GPU dispatch limit".into());
        }
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    let data = crate::gpu::submit_and_readback(ctx, encoder, &[(&output, &staging, total)])
        .ok_or("GPU readback failure; no fallback")?;
    let v = &data[0];
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite GPU output".into());
    }
    Ok(Output {
        estimates: v[..n].iter().map(|&v| v as f64).collect(),
        jacobian: v[n..n + np].iter().map(|&v| v as f64).collect(),
        covariance: v[n + 2 * np..].iter().map(|&v| v as f64).collect(),
        backend: ctx.backend().into(),
        device: ctx.device_name().into(),
        dispatches: 3,
    })
}
