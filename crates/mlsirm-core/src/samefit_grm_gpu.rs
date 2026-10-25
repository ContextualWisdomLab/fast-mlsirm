//! Strict GPU cumulative marginal moments and within-group Xi; no CPU fallback.
//! Maydeu-Olivares (2013), equations 4, 10–11, defines the moment/Jacobian/Xi
//! inputs. This module computes only moments and Xi, not the M2 statistic.
//! Reference: Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item
//! response theory models. Measurement, 11, 71–101. doi:10.1080/15366367.2013.831680.
use crate::samefit_grm::{decode, PackedArchive};
use wgpu::util::DeviceExt;

pub struct GpuMoments {
    pub moments: Vec<f64>,
    pub xi: Vec<f64>,
    pub backend: String,
    pub device_name: String,
    pub dispatches: usize,
}

/// GPU f32 evaluation, group-major item/cut then item-pair/cut-pair ordering.
/// Xi is per observation within each group, not the joint N/N_g-scaled matrix.
/// Quadrature nodes/normalized weights are explicit. Inputs/decoding run on CPU;
/// all marginal integration, cross moments, and covariance subtraction run GPU.
pub fn moments_gpu(
    a: &PackedArchive,
    nodes: &[f64],
    weights: &[f64],
) -> Result<GpuMoments, String> {
    let d = decode(a)?;
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let g = a.n_groups();
    if nodes.is_empty()
        || nodes.len() != weights.len()
        || nodes.iter().any(|v| !v.is_finite())
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        return Err("invalid normalized quadrature".into());
    }
    let s = j
        .checked_mul(z)
        .and_then(|u| {
            j.checked_mul(j - 1)
                .and_then(|p| p.checked_div(2))
                .and_then(|p| p.checked_mul(z))
                .and_then(|p| p.checked_mul(z))
                .and_then(|p| p.checked_add(u))
        })
        .ok_or("moment overflow")?;
    let per = s
        .checked_mul(s)
        .and_then(|n| n.checked_add(s))
        .ok_or("Xi overflow")?;
    let total = g.checked_mul(per).ok_or("group workspace overflow")?;
    if total > 16_000_000 {
        return Err("GPU workspace exceeds 16 million elements".into());
    }
    let stride = j * (z + 1) + 2;
    let mut p = Vec::new();
    for group in 0..g {
        p.extend_from_slice(&d.slope[group * j..(group + 1) * j]);
        p.extend_from_slice(&d.intercepts[group * j * z..(group + 1) * j * z]);
        p.push(d.means[group]);
        p.push(d.sds[group]);
    }
    let f32s = |v: &[f64]| -> Result<Vec<f32>, String> {
        let x: Vec<_> = v.iter().map(|&x| x as f32).collect();
        if x.iter().any(|x| !x.is_finite()) {
            Err("f32 range exceeded".into())
        } else {
            Ok(x)
        }
    };
    let p = f32s(&p)?;
    let n = f32s(nodes)?;
    let w = f32s(weights)?;
    let mut cons = Vec::<u32>::new();
    for i in 0..j {
        for k in 1..=z {
            let mut c = vec![0; j];
            c[i] = k as u32;
            cons.extend(c);
        }
    }
    for i in 0..j {
        for h in i + 1..j {
            for k in 1..=z {
                for l in 1..=z {
                    let mut c = vec![0; j];
                    c[i] = k as u32;
                    c[h] = l as u32;
                    cons.extend(c);
                }
            }
        }
    }
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable; CPU fallback prohibited")?;
    let device = &ctx.device;
    let shape = [g, j, z, nodes.len(), s, stride, total, 0]
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| "GPU u32 overflow".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let make = |label, data: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    };
    let uniform = make(
        "samefit shape",
        bytemuck::cast_slice(&shape),
        wgpu::BufferUsages::UNIFORM,
    );
    let pb = make(
        "samefit params",
        bytemuck::cast_slice(&p),
        wgpu::BufferUsages::STORAGE,
    );
    let nb = make(
        "samefit nodes",
        bytemuck::cast_slice(&n),
        wgpu::BufferUsages::STORAGE,
    );
    let wb = make(
        "samefit weights",
        bytemuck::cast_slice(&w),
        wgpu::BufferUsages::STORAGE,
    );
    let cb = make(
        "samefit constraints",
        bytemuck::cast_slice(&cons),
        wgpu::BufferUsages::STORAGE,
    );
    let output = crate::gpu::output_buffer(device, "samefit output", total);
    let staging = crate::gpu::staging_buffer(device, "samefit readback", total);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("samefit moments Xi"),
        source: wgpu::ShaderSource::Wgsl(include_str!("samefit_grm_moments.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("samefit moments Xi"),
        layout: None,
        module: &shader,
        entry_point: Some("moments_xi"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers = [&uniform, &pb, &nb, &wb, &cb, &output];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("samefit moments Xi"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let workgroups = total.div_ceil(64) as u32;
    if workgroups > device.limits().max_compute_workgroups_per_dimension {
        return Err("dispatch exceeds one-dimensional device limit".into());
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("samefit moments Xi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let result = crate::gpu::submit_and_readback(ctx, encoder, &[(&output, &staging, total)])
        .ok_or("GPU readback failed; CPU fallback prohibited")?;
    if result[0].iter().any(|v| !v.is_finite()) {
        return Err("GPU produced nonfinite moments".into());
    }
    let mut moments = Vec::with_capacity(g * s);
    let mut xi = Vec::with_capacity(g * s * s);
    for row in result[0].chunks_exact(per) {
        moments.extend(row[..s].iter().map(|&v| v as f64));
        xi.extend(row[s..].iter().map(|&v| v as f64));
    }
    Ok(GpuMoments {
        moments,
        xi,
        backend: ctx.backend().to_string(),
        device_name: ctx.device_name().to_string(),
        dispatches: 1,
    })
}
