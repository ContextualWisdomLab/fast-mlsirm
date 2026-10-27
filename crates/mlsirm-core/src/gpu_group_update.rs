//! Device completion of posterior moment updates. The existing reduced E-step
//! owns posterior reductions; this layer owns division, centering and square root.
//! It does not claim ownership of host probability-table construction.

use wgpu::util::DeviceExt;

/// Complete the existing E-step sufficient statistics on device; no fallback.
/// Each row is [mass, first moment sum, second moment sum]. Fixed means are zero.
pub(crate) fn update(rows: &[[f64; 3]], free_mean: &[bool]) -> Result<Vec<[f64; 2]>, String> {
    if rows.is_empty() || rows.len() != free_mean.len() {
        return Err("invalid moment dimensions".into());
    }
    let mut data = Vec::<f32>::new();
    for (r, &free) in rows.iter().zip(free_mean) {
        if r.iter().any(|x| !x.is_finite() || !(*x as f32).is_finite()) || r[0] <= 0.0 {
            return Err("invalid posterior mass/moments".into());
        }
        data.extend(r.iter().map(|&v| v as f32));
        data.push(if free { 1.0 } else { 0.0 });
    }
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable")?;
    let d = &ctx.device;
    let count = u32::try_from(rows.len()).map_err(|_| "moment size overflow")?;
    let groups = count.div_ceil(64);
    if groups > d.limits().max_compute_workgroups_per_dimension { return Err("moment dispatch limit".into()); }
    let input = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("group moments"), contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE,
    });
    let output = crate::gpu::output_buffer(d, "group update", rows.len()*3);
    let staging = crate::gpu::staging_buffer(d, "group update read", rows.len()*3);
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("group update"), source: wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&x) { return; }
    let r = x[i];
    let mean = select(0.0, r.y/r.x, r.w == 1.0);
    let variance = r.z/r.x - mean*mean;
    y[3u*i] = mean;
    y[3u*i+1u] = sqrt(max(0.0, variance));
    y[3u*i+2u] = select(0.0, 1.0, variance > 0.0 && r.x > 0.0);
}
"#.into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("group update"), layout: None, module: &shader, entry_point: Some("update"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(groups,1,1);
    }
    let read = crate::gpu::submit_and_readback(ctx, enc, &[(&output,&staging,rows.len()*3)]).ok_or("group update readback failed")?;
    read[0].chunks_exact(3).map(|r| {
        if r[2] != 1.0 || r.iter().any(|x| !x.is_finite()) { Err("non-positive/nonfinite group variance".into()) }
        else { Ok([f64::from(r[0]), f64::from(r[1])]) }
    }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu_group_update_known_moments() {
        assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
        let result = super::update(&[[2.0, 4.0, 10.0], [2.0, 4.0, 10.0]], &[true, false]).unwrap();
        assert_eq!(result[0], [2.0, 1.0]);
        assert_eq!(result[1], [0.0, 5.0_f64.sqrt() as f32 as f64]);
        assert!(super::update(&[[0.0, 0.0, 0.0]], &[true]).is_err());
        assert!(super::update(&[[1.0, 2.0, 1.0]], &[true]).is_err());
    }
}
