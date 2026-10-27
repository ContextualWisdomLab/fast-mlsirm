//! Device completion of posterior moment updates. The existing reduced E-step
//! owns posterior reductions; this layer owns division, centering and square root.
//! It does not claim ownership of host probability-table construction.

use wgpu::util::DeviceExt;

/// Complete the existing E-step sufficient statistics on device; no fallback.
/// Each row is [mass, first moment sum, second moment sum]. Fixed means are zero.
pub(crate) fn update(rows: &[[f64; 3]], free_mean: &[bool]) -> Result<Vec<[f64; 2]>, String> {
    update_selected(rows, free_mean, &vec![true; rows.len()])
}

/// Complete mean-only rows without evaluating a variance; fixed SD remains one.
pub(crate) fn update_selected(
    rows: &[[f64; 3]],
    free_mean: &[bool],
    free_sd: &[bool],
) -> Result<Vec<[f64; 2]>, String> {
    if rows.is_empty() || rows.len() != free_mean.len() || rows.len() != free_sd.len() {
        return Err("invalid moment dimensions".into());
    }
    let (limit, _) = crate::gpu_mstep::limits()?;
    crate::gpu_mstep::checked_buffer(rows.len().checked_mul(4).ok_or("moment overflow")?, limit)?;
    let mut data = Vec::<f32>::new();
    for ((r, &free), &variance) in rows.iter().zip(free_mean).zip(free_sd) {
        if r.iter().any(|x| !x.is_finite() || !(*x as f32).is_finite())
            || r[0] <= 0.0
            || (r[0] as f32) <= 0.0
        {
            return Err("invalid posterior mass/moments".into());
        }
        data.extend(r.iter().map(|&v| v as f32));
        data.push((u32::from(free) + 2 * u32::from(variance)) as f32);
    }
    crate::gpu_mstep::checked_buffer(data.len(), limit)?;
    crate::gpu_mstep::checked_buffer(rows.len().checked_mul(3).ok_or("moment overflow")?, limit)?;
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable")?;
    let d = &ctx.device;
    let count = u32::try_from(rows.len()).map_err(|_| "moment size overflow")?;
    let groups = count.div_ceil(64);
    if groups > d.limits().max_compute_workgroups_per_dimension {
        return Err("moment dispatch limit".into());
    }
    let input = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("group moments"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = crate::gpu::output_buffer(d, "group update", rows.len() * 3);
    let staging = crate::gpu::staging_buffer(d, "group update read", rows.len() * 3);
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("group update"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&x) { return; }
    let r = x[i];
    let mean = select(0.0, r.y/r.x, (u32(r.w) & 1u) != 0u);
    var sd = 1.0; var valid = r.x > 0.0 && abs(mean) <= 3.402823466e38;
    if ((u32(r.w) & 2u) != 0u) {
        let variance = r.z/r.x - mean*mean;
        sd = sqrt(max(0.0, variance));
        valid = valid && variance > 0.0 && abs(variance) <= 3.402823466e38;
    }
    y[3u*i] = mean;
    y[3u*i+1u] = sd;
    y[3u*i+2u] = select(0.0, 1.0, valid);
}
"#
            .into(),
        ),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("group update"),
        layout: None,
        module: &shader,
        entry_point: Some("update"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
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
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    let read = crate::gpu::submit_and_readback(ctx, enc, &[(&output, &staging, rows.len() * 3)])
        .ok_or("group update readback failed")?;
    let result: Result<Vec<_>, String> = read[0]
        .chunks_exact(3)
        .map(|r| {
            if r[2] != 1.0 || r.iter().any(|x| !x.is_finite()) {
                Err("non-positive/nonfinite group variance".into())
            } else {
                Ok([f64::from(r[0]), f64::from(r[1])])
            }
        })
        .collect();
    #[cfg(test)]
    if result.is_ok() {
        crate::bifactor_grm::scoring_completion_tests::device_mean_only(
            free_sd.iter().filter(|&&v| !v).count(),
        );
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_moments_refuse_before_hardware() {
        assert!(super::update(&[], &[]).is_err());
        assert!(super::update(&[[1.0, 0.0, 1.0]], &[]).is_err());
        for row in [[0.0, 0.0, 0.0], [f64::NAN, 0.0, 1.0], [1.0, f64::INFINITY, 1.0]] {
            assert_eq!(super::update(&[row], &[true]).unwrap_err(), "invalid posterior mass/moments");
        }
    }
    #[test]
    fn hardware_enrollment_is_explicit() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--list", "--ignored"])
            .output().unwrap();
        assert!(output.status.success());
        let listing = String::from_utf8(output.stdout).unwrap();
        assert!(listing.lines().any(|line|
            line == "gpu_group_update::tests::gpu_group_update_known_moments: test"));
    }
    #[test]
    #[ignore = "required physical GPU cohort: tools/verify_lifecycle_hardware.py with the canonical host lease"]
    fn gpu_group_update_known_moments() {
        assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
        let result = super::update(&[[2.0, 4.0, 10.0], [2.0, 4.0, 10.0]], &[true, false]).unwrap();
        assert_eq!(result[0], [2.0, 1.0]);
        assert_eq!(result[1], [0.0, 5.0_f64.sqrt() as f32 as f64]);
        assert!(super::update(&[[0.0, 0.0, 0.0]], &[true]).is_err());
        assert!(super::update(&[[1.0, 2.0, 1.0]], &[true]).is_err());
    }
}
