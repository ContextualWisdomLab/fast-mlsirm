//! Strict two-word E-step transport. The finite-grid likelihood, counts and
//! moments are evaluated on device using the same represented log tables.
use super::{ReducedEstepInputs, ReducedEstepOutputs};
use wgpu::util::DeviceExt;
pub(super) fn expectation(x: &ReducedEstepInputs) -> Result<ReducedEstepOutputs, String> {
    let (limit, max_groups) = crate::gpu_mstep::limits()?;
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or("strict E size overflow");
    let add = |a: usize, b: usize| a.checked_add(b).ok_or("strict E size overflow");
    let np = x.n_persons;
    let ni = x.n_items;
    let ns = x.n_specific;
    let nc = x.n_cat;
    let qg = x.qg;
    let qs = x.qs;
    let ng = x.n_groups;
    if [np, ni, nc, qg, qs, ng].contains(&0)
        || nc < 2
        || x.y.len() != mul(np, ni)?
        || x.observed.is_some_and(|o| o.len() != np * ni)
        || x.group_id
            .is_some_and(|g| g.len() != np || g.iter().any(|g| *g >= ng))
        || x.item_block.len() != ni
        || x.tables_groups.len() != ng
        || x.tg_groups.len() != ng
        || x.ts_groups.len() != ng
        || x.log_wg.len() != qg
        || x.log_ws.len() != qs
        || x.log_wg.iter().any(|v| v.is_nan() || *v == f64::INFINITY)
        || !x.log_wg.iter().any(|v| v.is_finite())
        || x.log_ws.iter().any(|v| v.is_nan() || *v == f64::INFINITY)
        || !x.log_ws.iter().any(|v| v.is_finite())
    {
        return Err("invalid strict E input geometry/prior".into());
    }
    for g in 0..ng {
        if x.tables_groups[g].len() != ni
            || x.tg_groups[g].len() != qg
            || x.ts_groups[g].len() != ns
        {
            return Err("invalid strict E table/node geometry".into());
        }
        for i in 0..ni {
            let nodes = if x.item_block[i].is_some() {
                mul(qg, qs)?
            } else {
                qg
            };
            if x.item_block[i].is_some_and(|s| s >= ns)
                || x.tables_groups[g][i].len() != mul(nodes, nc)?
                || x.tables_groups[g][i]
                    .iter()
                    .any(|v| v.is_nan() || *v == f64::INFINITY)
            {
                return Err("invalid strict E probability table".into());
            }
        }
        if x.tg_groups[g].iter().any(|v| !v.is_finite())
            || x.ts_groups[g]
                .iter()
                .any(|v| v.len() != qs || v.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid strict E quadrature nodes".into());
        }
    }
    let stride = mul(qg, qs)?;
    let ncount = mul(mul(mul(ng, ni)?, stride)?, nc)?;
    let nmoment = mul(ng, add(3, mul(3, ns)?)?)?;
    let mpg = mul(np, qg)?;
    let block = mul(mul(mpg, ns)?, qs)?;
    let logi = mul(mpg, ns)?;
    let moff = add(1, ncount)?;
    let pgoff = add(moff, nmoment)?;
    let baoff = add(pgoff, mpg)?;
    let joff = add(baoff, block)?;
    let lioff = add(joff, block)?;
    let lloff = add(lioff, logi)?;
    let flagoff = add(lloff, np)?;
    let size = add(flagoff, np)?;
    crate::gpu_mstep::checked_buffer(mul(size, 2)?, limit)?;
    let table_cells = x
        .tables_groups
        .iter()
        .flatten()
        .try_fold(0usize, |n, t| add(n, t.len()))?;
    let input_cells = [
        mul(np, ni)?,
        np,
        ni,
        mul(ng, ni)?,
        table_cells,
        qg,
        qs,
        mul(ng, qg)?,
        mul(mul(ng, ns)?, qs)?,
    ]
    .into_iter()
    .try_fold(24usize, add)?;
    crate::gpu_mstep::checked_buffer(mul(input_cells, 2)?, limit)?;
    if input_cells > 16_777_216 {
        return Err("strict E exact input index range exceeded".into());
    }
    let mut d = vec![0.0f64; 24];
    d[..7].copy_from_slice(&[
        np as f64, ni as f64, ns as f64, nc as f64, qg as f64, qs as f64, ng as f64,
    ]);
    let mut append = |slot: usize, v: Vec<f64>| {
        d[slot] = d.len() as f64;
        d.extend(v);
    };
    append(
        7,
        (0..np * ni)
            .map(|i| {
                if x.observed.is_some_and(|o| !o[i]) {
                    -1.0
                } else {
                    x.y[i] as f64
                }
            })
            .collect(),
    );
    append(
        8,
        (0..np)
            .map(|p| x.group_id.map_or(0, |g| g[p]) as f64)
            .collect(),
    );
    append(
        9,
        x.item_block
            .iter()
            .map(|s| s.map_or(-1.0, |s| s as f64))
            .collect(),
    );
    let mut offsets = Vec::new();
    let mut tables = Vec::new();
    for g in x.tables_groups {
        for t in g {
            offsets.push(tables.len() as f64);
            tables.extend_from_slice(t);
        }
    }
    append(10, offsets);
    append(11, tables);
    append(12, x.log_wg.to_vec());
    append(13, x.log_ws.to_vec());
    append(14, x.tg_groups.iter().flatten().copied().collect());
    append(
        15,
        x.ts_groups.iter().flatten().flatten().copied().collect(),
    );
    d[16] = moff as f64;
    d[18] = pgoff as f64;
    d[19] = baoff as f64;
    d[20] = joff as f64;
    d[21] = lioff as f64;
    d[22] = lloff as f64;
    d[23] = flagoff as f64;
    // Integer offsets stored in f32 high words must be exact; no shape aliasing.
    if d[..24].iter().any(|v| *v > 16_777_216.0)
        || [np, ni, ns, nc, qg, qs, ng, size]
            .iter()
            .any(|v| *v > 16_777_216)
    {
        return Err("strict E exact index range exceeded".into());
    }
    if d.iter().any(|v| v.is_finite() && !(*v as f32).is_finite()) {
        return Err("strict E input range exceeded".into());
    }
    let packed: Vec<f32> = d
        .into_iter()
        .flat_map(|v| {
            let h = v as f32;
            [
                h,
                if v.is_finite() {
                    (v - h as f64) as f32
                } else {
                    0.0
                },
            ]
        })
        .collect();
    crate::gpu_mstep::checked_buffer(packed.len(), limit)?;
    let ctx = crate::gpu::GpuContext::get().ok_or("GPU unavailable")?;
    let dev = &ctx.device;
    let input = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("strict bifactor E input"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = crate::gpu::output_buffer(dev, "strict bifactor E output", size * 2);
    let staging = crate::gpu::staging_buffer(dev, "strict bifactor E read", size * 2);
    let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = dev.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("strict bifactor E"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_bifactor_precision.wgsl").into()),
    });
    let make = |entry: &str| {
        dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = [
        make("posterior"),
        make("counts"),
        make("moments"),
        make("likelihood"),
    ];
    let mut encoder = dev.create_command_encoder(&Default::default());
    for (p, groups) in pipelines.iter().zip([np, ncount, ng * (ns + 1), 1]) {
        if groups == 0 {
            return Err("strict E empty dispatch".into());
        }
        let (dx, dy, dz) = crate::gpu::dispatch_workgroups_nd(groups as u32, max_groups as u32)
            .ok_or("strict E workgroup limit exceeded")?;
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
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
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }
    let raw = crate::gpu::submit_and_readback(ctx, encoder, &[(&output, &staging, size * 2)])
        .ok_or("strict E readback failed")?;
    let validation_error = pollster::block_on(validation.pop());
    let internal_error = pollster::block_on(internal.pop());
    let oom_error = pollster::block_on(oom.pop());
    if let Some(e) = validation_error.or(internal_error).or(oom_error) {
        return Err(format!("strict E device execution rejected: {e}"));
    }
    let v: Vec<f64> = raw[0]
        .chunks_exact(2)
        .map(|r| r[0] as f64 + r[1] as f64)
        .collect();
    if v[..pgoff].iter().any(|v| !v.is_finite()) || v[flagoff..].iter().any(|v| *v != 0.0) {
        return Err(format!(
            "strict E arithmetic range refused; CPU fallback forbidden: invalid={} flags={} ll={}",
            v[..pgoff].iter().filter(|v| !v.is_finite()).count(),
            v[flagoff..].iter().filter(|v| **v != 0.0).count(),
            v[0]
        ));
    }
    let row = 3 + 3 * ns;
    let mut w_acc = vec![0.0; ng];
    let mut s1_g = w_acc.clone();
    let mut s2_g = w_acc.clone();
    let mut s1_spec = vec![0.0; ng * ns];
    let mut s2_spec = s1_spec.clone();
    let mut w_spec = s1_spec.clone();
    for g in 0..ng {
        let m = &v[moff + g * row..];
        w_acc[g] = m[0];
        s1_g[g] = m[1];
        s2_g[g] = m[2];
        for s in 0..ns {
            w_spec[g * ns + s] = m[3 + s];
            s1_spec[g * ns + s] = m[3 + ns + s];
            s2_spec[g * ns + s] = m[3 + 2 * ns + s];
        }
    }
    Ok(ReducedEstepOutputs {
        loglik: v[0],
        counts: v[1..moff].to_vec(),
        counts_stride_nodes: stride,
        w_acc,
        s1_g,
        s2_g,
        s1_spec,
        s2_spec,
        w_spec,
        postg: v[pgoff..baoff].to_vec(),
    })
}
