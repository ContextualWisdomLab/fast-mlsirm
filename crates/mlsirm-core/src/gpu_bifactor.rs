//! GPU-parallel E-step for the polytomous bifactor graded response model.
//!
//! Implements the same E-step arithmetic as the CPU path in
//! [`crate::bifactor_grm`] (posterior node weights, expected category counts,
//! and marginal log-likelihood) in WGSL `f32`, the widest float WebGPU
//! exposes. The CPU reference path is `f64`; parity targets the agreement
//! appropriate for single precision (see the `Precision` section below).
//! When no GPU adapter is available the entry point returns `None` and the
//! caller falls back to the `f64` CPU implementation.
//!
//! # Precision
//!
//! Kernels accumulate in `f32` (machine epsilon ≈ 1.19e-7). Expected counts
//! are sums over persons of posterior weights in [0, 1], so the worst-case
//! absolute error per count entry grows with `n_persons`; the parity tests
//! assert fit-level agreement derived from that bound (see
//! `tests/test_bifactor_gpu.py`).
//!
//! # References
//!
//! - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//!   analysis. *Psychometrika, 57*(3), 423–436.
//!   https://doi.org/10.1007/BF02295430
//! - Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//!   item bifactor analysis. *Psychological Methods, 16*(3), 221–248.
//!   https://doi.org/10.1037/a0023350
//! - Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van
//!   der Linden & R. K. Hambleton (Eds.), *Handbook of modern item response
//!   theory*. Springer.

#[cfg(all(feature = "gpu", not(coverage)))]
use crate::gpu::GpuContext;
#[cfg(all(feature = "gpu", not(coverage)))]
use wgpu::util::DeviceExt;

// Only constructed by the wgpu path; kept for the CPU-only build so the
// E-step call sites stay cfg-independent.
#[cfg_attr(any(not(feature = "gpu"), coverage), allow(dead_code))]
pub(crate) struct BifactorEstepInputs<'a> {
    pub y: &'a [usize],
    pub observed: Option<&'a [bool]>,
    pub group_ids: Option<&'a [usize]>,
    pub n_persons: usize,
    pub n_items: usize,
    pub n_cat: usize,
    pub qn: usize,
    pub effective_groups: usize,
    pub group_lp: &'a [Vec<Vec<f64>>], // Outer len: effective_groups, Middle len: n_items, Inner len: qn * n_cat
}

pub(crate) struct BifactorEstepOutputs {
    pub counts: Vec<f64>, // n_items * qn * n_cat
    pub group_post_sums: Vec<f64>, // effective_groups * qn
    pub total_ll: f64,
}

#[cfg(all(feature = "gpu", not(coverage)))]
const SHADER: &str = "
struct Dimensions {
    n_persons: u32,
    n_items: u32,
    qn: u32,
    n_cat: u32,
    effective_groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> y: array<u32>;
@group(0) @binding(2) var<storage, read> observed: array<u32>;
@group(0) @binding(3) var<storage, read> group_ids: array<u32>;
@group(0) @binding(4) var<storage, read> group_lp: array<f32>;

@group(0) @binding(5) var<storage, read_write> log_node_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> ll_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> group_post_sums: array<f32>;
@group(0) @binding(8) var<storage, read_write> counts: array<f32>;

@compute @workgroup_size(64, 4)
fn compute_log_node(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let nd = gid.y;
    if (p >= dims.n_persons || nd >= dims.qn) { return; }

    let g = group_ids[p];
    var acc = 0.0;

    for (var i = 0u; i < dims.n_items; i = i + 1u) {
        if (observed[p * dims.n_items + i] != 0u) {
            let yc = y[p * dims.n_items + i];
            let lp_idx = g * (dims.n_items * dims.qn * dims.n_cat)
                       + i * (dims.qn * dims.n_cat)
                       + nd * dims.n_cat
                       + yc;
            acc = acc + group_lp[lp_idx];
        }
    }

    log_node_buf[p * dims.qn + nd] = acc;
}

@compute @workgroup_size(64)
fn compute_post_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= dims.n_persons) { return; }

    var mx = -1e38;
    for (var nd = 0u; nd < dims.qn; nd = nd + 1u) {
        let v = log_node_buf[p * dims.qn + nd];
        if (v > mx) { mx = v; }
    }

    var denom = 0.0;
    for (var nd = 0u; nd < dims.qn; nd = nd + 1u) {
        let v = log_node_buf[p * dims.qn + nd];
        denom = denom + exp(v - mx);
    }

    let log_qn = log(f32(dims.qn));
    ll_buf[p] = mx + log(denom) - log_qn;

    for (var nd = 0u; nd < dims.qn; nd = nd + 1u) {
        let v = log_node_buf[p * dims.qn + nd];
        let post_q = exp(v - mx) / denom;
        log_node_buf[p * dims.qn + nd] = post_q;
    }
}

@compute @workgroup_size(64)
fn reduce_group_post(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.effective_groups * dims.qn;
    if (idx >= total) { return; }

    let g = idx / dims.qn;
    let nd = idx % dims.qn;

    var sum = 0.0;
    for (var p = 0u; p < dims.n_persons; p = p + 1u) {
        if (group_ids[p] == g) {
            sum = sum + log_node_buf[p * dims.qn + nd];
        }
    }
    group_post_sums[idx] = sum;
}

@compute @workgroup_size(64)
fn reduce_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.n_items * dims.qn * dims.n_cat;
    if (idx >= total) { return; }

    let i = idx / (dims.qn * dims.n_cat);
    let rem = idx % (dims.qn * dims.n_cat);
    let nd = rem / dims.n_cat;
    let k = rem % dims.n_cat;

    var sum = 0.0;
    for (var p = 0u; p < dims.n_persons; p = p + 1u) {
        if (observed[p * dims.n_items + i] != 0u && y[p * dims.n_items + i] == k) {
            sum = sum + log_node_buf[p * dims.qn + nd];
        }
    }
    counts[idx] = sum;
}
";

/// GPU E-step for the polytomous bifactor GRM.
///
/// Returns `None` when no compatible GPU adapter can be initialized (or when
/// the `gpu` feature is disabled), signalling the caller to fall back to the
/// CPU implementation.
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn e_step_bifactor_gpu(inputs: &BifactorEstepInputs) -> Option<BifactorEstepOutputs> {
    use crate::gpu::{dispatch_count, output_buffer, staging_buffer, storage_entry, submit_and_readback};

    let ctx = GpuContext::get()?;
    let device = &ctx.device;

    let n = inputs.n_persons * inputs.n_items;
    let mut y_u32 = vec![0u32; n];
    for (i, &v) in inputs.y.iter().enumerate() {
        y_u32[i] = v as u32;
    }
    let mut obs_u32 = vec![1u32; n];
    if let Some(obs) = inputs.observed {
        for (i, &v) in obs.iter().enumerate() {
            obs_u32[i] = u32::from(v);
        }
    }
    let mut gid_u32 = vec![0u32; inputs.n_persons];
    if let Some(gids) = inputs.group_ids {
        for (i, &v) in gids.iter().enumerate() {
            gid_u32[i] = v as u32;
        }
    }

    let mut group_lp_f32 = Vec::with_capacity(
        inputs.effective_groups * inputs.n_items * inputs.qn * inputs.n_cat,
    );
    for g in 0..inputs.effective_groups {
        for i in 0..inputs.n_items {
            for v in &inputs.group_lp[g][i] {
                group_lp_f32.push(*v as f32);
            }
        }
    }

    let dims = [
        inputs.n_persons as u32,
        inputs.n_items as u32,
        inputs.qn as u32,
        inputs.n_cat as u32,
        inputs.effective_groups as u32,
    ];

    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims_buf"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y_buf"),
        contents: bytemuck::cast_slice(&y_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let obs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("obs_buf"),
        contents: bytemuck::cast_slice(&obs_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gid_buf"),
        contents: bytemuck::cast_slice(&gid_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let group_lp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("group_lp_buf"),
        contents: bytemuck::cast_slice(&group_lp_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let log_node_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_node_buf"),
        size: (inputs.n_persons * inputs.qn * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let ll_buf = output_buffer(device, "ll_buf", inputs.n_persons);
    let group_post_sums_buf = output_buffer(
        device,
        "group_post_sums_buf",
        inputs.effective_groups * inputs.qn,
    );
    let counts_buf = output_buffer(
        device,
        "counts_buf",
        inputs.n_items * inputs.qn * inputs.n_cat,
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bifactor_grm"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: obs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: group_lp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: log_node_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: ll_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: group_post_sums_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: counts_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pl_log_node = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("log_node_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("compute_log_node"),
        cache: None,
        compilation_options: Default::default(),
    });
    let pl_post_q = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("post_q_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("compute_post_q"),
        cache: None,
        compilation_options: Default::default(),
    });
    let pl_reduce_group = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reduce_group_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("reduce_group_post"),
        cache: None,
        compilation_options: Default::default(),
    });
    let pl_reduce_counts = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reduce_counts_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("reduce_counts"),
        cache: None,
        compilation_options: Default::default(),
    });

    // One compute pass per kernel so that storage writes from an earlier
    // kernel are visible to the next within this submission.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for (pipeline, gx, gy) in [
        (&pl_log_node, inputs.n_persons.div_ceil(64) as u32, inputs.qn.div_ceil(4) as u32),
        (&pl_post_q, dispatch_count(inputs.n_persons), 1),
        (
            &pl_reduce_group,
            dispatch_count(inputs.effective_groups * inputs.qn),
            1,
        ),
        (
            &pl_reduce_counts,
            dispatch_count(inputs.n_items * inputs.qn * inputs.n_cat),
            1,
        ),
    ] {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(gx.max(1), gy.max(1), 1);
    }

    let ll_staging = staging_buffer(device, "ll_read", inputs.n_persons);
    let group_staging = staging_buffer(
        device,
        "group_post_sums_read",
        inputs.effective_groups * inputs.qn,
    );
    let counts_staging = staging_buffer(
        device,
        "counts_read",
        inputs.n_items * inputs.qn * inputs.n_cat,
    );
    let read = submit_and_readback(
        ctx,
        encoder,
        &[
            (&ll_buf, &ll_staging, inputs.n_persons),
            (
                &group_post_sums_buf,
                &group_staging,
                inputs.effective_groups * inputs.qn,
            ),
            (
                &counts_buf,
                &counts_staging,
                inputs.n_items * inputs.qn * inputs.n_cat,
            ),
        ],
    )?;
    let mut iter = read.into_iter();
    let ll_vec = iter.next()?;
    let group_vec = iter.next()?;
    let counts_vec = iter.next()?;

    let mut total_ll = 0.0;
    for &v in &ll_vec {
        total_ll += f64::from(v);
    }

    Some(BifactorEstepOutputs {
        counts: counts_vec.into_iter().map(f64::from).collect(),
        group_post_sums: group_vec.into_iter().map(f64::from).collect(),
        total_ll,
    })
}

/// CPU-fallback stub used when the `gpu` feature is disabled or under
/// coverage: always returns `None` so the caller runs the CPU E-step.
#[cfg(any(not(feature = "gpu"), coverage))]
#[allow(dead_code)]
pub(crate) fn e_step_bifactor_gpu(_inputs: &BifactorEstepInputs) -> Option<BifactorEstepOutputs> {
    None
}
