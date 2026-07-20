//! wgpu f32 kernels for the marginal-EM E-step hot path.
//!
//! The E-step dominates marginal fitting (measured ~110 s/iteration on CPU f64
//! for a 31k-person multilevel fit). This module offloads it with the same
//! race-free slot-ownership reduction strategy as `gpu.rs`:
//!
//! * `lp_pass`   — one thread per (person, context): streams the per-dimension
//!   online log-sum-exp over the trait nodes and writes `logz[(p,s,d,x)]` and
//!   the person log-marginal `lp[(p,s)]`.
//! * `nbar_pass` — one thread per (context, dim, t, x) grid slot: reduces the
//!   posterior over persons (reads `logz`/`lp`, recomputes the cheap
//!   per-person cell value from the sparse positive/missing lists).
//! * `rbar_pass` / `mbar_pass` — one thread per (context, item, t, x): reduces
//!   over the item-major positive (resp. missing) person lists.
//!
//! Kernels run in f32 (WGSL has no f64); accumulation noise is ~1e-4 relative,
//! which perturbs the EM trajectory but not the fixed point materially. The
//! driver in `marginal.rs` therefore uses the GPU only for E-step iterations
//! and always runs the final EAP pass (and the M-step) on the CPU in f64. When
//! no adapter is present, `e_step_gpu` returns `None` and the caller falls
//! back to the CPU E-step — behaviour identical, CI-safe.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::ModelConfig;

const WORKGROUP_SIZE: u32 = 64;
/// Compile-time bound for the per-invocation streaming buffers; validated at
/// dispatch (q_theta <= 41 by table construction).
const MAX_QT: usize = 41;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    n_ctx: u32,
    q_t: u32,
    n_x: u32,
    /// 1 when every person spans every context (multilevel), 0 when each
    /// person has exactly one context (single/multigroup).
    all_ctx: u32,
    _pad: u32,
}

const SHADER: &str = r#"
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    n_ctx: u32,
    q_t: u32,
    n_x: u32,
    all_ctx: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> logp0: array<f32>;
@group(0) @binding(2) var<storage, read> logp1: array<f32>;
@group(0) @binding(3) var<storage, read> c0: array<f32>;
@group(0) @binding(4) var<storage, read> t_logw: array<f32>;
@group(0) @binding(5) var<storage, read> x_logw: array<f32>;
@group(0) @binding(6) var<storage, read> factor_id: array<u32>;
@group(0) @binding(7) var<storage, read> ctx_of_person: array<u32>;
@group(0) @binding(8) var<storage, read> pos_off: array<u32>;
@group(0) @binding(9) var<storage, read> pos_items: array<u32>;
@group(0) @binding(10) var<storage, read> miss_off: array<u32>;
@group(0) @binding(11) var<storage, read> miss_items: array<u32>;
@group(0) @binding(12) var<storage, read_write> logz: array<f32>;
@group(0) @binding(13) var<storage, read_write> lp: array<f32>;
@group(0) @binding(14) var<storage, read> w_outer: array<f32>;
@group(0) @binding(15) var<storage, read_write> out_acc: array<f32>;
@group(0) @binding(16) var<storage, read> item_off: array<u32>;
@group(0) @binding(17) var<storage, read> item_persons: array<u32>;

const MAX_QT: u32 = 41u;

fn cell_l(p: u32, s: u32, d: u32, t: u32, x: u32) -> f32 {
    let cell = U.q_t * U.n_x;
    var v = c0[(s * U.n_dims + d) * cell + t * U.n_x + x];
    for (var j = pos_off[p]; j < pos_off[p + 1u]; j = j + 1u) {
        let i = pos_items[j];
        if (factor_id[i] == d) {
            let idx = (s * U.n_items + i) * cell + t * U.n_x + x;
            v = v + logp1[idx] - logp0[idx];
        }
    }
    for (var j = miss_off[p]; j < miss_off[p + 1u]; j = j + 1u) {
        let i = miss_items[j];
        if (factor_id[i] == d) {
            let idx = (s * U.n_items + i) * cell + t * U.n_x + x;
            v = v - logp0[idx];
        }
    }
    return v;
}

@compute @workgroup_size(64)
fn lp_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = U.n_persons * U.n_ctx;
    if (idx >= total) { return; }
    let p = idx / U.n_ctx;
    let s = idx % U.n_ctx;
    if (U.all_ctx == 0u && ctx_of_person[p] != s) { return; }

    // per-x accumulator for sum_d logz — streamed, then lse over x.
    var mx = -3.4e38;
    var sx = 0.0;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        var sum_d = x_logw[x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            // online log-sum-exp over t
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = t_logw[t] + cell_l(p, s, d, t, x);
                if (v > m) {
                    acc = acc * exp(m - v) + 1.0;
                    m = v;
                } else {
                    acc = acc + exp(v - m);
                }
            }
            let z = m + log(acc);
            logz[((p * U.n_ctx + s) * U.n_dims + d) * U.n_x + x] = z;
            sum_d = sum_d + z;
        }
        if (sum_d > mx) {
            sx = sx * exp(mx - sum_d) + 1.0;
            mx = sum_d;
        } else {
            sx = sx + exp(sum_d - mx);
        }
    }
    lp[p * U.n_ctx + s] = mx + log(sx);
}

@compute @workgroup_size(64)
fn nbar_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cell = U.q_t * U.n_x;
    let total = U.n_ctx * U.n_dims * cell;
    if (idx >= total) { return; }
    let s = idx / (U.n_dims * cell);
    let rem = idx % (U.n_dims * cell);
    let d = rem / cell;
    let t = (rem % cell) / U.n_x;
    let x = (rem % cell) % U.n_x;

    var acc = 0.0;
    for (var p = 0u; p < U.n_persons; p = p + 1u) {
        if (U.all_ctx == 0u && ctx_of_person[p] != s) { continue; }
        let w = w_outer[s * U.n_persons + p];
        if (w < 1e-14) { continue; }
        var sum_d = x_logw[x];
        for (var dd = 0u; dd < U.n_dims; dd = dd + 1u) {
            sum_d = sum_d + logz[((p * U.n_ctx + s) * U.n_dims + dd) * U.n_x + x];
        }
        let px = exp(sum_d - lp[p * U.n_ctx + s]);
        let lz = logz[((p * U.n_ctx + s) * U.n_dims + d) * U.n_x + x];
        let pt = exp(t_logw[t] + cell_l(p, s, d, t, x) - lz);
        acc = acc + w * px * pt;
    }
    out_acc[idx] = acc;
}

// One thread per (ctx, item, t, x); reduces over the item-major person list
// (positives for rbar, missing for mbar — the host binds the matching list).
@compute @workgroup_size(64)
fn item_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cell = U.q_t * U.n_x;
    let total = U.n_ctx * U.n_items * cell;
    if (idx >= total) { return; }
    let s = idx / (U.n_items * cell);
    let rem = idx % (U.n_items * cell);
    let i = rem / cell;
    let t = (rem % cell) / U.n_x;
    let x = (rem % cell) % U.n_x;
    let d = factor_id[i];

    var acc = 0.0;
    for (var j = item_off[i]; j < item_off[i + 1u]; j = j + 1u) {
        let p = item_persons[j];
        if (U.all_ctx == 0u && ctx_of_person[p] != s) { continue; }
        let w = w_outer[s * U.n_persons + p];
        if (w < 1e-14) { continue; }
        var sum_d = x_logw[x];
        for (var dd = 0u; dd < U.n_dims; dd = dd + 1u) {
            sum_d = sum_d + logz[((p * U.n_ctx + s) * U.n_dims + dd) * U.n_x + x];
        }
        let px = exp(sum_d - lp[p * U.n_ctx + s]);
        let lz = logz[((p * U.n_ctx + s) * U.n_dims + d) * U.n_x + x];
        let pt = exp(t_logw[t] + cell_l(p, s, d, t, x) - lz);
        acc = acc + w * px * pt;
    }
    out_acc[idx] = acc;
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_lp: wgpu::ComputePipeline,
    pipeline_nbar: wgpu::ComputePipeline,
    pipeline_item: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pipeline_score: wgpu::ComputePipeline,
    score_layout: wgpu::BindGroupLayout,
}

static CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

fn context() -> Option<&'static GpuContext> {
    CONTEXT
        .get_or_init(|| {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .ok()?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("mlsirm-marginal-gpgpu"),
                    // The adapter's real limits: the 18-binding layout and the
                    // large logz buffer exceed the downlevel defaults.
                    required_limits: adapter.limits(),
                    ..Default::default()
                }))
                .ok()?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mlsirm-marginal-estep"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..18)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else if matches!(binding, 12 | 13 | 15) {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mlsirm-marginal-layout"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mlsirm-marginal-pipeline-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let make = |entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
            let score_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mlsirm-score"),
                source: wgpu::ShaderSource::Wgsl(SCORE_SHADER.into()),
            });
            let score_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..19)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else if binding >= 15 {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let score_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mlsirm-score-layout"),
                entries: &score_entries,
            });
            let score_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mlsirm-score-pl"),
                bind_group_layouts: &[Some(&score_layout)],
                immediate_size: 0,
            });
            let pipeline_score = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("score_pass"),
                layout: Some(&score_pl),
                module: &score_shader,
                entry_point: Some("score_pass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            Some(GpuContext {
                pipeline_lp: make("lp_pass"),
                pipeline_nbar: make("nbar_pass"),
                pipeline_item: make("item_pass"),
                pipeline_score,
                score_layout,
                device,
                queue,
                layout,
            })
        })
        .as_ref()
}

/// Inputs shared by every dispatch of one E-step.
pub(crate) struct GpuEStepInputs<'a> {
    pub logp0: &'a [f64],
    pub logp1: &'a [f64],
    pub c0: &'a [f64],
    pub t_logw: &'a [f64],
    pub x_logw: &'a [f64],
    pub factor_id: &'a [usize],
    /// Person's own context (single: 0, multigroup: group). Ignored when
    /// `all_ctx` (multilevel).
    pub ctx_of_person: &'a [u32],
    pub all_ctx: bool,
    pub n_ctx: usize,
    pub pos_off: &'a [u32],
    pub pos_items: &'a [u32],
    pub miss_off: &'a [u32],
    pub miss_items: &'a [u32],
    /// Item-major positives: CSR over items -> person ids.
    pub item_pos_off: &'a [u32],
    pub item_pos_persons: &'a [u32],
    /// Item-major missing cells.
    pub item_miss_off: &'a [u32],
    pub item_miss_persons: &'a [u32],
}

/// Outputs of the person pass, needed by the caller to build cluster
/// posteriors before the accumulation dispatches.
pub(crate) struct GpuEStepOutputs {
    /// Person log-marginals (kept for future consumers; the adapter derives
    /// its log-likelihood inside `w_outer_fn`).
    #[allow(dead_code)]
    pub lp: Vec<f64>,
    pub nbar: Vec<f64>,
    pub rbar: Vec<f64>,
    pub mbar: Vec<f64>,
}

fn as_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn storage(device: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    // wgpu rejects zero-sized bindings; pad empty inputs to one element.
    let padded: &[u8] = if data.is_empty() { &[0u8; 4] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: padded,
        usage,
    })
}

/// Run one full E-step on the GPU.
///
/// `w_outer_fn` is called after the lp pass with the downloaded `lp` values
/// (f64, shape persons x n_ctx) and must return the outer weights (context-
/// major, shape n_ctx x persons): cluster posteriors for multilevel, all-ones
/// (own context) otherwise. Returns `None` when no GPU adapter is available.
pub(crate) fn e_step_gpu(
    config: &ModelConfig,
    inputs: &GpuEStepInputs<'_>,
    w_outer_fn: &mut dyn FnMut(&[f64]) -> Vec<f64>,
) -> Option<GpuEStepOutputs> {
    let ctx = context()?;
    let (n_persons, n_items, n_dims) = (config.n_persons, config.n_items, config.n_dims);
    let q_t = inputs.t_logw.len();
    let n_x = inputs.x_logw.len();
    let n_ctx = inputs.n_ctx;
    if q_t > MAX_QT {
        return None;
    }
    let cell = q_t * n_x;

    let uniforms = Uniforms {
        n_persons: n_persons as u32,
        n_items: n_items as u32,
        n_dims: n_dims as u32,
        n_ctx: n_ctx as u32,
        q_t: q_t as u32,
        n_x: n_x as u32,
        all_ctx: inputs.all_ctx as u32,
        _pad: 0,
    };
    let device = &ctx.device;
    let queue = &ctx.queue;

    use wgpu::BufferUsages as BU;
    let u_buf = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let logp0 = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.logp0)),
        BU::STORAGE,
    );
    let logp1 = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.logp1)),
        BU::STORAGE,
    );
    let c0 = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.c0)),
        BU::STORAGE,
    );
    let t_logw = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.t_logw)),
        BU::STORAGE,
    );
    let x_logw = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.x_logw)),
        BU::STORAGE,
    );
    let fid: Vec<u32> = inputs.factor_id.iter().map(|&d| d as u32).collect();
    let fid_buf = storage(device, bytemuck::cast_slice(&fid), BU::STORAGE);
    let ctx_person = storage(
        device,
        bytemuck::cast_slice(inputs.ctx_of_person),
        BU::STORAGE,
    );
    let pos_off = storage(device, bytemuck::cast_slice(inputs.pos_off), BU::STORAGE);
    let pos_items = storage(device, bytemuck::cast_slice(inputs.pos_items), BU::STORAGE);
    let miss_off = storage(device, bytemuck::cast_slice(inputs.miss_off), BU::STORAGE);
    let miss_items = storage(device, bytemuck::cast_slice(inputs.miss_items), BU::STORAGE);

    let logz_size = (n_persons * n_ctx * n_dims * n_x * 4) as u64;
    let logz = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("logz"),
        size: logz_size,
        usage: BU::STORAGE,
        mapped_at_creation: false,
    });
    let lp_size = (n_persons * n_ctx * 4) as u64;
    let lp = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lp"),
        size: lp_size,
        usage: BU::STORAGE | BU::COPY_SRC,
        mapped_at_creation: false,
    });
    // Placeholder single-element buffers for bindings unused by a pass. The
    // read-only and read-write slots need distinct buffers — binding one
    // buffer with both usages in a single dispatch is a validation error.
    let dummy_ro = storage(device, bytemuck::cast_slice(&[0.0f32]), BU::STORAGE);
    let dummy_rw = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-rw"),
        size: 4,
        usage: BU::STORAGE,
        mapped_at_creation: false,
    });
    let dummy_u32 = storage(device, bytemuck::cast_slice(&[0u32, 0u32]), BU::STORAGE);

    let bind = |w_outer: &wgpu::Buffer,
                out_acc: &wgpu::Buffer,
                item_off: &wgpu::Buffer,
                item_persons: &wgpu::Buffer| {
        let entries = [
            (0, &u_buf),
            (1, &logp0),
            (2, &logp1),
            (3, &c0),
            (4, &t_logw),
            (5, &x_logw),
            (6, &fid_buf),
            (7, &ctx_person),
            (8, &pos_off),
            (9, &pos_items),
            (10, &miss_off),
            (11, &miss_items),
            (12, &logz),
            (13, &lp),
            (14, w_outer),
            (15, out_acc),
            (16, item_off),
            (17, item_persons),
        ]
        .map(
            |(binding, buffer): (u32, &wgpu::Buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            },
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.layout,
            entries: &entries,
        })
    };

    // --- Pass 1: lp / logz ---
    let bg = bind(&dummy_ro, &dummy_rw, &dummy_u32, &dummy_u32);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipeline_lp);
        pass.set_bind_group(0, &bg, &[]);
        let total = (n_persons * n_ctx) as u32;
        pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    let lp_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lp-read"),
        size: lp_size,
        usage: BU::MAP_READ | BU::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&lp, 0, &lp_read, 0, lp_size);
    queue.submit([encoder.finish()]);
    lp_read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let lp_host: Vec<f64> = {
        let view = lp_read.slice(..).get_mapped_range().ok()?;
        let floats: &[f32] = bytemuck::cast_slice(&view);
        floats.iter().map(|&v| v as f64).collect()
    };
    lp_read.unmap();

    // Cluster posteriors (or all-ones) computed on the host in f64.
    let w_outer_host = w_outer_fn(&lp_host);
    debug_assert_eq!(w_outer_host.len(), n_ctx * n_persons);
    let w_outer = storage(
        device,
        bytemuck::cast_slice(&as_f32(&w_outer_host)),
        BU::STORAGE,
    );

    let run_reduce = |pipeline: &wgpu::ComputePipeline,
                      total: usize,
                      item_off: &wgpu::Buffer,
                      item_persons: &wgpu::Buffer|
     -> Option<Vec<f64>> {
        let out_size = (total * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("acc-out"),
            size: out_size,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = bind(&w_outer, &out, item_off, item_persons);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((total as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("acc-read"),
            size: out_size,
            usage: BU::MAP_READ | BU::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, out_size);
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let view = read.slice(..).get_mapped_range().ok()?;
        let floats: &[f32] = bytemuck::cast_slice(&view);
        let host: Vec<f64> = floats.iter().map(|&v| v as f64).collect();
        drop(view);
        read.unmap();
        Some(host)
    };

    // --- Pass 2: nbar ---
    let nbar = run_reduce(
        &ctx.pipeline_nbar,
        n_ctx * n_dims * cell,
        &dummy_u32,
        &dummy_u32,
    )?;

    // --- Pass 3: rbar (item-major positives) ---
    let ipo = storage(
        device,
        bytemuck::cast_slice(inputs.item_pos_off),
        BU::STORAGE,
    );
    let ipp = storage(
        device,
        bytemuck::cast_slice(inputs.item_pos_persons),
        BU::STORAGE,
    );
    let rbar = run_reduce(&ctx.pipeline_item, n_ctx * n_items * cell, &ipo, &ipp)?;

    // --- Pass 4: mbar (item-major missing) — skipped when nothing is missing.
    let mbar = if inputs.item_miss_persons.is_empty() {
        vec![0.0; n_ctx * n_items * cell]
    } else {
        let imo = storage(
            device,
            bytemuck::cast_slice(inputs.item_miss_off),
            BU::STORAGE,
        );
        let imp = storage(
            device,
            bytemuck::cast_slice(inputs.item_miss_persons),
            BU::STORAGE,
        );
        run_reduce(&ctx.pipeline_item, n_ctx * n_items * cell, &imo, &imp)?
    };

    Some(GpuEStepOutputs {
        lp: lp_host,
        nbar,
        rbar,
        mbar,
    })
}

// ---------------------------------------------------------------------------
// GPU EAP scoring (Bock & Mislevy 1982). One thread per person, race-free:
// each person owns its output slots, so no atomics / slot ownership (unlike
// the E-step). Reuses the same `cell_l` binary-sparsity decomposition. f32,
// so parity with the f64 CPU path is ~1e-4.
// ---------------------------------------------------------------------------

const SCORE_SHADER: &str = r#"
struct SU {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    q_t: u32,
    n_x: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var<storage, read> logp0: array<f32>;
@group(0) @binding(2) var<storage, read> logp1: array<f32>;
@group(0) @binding(3) var<storage, read> c0: array<f32>;
@group(0) @binding(4) var<storage, read> t_logw: array<f32>;
@group(0) @binding(5) var<storage, read> x_logw: array<f32>;
@group(0) @binding(6) var<storage, read> t_nodes: array<f32>;
@group(0) @binding(7) var<storage, read> x_grid: array<f32>;
@group(0) @binding(8) var<storage, read> prior_mean: array<f32>;
@group(0) @binding(9) var<storage, read> prior_sd: array<f32>;
@group(0) @binding(10) var<storage, read> factor_id: array<u32>;
@group(0) @binding(11) var<storage, read> pos_off: array<u32>;
@group(0) @binding(12) var<storage, read> pos_items: array<u32>;
@group(0) @binding(13) var<storage, read> miss_off: array<u32>;
@group(0) @binding(14) var<storage, read> miss_items: array<u32>;
@group(0) @binding(15) var<storage, read_write> theta_eap: array<f32>;
@group(0) @binding(16) var<storage, read_write> theta_sd: array<f32>;
@group(0) @binding(17) var<storage, read_write> xi_eap: array<f32>;
@group(0) @binding(18) var<storage, read_write> loglik: array<f32>;

fn cell_l(p: u32, d: u32, t: u32, x: u32) -> f32 {
    let cell = U.q_t * U.n_x;
    var v = c0[d * cell + t * U.n_x + x];
    for (var j = pos_off[p]; j < pos_off[p + 1u]; j = j + 1u) {
        let i = pos_items[j];
        if (factor_id[i] == d) {
            let idx = i * cell + t * U.n_x + x;
            v = v + logp1[idx] - logp0[idx];
        }
    }
    for (var j = miss_off[p]; j < miss_off[p + 1u]; j = j + 1u) {
        let i = miss_items[j];
        if (factor_id[i] == d) {
            let idx = i * cell + t * U.n_x + x;
            v = v - logp0[idx];
        }
    }
    return v;
}

@compute @workgroup_size(64)
fn score_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= U.n_persons) { return; }

    // pass A: person log-marginal lp
    var mx = -3.4e38;
    var sx = 0.0;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        var sum_d = x_logw[x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = t_logw[t] + cell_l(p, d, t, x);
                if (v > m) { acc = acc * exp(m - v) + 1.0; m = v; } else { acc = acc + exp(v - m); }
            }
            sum_d = sum_d + (m + log(acc));
        }
        if (sum_d > mx) { sx = sx * exp(mx - sum_d) + 1.0; mx = sum_d; } else { sx = sx + exp(sum_d - mx); }
    }
    let lp = mx + log(sx);
    loglik[p] = lp;

    // pass B: posterior moments
    var te: array<f32, 8u>;
    var tm2: array<f32, 8u>;
    var xe: array<f32, 8u>;
    for (var d = 0u; d < U.n_dims; d = d + 1u) { te[d] = 0.0; tm2[d] = 0.0; }
    for (var k = 0u; k < U.latent_dim; k = k + 1u) { xe[k] = 0.0; }
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        var zbuf: array<f32, 8u>;
        var sum_d = x_logw[x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = t_logw[t] + cell_l(p, d, t, x);
                if (v > m) { acc = acc * exp(m - v) + 1.0; m = v; } else { acc = acc + exp(v - m); }
            }
            let z = m + log(acc);
            zbuf[d] = z;
            sum_d = sum_d + z;
        }
        let px = exp(sum_d - lp);
        for (var k = 0u; k < U.latent_dim; k = k + 1u) {
            xe[k] = xe[k] + px * x_grid[x * U.latent_dim + k];
        }
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let theta = prior_mean[d] + prior_sd[d] * t_nodes[t];
                let pt = exp(t_logw[t] + cell_l(p, d, t, x) - zbuf[d]);
                te[d] = te[d] + px * pt * theta;
                tm2[d] = tm2[d] + px * pt * theta * theta;
            }
        }
    }
    for (var d = 0u; d < U.n_dims; d = d + 1u) {
        theta_eap[p * U.n_dims + d] = te[d];
        let vv = tm2[d] - te[d] * te[d];
        theta_sd[p * U.n_dims + d] = sqrt(max(vv, 0.0));
    }
    for (var k = 0u; k < U.latent_dim; k = k + 1u) {
        xi_eap[p * U.latent_dim + k] = xe[k];
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScoreUniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    q_t: u32,
    n_x: u32,
    _p0: u32,
    _p1: u32,
}

/// Flattened inputs for `score_eap_gpu` (built CPU-side, reusing the same
/// tables/grids/response index as the CPU scoring path).
pub(crate) struct GpuScoreInputs<'a> {
    pub n_persons: usize,
    pub n_items: usize,
    pub n_dims: usize,
    pub latent_dim: usize,
    pub q_t: usize,
    pub n_x: usize,
    pub logp0: &'a [f64],
    pub logp1: &'a [f64],
    pub c0: &'a [f64],
    pub t_logw: &'a [f64],
    pub x_logw: &'a [f64],
    pub t_nodes: &'a [f64],
    pub x_grid: &'a [f64],
    pub prior_mean: &'a [f64],
    pub prior_sd: &'a [f64],
    pub factor_id: &'a [usize],
    pub pos_off: &'a [u32],
    pub pos_items: &'a [u32],
    pub miss_off: &'a [u32],
    pub miss_items: &'a [u32],
}

pub(crate) struct GpuScoreOutputs {
    pub theta_eap: Vec<f64>,
    pub theta_sd: Vec<f64>,
    pub xi_eap: Vec<f64>,
    pub loglik: Vec<f64>,
}

/// EAP scoring on the GPU; `None` when no adapter is present or the model
/// exceeds the fixed kernel bounds (n_dims, latent_dim <= 8; q_t <= 41).
pub(crate) fn score_eap_gpu(inp: &GpuScoreInputs<'_>) -> Option<GpuScoreOutputs> {
    let ctx = context()?;
    if inp.n_dims > 8 || inp.latent_dim > 8 || inp.q_t > MAX_QT {
        return None;
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    use wgpu::BufferUsages as BU;

    let uniforms = ScoreUniforms {
        n_persons: inp.n_persons as u32,
        n_items: inp.n_items as u32,
        n_dims: inp.n_dims as u32,
        latent_dim: inp.latent_dim as u32,
        q_t: inp.q_t as u32,
        n_x: inp.n_x as u32,
        _p0: 0,
        _p1: 0,
    };
    let u_buf = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let logp0 = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.logp0)),
        BU::STORAGE,
    );
    let logp1 = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.logp1)),
        BU::STORAGE,
    );
    let c0 = storage(device, bytemuck::cast_slice(&as_f32(inp.c0)), BU::STORAGE);
    let t_logw = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.t_logw)),
        BU::STORAGE,
    );
    let x_logw = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.x_logw)),
        BU::STORAGE,
    );
    let t_nodes = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.t_nodes)),
        BU::STORAGE,
    );
    let x_grid = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.x_grid)),
        BU::STORAGE,
    );
    let prior_mean = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.prior_mean)),
        BU::STORAGE,
    );
    let prior_sd = storage(
        device,
        bytemuck::cast_slice(&as_f32(inp.prior_sd)),
        BU::STORAGE,
    );
    let fid: Vec<u32> = inp.factor_id.iter().map(|&d| d as u32).collect();
    let fid_buf = storage(device, bytemuck::cast_slice(&fid), BU::STORAGE);
    let pos_off = storage(device, bytemuck::cast_slice(inp.pos_off), BU::STORAGE);
    let pos_items = storage(device, bytemuck::cast_slice(inp.pos_items), BU::STORAGE);
    let miss_off = storage(device, bytemuck::cast_slice(inp.miss_off), BU::STORAGE);
    let miss_items = storage(device, bytemuck::cast_slice(inp.miss_items), BU::STORAGE);

    let mk_out = |n: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("score-out"),
            size: (n.max(1) * 4) as u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let theta_eap = mk_out(inp.n_persons * inp.n_dims);
    let theta_sd = mk_out(inp.n_persons * inp.n_dims);
    let xi_eap = mk_out(inp.n_persons * inp.latent_dim);
    let loglik = mk_out(inp.n_persons);

    let entries = [
        (0, &u_buf),
        (1, &logp0),
        (2, &logp1),
        (3, &c0),
        (4, &t_logw),
        (5, &x_logw),
        (6, &t_nodes),
        (7, &x_grid),
        (8, &prior_mean),
        (9, &prior_sd),
        (10, &fid_buf),
        (11, &pos_off),
        (12, &pos_items),
        (13, &miss_off),
        (14, &miss_items),
        (15, &theta_eap),
        (16, &theta_sd),
        (17, &xi_eap),
        (18, &loglik),
    ]
    .map(
        |(binding, buffer): (u32, &wgpu::Buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        },
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &ctx.score_layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipeline_score);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((inp.n_persons as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit([encoder.finish()]);

    let read = |buf: &wgpu::Buffer, n: usize| -> Option<Vec<f64>> {
        let sz = (n.max(1) * 4) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("score-read"),
            size: sz,
            usage: BU::MAP_READ | BU::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &rb, 0, sz);
        queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let view = rb.slice(..).get_mapped_range().ok()?;
        let floats: &[f32] = bytemuck::cast_slice(&view);
        let host: Vec<f64> = floats.iter().take(n).map(|&v| v as f64).collect();
        drop(view);
        rb.unmap();
        Some(host)
    };

    Some(GpuScoreOutputs {
        theta_eap: read(&theta_eap, inp.n_persons * inp.n_dims)?,
        theta_sd: read(&theta_sd, inp.n_persons * inp.n_dims)?,
        xi_eap: read(&xi_eap, inp.n_persons * inp.latent_dim)?,
        loglik: read(&loglik, inp.n_persons)?,
    })
}
