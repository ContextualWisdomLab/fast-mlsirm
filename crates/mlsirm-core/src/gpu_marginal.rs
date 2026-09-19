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
const ESTEP_STORAGE_BINDINGS_PER_STAGE: u32 = 8;
const SCORE_STORAGE_BINDINGS_PER_STAGE: u32 = 3;
const REQUIRED_STORAGE_BINDINGS_PER_STAGE: u32 = ESTEP_STORAGE_BINDINGS_PER_STAGE;
const REQUIRED_UNIFORM_BINDINGS_PER_STAGE: u32 = 1;
const REQUIRED_COMBINED_BINDINGS_PER_STAGE: u32 =
    REQUIRED_STORAGE_BINDINGS_PER_STAGE + REQUIRED_UNIFORM_BINDINGS_PER_STAGE;
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
    logp1_off: u32,
    c0_off: u32,
    t_logw_off: u32,
    x_logw_off: u32,
    ctx_of_person_off: u32,
    pos_off_off: u32,
    pos_items_off: u32,
    miss_off_off: u32,
    miss_items_off: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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
    logp1_off: u32,
    c0_off: u32,
    t_logw_off: u32,
    x_logw_off: u32,
    ctx_of_person_off: u32,
    pos_off_off: u32,
    pos_items_off: u32,
    miss_off_off: u32,
    miss_items_off: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> f32_ro: array<f32>;
@group(0) @binding(2) var<storage, read> u32_ro: array<u32>;
@group(0) @binding(3) var<storage, read_write> logz: array<f32>;
@group(0) @binding(4) var<storage, read_write> lp: array<f32>;
@group(0) @binding(5) var<storage, read> w_outer: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_acc: array<f32>;
@group(0) @binding(7) var<storage, read> item_off: array<u32>;
@group(0) @binding(8) var<storage, read> item_persons: array<u32>;

const MAX_QT: u32 = 41u;

fn cell_l(p: u32, s: u32, d: u32, t: u32, x: u32) -> f32 {
    let cell = U.q_t * U.n_x;
    var v = f32_ro[U.c0_off + (s * U.n_dims + d) * cell + t * U.n_x + x];
    for (var j = u32_ro[U.pos_off_off + p]; j < u32_ro[U.pos_off_off + p + 1u]; j = j + 1u) {
        let i = u32_ro[U.pos_items_off + j];
        if (u32_ro[i] == d) {
            let idx = (s * U.n_items + i) * cell + t * U.n_x + x;
            v = v + f32_ro[U.logp1_off + idx] - f32_ro[idx];
        }
    }
    for (var j = u32_ro[U.miss_off_off + p]; j < u32_ro[U.miss_off_off + p + 1u]; j = j + 1u) {
        let i = u32_ro[U.miss_items_off + j];
        if (u32_ro[i] == d) {
            let idx = (s * U.n_items + i) * cell + t * U.n_x + x;
            v = v - f32_ro[idx];
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
    if (U.all_ctx == 0u && u32_ro[U.ctx_of_person_off + p] != s) { return; }

    // per-x accumulator for sum_d logz — streamed, then lse over x.
    var mx = -3.4e38;
    var sx = 0.0;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        var sum_d = f32_ro[U.x_logw_off + x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            // online log-sum-exp over t
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = f32_ro[U.t_logw_off + t] + cell_l(p, s, d, t, x);
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
        if (U.all_ctx == 0u && u32_ro[U.ctx_of_person_off + p] != s) { continue; }
        let w = w_outer[s * U.n_persons + p];
        if (w < 1e-14) { continue; }
        var sum_d = f32_ro[U.x_logw_off + x];
        for (var dd = 0u; dd < U.n_dims; dd = dd + 1u) {
            sum_d = sum_d + logz[((p * U.n_ctx + s) * U.n_dims + dd) * U.n_x + x];
        }
        let px = exp(sum_d - lp[p * U.n_ctx + s]);
        let lz = logz[((p * U.n_ctx + s) * U.n_dims + d) * U.n_x + x];
        let pt = exp(f32_ro[U.t_logw_off + t] + cell_l(p, s, d, t, x) - lz);
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
    let d = u32_ro[i];

    var acc = 0.0;
    for (var j = item_off[i]; j < item_off[i + 1u]; j = j + 1u) {
        let p = item_persons[j];
        if (U.all_ctx == 0u && u32_ro[U.ctx_of_person_off + p] != s) { continue; }
        let w = w_outer[s * U.n_persons + p];
        if (w < 1e-14) { continue; }
        var sum_d = f32_ro[U.x_logw_off + x];
        for (var dd = 0u; dd < U.n_dims; dd = dd + 1u) {
            sum_d = sum_d + logz[((p * U.n_ctx + s) * U.n_dims + dd) * U.n_x + x];
        }
        let px = exp(sum_d - lp[p * U.n_ctx + s]);
        let lz = logz[((p * U.n_ctx + s) * U.n_dims + d) * U.n_x + x];
        let pt = exp(f32_ro[U.t_logw_off + t] + cell_l(p, s, d, t, x) - lz);
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
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
}

static CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

fn context() -> Option<&'static GpuContext> {
    CONTEXT
        .get_or_init(|| {
            let instance = crate::gpu_init::new_instance();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .ok()?;
            let adapter_limits = adapter.limits();
            // The packed marginal layout deliberately keeps every compute stage
            // below the controlled software-Vulkan storage and combined binding
            // budgets. Binding-size limits are checked per workload before
            // buffers are created; a workload that exceeds them falls back to
            // the f64 CPU reference instead of panicking during validation.
            if adapter_limits.max_storage_buffers_per_shader_stage
                < REQUIRED_STORAGE_BINDINGS_PER_STAGE
                || adapter_limits.max_uniform_buffers_per_shader_stage
                    < REQUIRED_UNIFORM_BINDINGS_PER_STAGE
                || adapter_limits.max_buffers_and_acceleration_structures_per_shader_stage
                    < REQUIRED_COMBINED_BINDINGS_PER_STAGE
            {
                return None;
            }
            let max_storage_buffer_binding_size =
                u64::from(adapter_limits.max_storage_buffer_binding_size);
            let max_buffer_size = adapter_limits.max_buffer_size;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("mlsirm-marginal-gpgpu"),
                    required_limits: adapter_limits,
                    ..Default::default()
                }))
                .ok()?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mlsirm-marginal-estep"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else if matches!(binding, 3 | 4 | 6) {
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
            debug_assert_eq!(SCORE_STORAGE_BINDINGS_PER_STAGE, 3);
            let score_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else if binding == 3 {
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
                max_storage_buffer_binding_size,
                max_buffer_size,
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

fn binding_bytes<T>(len: usize) -> Option<u64> {
    let len = u64::try_from(len).ok()?;
    len.checked_mul(u64::try_from(std::mem::size_of::<T>()).ok()?)
}

fn storage_binding_fits(ctx: &GpuContext, bytes: u64) -> bool {
    let padded = bytes.max(4);
    padded <= ctx.max_storage_buffer_binding_size && padded <= ctx.max_buffer_size
}

fn append_f64_as_f32(arena: &mut Vec<f32>, values: &[f64]) -> Option<u32> {
    let offset = u32::try_from(arena.len()).ok()?;
    arena.extend(values.iter().map(|&value| value as f32));
    Some(offset)
}

fn append_u32(arena: &mut Vec<u32>, values: &[u32]) -> Option<u32> {
    let offset = u32::try_from(arena.len()).ok()?;
    arena.extend_from_slice(values);
    Some(offset)
}

fn append_usize_as_u32(arena: &mut Vec<u32>, values: &[usize]) -> Option<u32> {
    let offset = u32::try_from(arena.len()).ok()?;
    for &value in values {
        arena.push(u32::try_from(value).ok()?);
    }
    Some(offset)
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

    let mut f32_ro_host = Vec::new();
    let logp0_off = append_f64_as_f32(&mut f32_ro_host, inputs.logp0)?;
    debug_assert_eq!(logp0_off, 0);
    let logp1_off = append_f64_as_f32(&mut f32_ro_host, inputs.logp1)?;
    let c0_off = append_f64_as_f32(&mut f32_ro_host, inputs.c0)?;
    let t_logw_off = append_f64_as_f32(&mut f32_ro_host, inputs.t_logw)?;
    let x_logw_off = append_f64_as_f32(&mut f32_ro_host, inputs.x_logw)?;

    let mut u32_ro_host = Vec::new();
    let factor_id_off = append_usize_as_u32(&mut u32_ro_host, inputs.factor_id)?;
    debug_assert_eq!(factor_id_off, 0);
    let ctx_of_person_off = append_u32(&mut u32_ro_host, inputs.ctx_of_person)?;
    let pos_off_off = append_u32(&mut u32_ro_host, inputs.pos_off)?;
    let pos_items_off = append_u32(&mut u32_ro_host, inputs.pos_items)?;
    let miss_off_off = append_u32(&mut u32_ro_host, inputs.miss_off)?;
    let miss_items_off = append_u32(&mut u32_ro_host, inputs.miss_items)?;

    let uniforms = Uniforms {
        n_persons: u32::try_from(n_persons).ok()?,
        n_items: u32::try_from(n_items).ok()?,
        n_dims: u32::try_from(n_dims).ok()?,
        n_ctx: u32::try_from(n_ctx).ok()?,
        q_t: u32::try_from(q_t).ok()?,
        n_x: u32::try_from(n_x).ok()?,
        all_ctx: inputs.all_ctx as u32,
        _pad: 0,
        logp1_off,
        c0_off,
        t_logw_off,
        x_logw_off,
        ctx_of_person_off,
        pos_off_off,
        pos_items_off,
        miss_off_off,
        miss_items_off,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let device = &ctx.device;
    let queue = &ctx.queue;

    let f32_ro_bytes = binding_bytes::<f32>(f32_ro_host.len())?;
    let u32_ro_bytes = binding_bytes::<u32>(u32_ro_host.len())?;
    let logz_size = binding_bytes::<f32>(n_persons.checked_mul(n_ctx)?.checked_mul(n_dims)?.checked_mul(n_x)?)?;
    let lp_size = binding_bytes::<f32>(n_persons.checked_mul(n_ctx)?)?;
    if !storage_binding_fits(ctx, f32_ro_bytes)
        || !storage_binding_fits(ctx, u32_ro_bytes)
        || !storage_binding_fits(ctx, logz_size)
        || !storage_binding_fits(ctx, lp_size)
    {
        return None;
    }

    use wgpu::BufferUsages as BU;
    let u_buf = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let f32_ro = storage(device, bytemuck::cast_slice(&f32_ro_host), BU::STORAGE);
    let u32_ro = storage(device, bytemuck::cast_slice(&u32_ro_host), BU::STORAGE);
    let logz = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("logz"),
        size: logz_size,
        usage: BU::STORAGE,
        mapped_at_creation: false,
    });
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
            (1, &f32_ro),
            (2, &u32_ro),
            (3, &logz),
            (4, &lp),
            (5, w_outer),
            (6, out_acc),
            (7, item_off),
            (8, item_persons),
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
        let total = u32::try_from(n_persons.checked_mul(n_ctx)?).ok()?;
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
    let w_outer_bytes = binding_bytes::<f32>(w_outer_host.len())?;
    if !storage_binding_fits(ctx, w_outer_bytes) {
        return None;
    }
    let w_outer_f32 = as_f32(&w_outer_host);
    let w_outer = storage(device, bytemuck::cast_slice(&w_outer_f32), BU::STORAGE);

    let run_reduce = |pipeline: &wgpu::ComputePipeline,
                      total: usize,
                      item_off_host: &[u32],
                      item_persons_host: &[u32]|
     -> Option<Vec<f64>> {
        let out_size = binding_bytes::<f32>(total)?;
        let item_off_size = binding_bytes::<u32>(item_off_host.len())?;
        let item_persons_size = binding_bytes::<u32>(item_persons_host.len())?;
        if !storage_binding_fits(ctx, out_size)
            || !storage_binding_fits(ctx, item_off_size)
            || !storage_binding_fits(ctx, item_persons_size)
        {
            return None;
        }
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("acc-out"),
            size: out_size.max(4),
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });
        let item_off = storage(device, bytemuck::cast_slice(item_off_host), BU::STORAGE);
        let item_persons = storage(device, bytemuck::cast_slice(item_persons_host), BU::STORAGE);
        let bg = bind(&w_outer, &out, &item_off, &item_persons);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_u32 = u32::try_from(total).ok()?;
            pass.dispatch_workgroups(total_u32.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("acc-read"),
            size: out_size.max(4),
            usage: BU::MAP_READ | BU::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, out_size.max(4));
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let view = read.slice(..).get_mapped_range().ok()?;
        let floats: &[f32] = bytemuck::cast_slice(&view);
        let host: Vec<f64> = floats.iter().take(total).map(|&v| v as f64).collect();
        drop(view);
        read.unmap();
        Some(host)
    };

    // --- Pass 2: nbar ---
    let nbar = run_reduce(
        &ctx.pipeline_nbar,
        n_ctx.checked_mul(n_dims)?.checked_mul(cell)?,
        &[0u32, 0u32],
        &[0u32, 0u32],
    )?;

    // --- Pass 3: rbar (item-major positives) ---
    let rbar = run_reduce(
        &ctx.pipeline_item,
        n_ctx.checked_mul(n_items)?.checked_mul(cell)?,
        inputs.item_pos_off,
        inputs.item_pos_persons,
    )?;

    // --- Pass 4: mbar (item-major missing) — skipped when nothing is missing.
    let mbar = if inputs.item_miss_persons.is_empty() {
        vec![0.0; n_ctx.checked_mul(n_items)?.checked_mul(cell)?]
    } else {
        run_reduce(
            &ctx.pipeline_item,
            n_ctx.checked_mul(n_items)?.checked_mul(cell)?,
            inputs.item_miss_off,
            inputs.item_miss_persons,
        )?
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
    logp1_off: u32,
    c0_off: u32,
    t_logw_off: u32,
    x_logw_off: u32,
    t_nodes_off: u32,
    x_grid_off: u32,
    prior_mean_off: u32,
    prior_sd_off: u32,
    pos_off_off: u32,
    pos_items_off: u32,
    miss_off_off: u32,
    miss_items_off: u32,
    theta_sd_off: u32,
    xi_eap_off: u32,
    loglik_off: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var<storage, read> f32_ro: array<f32>;
@group(0) @binding(2) var<storage, read> u32_ro: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;

fn cell_l(p: u32, d: u32, t: u32, x: u32) -> f32 {
    let cell = U.q_t * U.n_x;
    var v = f32_ro[U.c0_off + d * cell + t * U.n_x + x];
    for (var j = u32_ro[U.pos_off_off + p]; j < u32_ro[U.pos_off_off + p + 1u]; j = j + 1u) {
        let i = u32_ro[U.pos_items_off + j];
        if (u32_ro[i] == d) {
            let idx = i * cell + t * U.n_x + x;
            v = v + f32_ro[U.logp1_off + idx] - f32_ro[idx];
        }
    }
    for (var j = u32_ro[U.miss_off_off + p]; j < u32_ro[U.miss_off_off + p + 1u]; j = j + 1u) {
        let i = u32_ro[U.miss_items_off + j];
        if (u32_ro[i] == d) {
            let idx = i * cell + t * U.n_x + x;
            v = v - f32_ro[idx];
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
        var sum_d = f32_ro[U.x_logw_off + x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = f32_ro[U.t_logw_off + t] + cell_l(p, d, t, x);
                if (v > m) { acc = acc * exp(m - v) + 1.0; m = v; } else { acc = acc + exp(v - m); }
            }
            sum_d = sum_d + (m + log(acc));
        }
        if (sum_d > mx) { sx = sx * exp(mx - sum_d) + 1.0; mx = sum_d; } else { sx = sx + exp(sum_d - mx); }
    }
    let lp = mx + log(sx);
    outputs[U.loglik_off + p] = lp;

    // pass B: posterior moments
    var te: array<f32, 8u>;
    var tm2: array<f32, 8u>;
    var xe: array<f32, 8u>;
    for (var d = 0u; d < U.n_dims; d = d + 1u) { te[d] = 0.0; tm2[d] = 0.0; }
    for (var k = 0u; k < U.latent_dim; k = k + 1u) { xe[k] = 0.0; }
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        var zbuf: array<f32, 8u>;
        var sum_d = f32_ro[U.x_logw_off + x];
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            var m = -3.4e38;
            var acc = 0.0;
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let v = f32_ro[U.t_logw_off + t] + cell_l(p, d, t, x);
                if (v > m) { acc = acc * exp(m - v) + 1.0; m = v; } else { acc = acc + exp(v - m); }
            }
            let z = m + log(acc);
            zbuf[d] = z;
            sum_d = sum_d + z;
        }
        let px = exp(sum_d - lp);
        for (var k = 0u; k < U.latent_dim; k = k + 1u) {
            xe[k] = xe[k] + px * f32_ro[U.x_grid_off + x * U.latent_dim + k];
        }
        for (var d = 0u; d < U.n_dims; d = d + 1u) {
            for (var t = 0u; t < U.q_t; t = t + 1u) {
                let theta = f32_ro[U.prior_mean_off + d]
                    + f32_ro[U.prior_sd_off + d] * f32_ro[U.t_nodes_off + t];
                let pt = exp(f32_ro[U.t_logw_off + t] + cell_l(p, d, t, x) - zbuf[d]);
                te[d] = te[d] + px * pt * theta;
                tm2[d] = tm2[d] + px * pt * theta * theta;
            }
        }
    }
    for (var d = 0u; d < U.n_dims; d = d + 1u) {
        outputs[p * U.n_dims + d] = te[d];
        let vv = tm2[d] - te[d] * te[d];
        outputs[U.theta_sd_off + p * U.n_dims + d] = sqrt(max(vv, 0.0));
    }
    for (var k = 0u; k < U.latent_dim; k = k + 1u) {
        outputs[U.xi_eap_off + p * U.latent_dim + k] = xe[k];
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
    logp1_off: u32,
    c0_off: u32,
    t_logw_off: u32,
    x_logw_off: u32,
    t_nodes_off: u32,
    x_grid_off: u32,
    prior_mean_off: u32,
    prior_sd_off: u32,
    pos_off_off: u32,
    pos_items_off: u32,
    miss_off_off: u32,
    miss_items_off: u32,
    theta_sd_off: u32,
    xi_eap_off: u32,
    loglik_off: u32,
    _p2: u32,
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

    let mut f32_ro_host = Vec::new();
    let logp0_off = append_f64_as_f32(&mut f32_ro_host, inp.logp0)?;
    debug_assert_eq!(logp0_off, 0);
    let logp1_off = append_f64_as_f32(&mut f32_ro_host, inp.logp1)?;
    let c0_off = append_f64_as_f32(&mut f32_ro_host, inp.c0)?;
    let t_logw_off = append_f64_as_f32(&mut f32_ro_host, inp.t_logw)?;
    let x_logw_off = append_f64_as_f32(&mut f32_ro_host, inp.x_logw)?;
    let t_nodes_off = append_f64_as_f32(&mut f32_ro_host, inp.t_nodes)?;
    let x_grid_off = append_f64_as_f32(&mut f32_ro_host, inp.x_grid)?;
    let prior_mean_off = append_f64_as_f32(&mut f32_ro_host, inp.prior_mean)?;
    let prior_sd_off = append_f64_as_f32(&mut f32_ro_host, inp.prior_sd)?;

    let mut u32_ro_host = Vec::new();
    let factor_id_off = append_usize_as_u32(&mut u32_ro_host, inp.factor_id)?;
    debug_assert_eq!(factor_id_off, 0);
    let pos_off_off = append_u32(&mut u32_ro_host, inp.pos_off)?;
    let pos_items_off = append_u32(&mut u32_ro_host, inp.pos_items)?;
    let miss_off_off = append_u32(&mut u32_ro_host, inp.miss_off)?;
    let miss_items_off = append_u32(&mut u32_ro_host, inp.miss_items)?;

    let theta_eap_len = inp.n_persons.checked_mul(inp.n_dims)?;
    let theta_sd_off = u32::try_from(theta_eap_len).ok()?;
    let xi_eap_len = inp.n_persons.checked_mul(inp.latent_dim)?;
    let xi_eap_off = u32::try_from(theta_eap_len.checked_add(theta_eap_len)?).ok()?;
    let loglik_off = u32::try_from(
        theta_eap_len
            .checked_add(theta_eap_len)?
            .checked_add(xi_eap_len)?,
    )
    .ok()?;
    let output_len = usize::try_from(loglik_off).ok()?.checked_add(inp.n_persons)?;

    let uniforms = ScoreUniforms {
        n_persons: u32::try_from(inp.n_persons).ok()?,
        n_items: u32::try_from(inp.n_items).ok()?,
        n_dims: u32::try_from(inp.n_dims).ok()?,
        latent_dim: u32::try_from(inp.latent_dim).ok()?,
        q_t: u32::try_from(inp.q_t).ok()?,
        n_x: u32::try_from(inp.n_x).ok()?,
        _p0: 0,
        _p1: 0,
        logp1_off,
        c0_off,
        t_logw_off,
        x_logw_off,
        t_nodes_off,
        x_grid_off,
        prior_mean_off,
        prior_sd_off,
        pos_off_off,
        pos_items_off,
        miss_off_off,
        miss_items_off,
        theta_sd_off,
        xi_eap_off,
        loglik_off,
        _p2: 0,
    };

    let f32_ro_bytes = binding_bytes::<f32>(f32_ro_host.len())?;
    let u32_ro_bytes = binding_bytes::<u32>(u32_ro_host.len())?;
    let output_bytes = binding_bytes::<f32>(output_len)?;
    if !storage_binding_fits(ctx, f32_ro_bytes)
        || !storage_binding_fits(ctx, u32_ro_bytes)
        || !storage_binding_fits(ctx, output_bytes)
    {
        return None;
    }

    let u_buf = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let f32_ro = storage(device, bytemuck::cast_slice(&f32_ro_host), BU::STORAGE);
    let u32_ro = storage(device, bytemuck::cast_slice(&u32_ro_host), BU::STORAGE);
    let outputs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("score-outputs"),
        size: output_bytes.max(4),
        usage: BU::STORAGE | BU::COPY_SRC,
        mapped_at_creation: false,
    });

    let entries = [
        (0, &u_buf),
        (1, &f32_ro),
        (2, &u32_ro),
        (3, &outputs),
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
        let n_persons = u32::try_from(inp.n_persons).ok()?;
        pass.dispatch_workgroups(n_persons.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("score-read"),
        size: output_bytes.max(4),
        usage: BU::MAP_READ | BU::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&outputs, 0, &read, 0, output_bytes.max(4));
    queue.submit([encoder.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let view = read.slice(..).get_mapped_range().ok()?;
    let floats: &[f32] = bytemuck::cast_slice(&view);
    let theta_eap = floats[..theta_eap_len]
        .iter()
        .map(|&value| value as f64)
        .collect();
    let theta_sd_start = usize::try_from(theta_sd_off).ok()?;
    let theta_sd = floats[theta_sd_start..theta_sd_start + theta_eap_len]
        .iter()
        .map(|&value| value as f64)
        .collect();
    let xi_eap_start = usize::try_from(xi_eap_off).ok()?;
    let xi_eap = floats[xi_eap_start..xi_eap_start + xi_eap_len]
        .iter()
        .map(|&value| value as f64)
        .collect();
    let loglik_start = usize::try_from(loglik_off).ok()?;
    let loglik = floats[loglik_start..loglik_start + inp.n_persons]
        .iter()
        .map(|&value| value as f64)
        .collect();
    drop(view);
    read.unmap();

    Some(GpuScoreOutputs {
        theta_eap,
        theta_sd,
        xi_eap,
        loglik,
    })
}
