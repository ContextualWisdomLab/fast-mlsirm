//! wgpu-based GPGPU implementation of the neg-loglik + gradient hot path.
//!
//! This module offloads the dominant `O(n_persons * n_items * latent_dim)`
//! likelihood and gradient work to a GPU through [`wgpu`], a permissive
//! (MIT/Apache-2.0) cross-platform GPGPU abstraction that targets Metal,
//! Vulkan, DX12 and GL. It is a sub-option of the Rust backend, not a separate
//! Python compute backend: callers reach it through
//! [`crate::neg_loglik_and_grad_device`] with [`crate::Device::Gpu`] or
//! [`crate::Device::Auto`].
//!
//! ## Race-free reduction strategy
//!
//! WGSL has no `f64` and no atomic floating-point add, so accumulating
//! gradients with `atomicAdd` is impossible. Instead every gradient slot is
//! owned by exactly one invocation that loops over its contributing axis, which
//! removes all write races without atomics:
//!
//! * `compute_e` — one thread per (person, item): residual `e`, per-entry
//!   objective and the `tau` gradient contribution.
//! * `grad_b_alpha` — one thread per item: reduces over persons.
//! * `grad_theta` — one thread per (person, dim): reduces over items.
//! * `grad_xi` — one thread per (person, latent): reduces over items.
//! * `grad_zeta` — one thread per (item, latent): reduces over persons.
//!
//! The per-entry objective and `tau` contributions are summed on the CPU after
//! read-back (an `O(n_persons * n_items)` pass), and the smooth L2 penalty is
//! added on the CPU by reusing [`crate::add_penalty`], exactly mirroring the CPU
//! reference path.
//!
//! ## Precision
//!
//! Kernels run in `f32` (WGSL's widest float). The CPU reference path is `f64`
//! and is what the numerical-parity tests assert against; the GPU path targets
//! the looser `~1e-4` agreement appropriate for single precision. When no GPU
//! adapter is available (e.g. CI), initialization returns `None` and the caller
//! falls back to the `f64` CPU implementation.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{add_penalty, model_exec_flags, Gradients, ModelConfig, Params, PenaltyConfig};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    free_alpha: u32,
    uses_space: u32,
    gamma: f32,
    eps: f32,
}

const SHADER: &str = r#"
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    free_alpha: u32,
    uses_space: u32,
    gamma: f32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read> factor_id: array<u32>;
@group(0) @binding(4) var<storage, read> theta: array<f32>;
@group(0) @binding(5) var<storage, read> alpha: array<f32>;
@group(0) @binding(6) var<storage, read> b: array<f32>;
@group(0) @binding(7) var<storage, read> xi: array<f32>;
@group(0) @binding(8) var<storage, read> zeta: array<f32>;
@group(0) @binding(9) var<storage, read_write> e_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> objpart: array<f32>;
@group(0) @binding(11) var<storage, read_write> taupart: array<f32>;
@group(0) @binding(12) var<storage, read_write> grad_b: array<f32>;
@group(0) @binding(13) var<storage, read_write> grad_alpha: array<f32>;
@group(0) @binding(14) var<storage, read_write> grad_theta: array<f32>;
@group(0) @binding(15) var<storage, read_write> grad_xi: array<f32>;
@group(0) @binding(16) var<storage, read_write> grad_zeta: array<f32>;

fn sigmoidf(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let ex = exp(x);
    return ex / (1.0 + ex);
}

fn softplusf(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

fn dist_r(p: u32, i: u32) -> f32 {
    var dist2 = U.eps;
    for (var k: u32 = 0u; k < U.latent_dim; k = k + 1u) {
        let diff = xi[p * U.latent_dim + k] - zeta[i * U.latent_dim + k];
        dist2 = dist2 + diff * diff;
    }
    return sqrt(dist2);
}

fn item_a(i: u32) -> f32 {
    if (U.free_alpha == 1u) {
        return exp(alpha[i]);
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn compute_e(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = U.n_persons * U.n_items;
    if (idx >= total) {
        return;
    }
    let p = idx / U.n_items;
    let i = idx % U.n_items;
    if (mask[idx] == 0u) {
        e_buf[idx] = 0.0;
        objpart[idx] = 0.0;
        taupart[idx] = 0.0;
        return;
    }
    let d = factor_id[i];
    let a = item_a(i);
    var r = 0.0;
    if (U.uses_space == 1u) {
        r = dist_r(p, i);
    }
    let eta = a * theta[p * U.n_dims + d] + b[i] - U.gamma * r;
    let pi = sigmoidf(eta);
    let resp = y[idx];
    objpart[idx] = softplusf(eta) - resp * eta;
    let ee = pi - resp;
    e_buf[idx] = ee;
    taupart[idx] = ee * (-U.gamma * r);
}

@compute @workgroup_size(64)
fn grad_b_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.n_items) {
        return;
    }
    let d = factor_id[i];
    let a = item_a(i);
    var sb = 0.0;
    var sa = 0.0;
    for (var p: u32 = 0u; p < U.n_persons; p = p + 1u) {
        let ee = e_buf[p * U.n_items + i];
        sb = sb + ee;
        if (U.free_alpha == 1u) {
            sa = sa + ee * a * theta[p * U.n_dims + d];
        }
    }
    grad_b[i] = sb;
    if (U.free_alpha == 1u) {
        grad_alpha[i] = sa;
    } else {
        grad_alpha[i] = 0.0;
    }
}

@compute @workgroup_size(64)
fn grad_theta_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = U.n_persons * U.n_dims;
    if (idx >= total) {
        return;
    }
    let p = idx / U.n_dims;
    let d = idx % U.n_dims;
    var s = 0.0;
    for (var i: u32 = 0u; i < U.n_items; i = i + 1u) {
        if (factor_id[i] == d) {
            s = s + e_buf[p * U.n_items + i] * item_a(i);
        }
    }
    grad_theta[idx] = s;
}

@compute @workgroup_size(64)
fn grad_xi_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = U.n_persons * U.latent_dim;
    if (idx >= total) {
        return;
    }
    if (U.uses_space == 0u) {
        grad_xi[idx] = 0.0;
        return;
    }
    let p = idx / U.latent_dim;
    let k = idx % U.latent_dim;
    var s = 0.0;
    for (var i: u32 = 0u; i < U.n_items; i = i + 1u) {
        let ee = e_buf[p * U.n_items + i];
        let r = dist_r(p, i);
        let diffk = xi[p * U.latent_dim + k] - zeta[i * U.latent_dim + k];
        let cterm = U.gamma * diffk / r;
        s = s + ee * (-cterm);
    }
    grad_xi[idx] = s;
}

@compute @workgroup_size(64)
fn grad_zeta_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = U.n_items * U.latent_dim;
    if (idx >= total) {
        return;
    }
    if (U.uses_space == 0u) {
        grad_zeta[idx] = 0.0;
        return;
    }
    let i = idx / U.latent_dim;
    let k = idx % U.latent_dim;
    var s = 0.0;
    for (var p: u32 = 0u; p < U.n_persons; p = p + 1u) {
        let ee = e_buf[p * U.n_items + i];
        let r = dist_r(p, i);
        let diffk = xi[p * U.latent_dim + k] - zeta[i * U.latent_dim + k];
        let cterm = U.gamma * diffk / r;
        s = s + ee * cterm;
    }
    grad_zeta[idx] = s;
}
"#;

/// Persistent GPU device, queue and compiled pipelines.
///
/// Initialization (adapter + device request, shader compilation) is expensive
/// and is done once; the optimizer calls the objective thousands of times.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    compute_e: wgpu::ComputePipeline,
    grad_b_alpha: wgpu::ComputePipeline,
    grad_theta: wgpu::ComputePipeline,
    grad_xi: wgpu::ComputePipeline,
    grad_zeta: wgpu::ComputePipeline,
}

static CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuContext {
    fn init() -> Option<GpuContext> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-gpgpu"),
            // Request the adapter's real limits so the 17-binding layout fits on
            // hardware that exceeds the conservative downlevel defaults.
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-neg-loglik"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
        // Bindings 1..=8 are read-only inputs; 9..=16 are read-write outputs.
        for binding in 1..=16u32 {
            entries.push(storage_entry(binding, binding <= 8));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlsirm-gpgpu-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mlsirm-gpgpu-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Some(GpuContext {
            compute_e: make("compute_e"),
            grad_b_alpha: make("grad_b_alpha"),
            grad_theta: make("grad_theta_kernel"),
            grad_xi: make("grad_xi_kernel"),
            grad_zeta: make("grad_zeta_kernel"),
            layout,
            device,
            queue,
        })
    }

    fn get() -> Option<&'static GpuContext> {
        CONTEXT.get_or_init(GpuContext::init).as_ref()
    }
}

fn storage_init(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buffer(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn staging_buffer(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_mapped(buffer: &wgpu::Buffer) -> Option<Vec<f32>> {
    let view = buffer.slice(..).get_mapped_range().ok()?;
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    buffer.unmap();
    Some(values)
}

fn dispatch_count(total: usize) -> u32 {
    total.div_ceil(WORKGROUP_SIZE as usize) as u32
}

/// GPGPU evaluation of the penalized negative log-likelihood and its gradient.
///
/// Returns `None` when no compatible GPU adapter can be initialized, signalling
/// the caller to fall back to the CPU implementation.
pub(crate) fn neg_loglik_and_grad_gpu(
    y: &[f64],
    mask: Option<&[bool]>,
    factor_id: &[usize],
    params: &Params,
    config: &ModelConfig,
    penalty: &PenaltyConfig,
) -> Option<(f64, Gradients, f64)> {
    let ctx = GpuContext::get()?;
    let device = &ctx.device;

    let n_persons = config.n_persons;
    let n_items = config.n_items;
    let n_dims = config.n_dims;
    let latent_dim = config.latent_dim;
    let n = n_persons * n_items;

    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
    let gamma = if uses_space { params.tau.exp() } else { 0.0 };

    // Host-side conversion to f32 (WGSL has no f64).
    let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mask_u32: Vec<u32> = match mask {
        Some(m) => m.iter().map(|&b| u32::from(b)).collect(),
        None => vec![1u32; n],
    };
    let factor_u32: Vec<u32> = factor_id.iter().map(|&v| v as u32).collect();
    let theta_f32: Vec<f32> = params.theta.iter().map(|&v| v as f32).collect();
    let alpha_f32: Vec<f32> = params.alpha.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = params.b.iter().map(|&v| v as f32).collect();
    let xi_f32: Vec<f32> = params.xi.iter().map(|&v| v as f32).collect();
    let zeta_f32: Vec<f32> = params.zeta.iter().map(|&v| v as f32).collect();

    let uniforms = Uniforms {
        n_persons: n_persons as u32,
        n_items: n_items as u32,
        n_dims: n_dims as u32,
        latent_dim: latent_dim as u32,
        free_alpha: u32::from(free_alpha),
        uses_space: u32::from(uses_space),
        gamma: gamma as f32,
        eps: config.eps_distance as f32,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let y_buffer = storage_init(device, "y", &y_f32);
    let mask_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mask"),
        contents: bytemuck::cast_slice(&mask_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let factor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("factor_id"),
        contents: bytemuck::cast_slice(&factor_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let theta_buffer = storage_init(device, "theta", &theta_f32);
    let alpha_buffer = storage_init(device, "alpha", &alpha_f32);
    let b_buffer = storage_init(device, "b", &b_f32);
    let xi_buffer = storage_init(device, "xi", &xi_f32);
    let zeta_buffer = storage_init(device, "zeta", &zeta_f32);

    let e_buffer = output_buffer(device, "e", n);
    let objpart_buffer = output_buffer(device, "objpart", n);
    let taupart_buffer = output_buffer(device, "taupart", n);
    let grad_b_buffer = output_buffer(device, "grad_b", n_items);
    let grad_alpha_buffer = output_buffer(device, "grad_alpha", n_items);
    let grad_theta_buffer = output_buffer(device, "grad_theta", n_persons * n_dims);
    let grad_xi_buffer = output_buffer(device, "grad_xi", n_persons * latent_dim);
    let grad_zeta_buffer = output_buffer(device, "grad_zeta", n_items * latent_dim);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-gpgpu-bind-group"),
        layout: &ctx.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mask_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: factor_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: theta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: alpha_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: xi_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: zeta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: e_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: objpart_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: taupart_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: grad_b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: grad_alpha_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: grad_theta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: grad_xi_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: grad_zeta_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mlsirm"),
    });

    // Each kernel runs in its own compute pass so that writes from one pass are
    // visible to the next within this submission (WebGPU pass ordering).
    let passes: [(&wgpu::ComputePipeline, u32); 5] = [
        (&ctx.compute_e, dispatch_count(n)),
        (&ctx.grad_b_alpha, dispatch_count(n_items)),
        (&ctx.grad_theta, dispatch_count(n_persons * n_dims)),
        (&ctx.grad_xi, dispatch_count(n_persons * latent_dim)),
        (&ctx.grad_zeta, dispatch_count(n_items * latent_dim)),
    ];
    for (pipeline, groups) in passes {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    let objpart_staging = staging_buffer(device, "objpart_read", n);
    let taupart_staging = staging_buffer(device, "taupart_read", n);
    let grad_b_staging = staging_buffer(device, "grad_b_read", n_items);
    let grad_alpha_staging = staging_buffer(device, "grad_alpha_read", n_items);
    let grad_theta_staging = staging_buffer(device, "grad_theta_read", n_persons * n_dims);
    let grad_xi_staging = staging_buffer(device, "grad_xi_read", n_persons * latent_dim);
    let grad_zeta_staging = staging_buffer(device, "grad_zeta_read", n_items * latent_dim);

    let copies: [(&wgpu::Buffer, &wgpu::Buffer, usize); 7] = [
        (&objpart_buffer, &objpart_staging, n),
        (&taupart_buffer, &taupart_staging, n),
        (&grad_b_buffer, &grad_b_staging, n_items),
        (&grad_alpha_buffer, &grad_alpha_staging, n_items),
        (&grad_theta_buffer, &grad_theta_staging, n_persons * n_dims),
        (&grad_xi_buffer, &grad_xi_staging, n_persons * latent_dim),
        (&grad_zeta_buffer, &grad_zeta_staging, n_items * latent_dim),
    ];
    for (src, dst, len) in copies {
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, (len * std::mem::size_of::<f32>()) as u64);
    }

    ctx.queue.submit(Some(encoder.finish()));

    for staging in [
        &objpart_staging,
        &taupart_staging,
        &grad_b_staging,
        &grad_alpha_staging,
        &grad_theta_staging,
        &grad_xi_staging,
        &grad_zeta_staging,
    ] {
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;

    let objpart = read_mapped(&objpart_staging)?;
    let taupart = read_mapped(&taupart_staging)?;
    let grad_b = read_mapped(&grad_b_staging)?;
    let grad_alpha = read_mapped(&grad_alpha_staging)?;
    let grad_theta = read_mapped(&grad_theta_staging)?;
    let grad_xi = read_mapped(&grad_xi_staging)?;
    let grad_zeta = read_mapped(&grad_zeta_staging)?;

    let objective_data: f64 = objpart.iter().map(|&v| v as f64).sum();
    let grad_tau: f64 = taupart.iter().map(|&v| v as f64).sum();
    let loglik = -objective_data;

    let mut grad = Gradients {
        theta: grad_theta.iter().map(|&v| v as f64).collect(),
        alpha: grad_alpha.iter().map(|&v| v as f64).collect(),
        b: grad_b.iter().map(|&v| v as f64).collect(),
        xi: grad_xi.iter().map(|&v| v as f64).collect(),
        zeta: grad_zeta.iter().map(|&v| v as f64).collect(),
        tau: grad_tau,
    };

    // The smooth L2 penalty is identical on CPU and GPU; add it in f64 on the
    // host by reusing the reference implementation.
    let mut objective = objective_data;
    objective += add_penalty(params, config, penalty, free_alpha, uses_space, &mut grad);

    Some((objective, grad, loglik))
}
