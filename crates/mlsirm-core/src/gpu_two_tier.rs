//! wgpu expected-raw totals for two-tier GRM scoring (closed form).
//!
//! One invocation per primary row evaluates
//! `sum_i sum_q w_q sum_k sigmoid(eta_i + beta_ik)` in f32 with compensated
//! (Kahan) summation. Rows are dispatched in bounded chunks so device memory
//! does not grow with the row count. Returning `None` lets the caller use the
//! f64 CPU reference in `two_tier_recursion`.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::two_tier_recursion::TwoTierItemParams;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    n_rows: u32,
    n_items: u32,
    n_primary: u32,
    m1: u32,
    n_nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SHADER: &str = r#"
struct Uniforms {
    n_rows: u32,
    n_items: u32,
    n_primary: u32,
    m1: u32,
    n_nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> theta: array<f32>;
@group(0) @binding(2) var<storage, read> a_primary: array<f32>;
@group(0) @binding(3) var<storage, read> a_specific: array<f32>;
@group(0) @binding(4) var<storage, read> thresholds: array<f32>;
@group(0) @binding(5) var<storage, read> specific_free: array<u32>;
@group(0) @binding(6) var<storage, read> nodes: array<f32>;
@group(0) @binding(7) var<storage, read> weights: array<f32>;
@group(0) @binding(8) var<storage, read_write> out: array<f32>;

fn sigmoid(z: f32) -> f32 {
    if (z >= 0.0) {
        return 1.0 / (1.0 + exp(-z));
    }
    let ez = exp(z);
    return ez / (1.0 + ez);
}

fn item_mean(item: u32, eta: f32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < U.m1; k = k + 1u) {
        s = s + sigmoid(eta + thresholds[item * U.m1 + k]);
    }
    return s;
}

@compute @workgroup_size(64)
fn expected_raw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= U.n_rows) { return; }
    var total = 0.0;
    var comp = 0.0;
    for (var i = 0u; i < U.n_items; i = i + 1u) {
        var base = 0.0;
        for (var d = 0u; d < U.n_primary; d = d + 1u) {
            base = base + a_primary[i * U.n_primary + d] * theta[row * U.n_primary + d];
        }
        var term = 0.0;
        if (specific_free[i] == 1u) {
            term = item_mean(i, base);
        } else {
            for (var q = 0u; q < U.n_nodes; q = q + 1u) {
                term = term + weights[q] * item_mean(i, base + a_specific[i] * nodes[q]);
            }
        }
        // Kahan compensated accumulation across items.
        let y = term - comp;
        let t = total + y;
        comp = (t - total) - y;
        total = t;
    }
    out[row] = total;
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

static CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    fn init() -> Option<Self> {
        let instance = crate::gpu_init::new_instance();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_uniform_buffers_per_shader_stage < 1
        {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-two-tier-score"),
            required_limits: limits,
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-two-tier-expected-raw"),
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
        for binding in 1..=8 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 8,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlsirm-two-tier-score-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mlsirm-two-tier-score-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mlsirm-two-tier-expected-raw"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("expected_raw"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Self {
            device,
            queue,
            layout,
            pipeline,
        })
    }

    fn get() -> Option<&'static Self> {
        CONTEXT.get_or_init(Self::init).as_ref()
    }
}

fn checked_f32(values: &[f64]) -> Option<Vec<f32>> {
    values
        .iter()
        .map(|&v| {
            let c = v as f32;
            c.is_finite().then_some(c)
        })
        .collect()
}

fn buffer_init(device: &wgpu::Device, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes,
        usage,
    })
}

fn fits(limits: &wgpu::Limits, len: usize) -> bool {
    len.checked_mul(4).is_some_and(|bytes| {
        bytes as u64 <= limits.max_buffer_size
            && bytes as u64 <= limits.max_storage_buffer_binding_size
    })
}

/// Conservative predictor admission before f64-to-f32 conversion. Bound the
/// input casts, products and dot-product additions by gamma_n times the sum
/// of absolute terms, not the possibly cancelled predictor. Sigmoid is
/// 1/4-Lipschitz. Reserve one quarter of the scoring test's per-boundary
/// error budget for this propagation; reject non-finite bounds as well.
/// This is deliberately conservative: rejection preserves the f64 owner.
pub(crate) fn predictor_precision_is_safe(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    theta_specific: &[f64],
) -> bool {
    let p = params.n_primary;
    let u = f64::from(f32::EPSILON) / 2.0;
    let operations = 4.0 * p as f64 + 8.0;
    let nu = operations * u;
    if nu >= 1.0 {
        return false;
    }
    let gamma = nu / (1.0 - nu);
    let terms = params.n_items() as f64 * (params.n_cat - 1) as f64 * theta_specific.len() as f64;
    let budget = (8.0 + terms) * u / 4.0;
    let mut theta_max = vec![0.0_f64; p];
    for row in theta_primary.chunks_exact(p) {
        for (maximum, value) in theta_max.iter_mut().zip(row) {
            *maximum = maximum.max(value.abs());
        }
    }
    let specific_max = theta_specific
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    for i in 0..params.n_items() {
        let primary_magnitude: f64 = params.a_primary[i * p..(i + 1) * p]
            .iter()
            .zip(&theta_max)
            .map(|(a, t)| a.abs() * t)
            .sum();
        let specific_magnitude = if params.specific_map[i] == -1 {
            0.0
        } else {
            params.a_specific[i].abs() * specific_max
        };
        for threshold in &params.thresholds[i * (params.n_cat - 1)..(i + 1) * (params.n_cat - 1)] {
            let magnitude = primary_magnitude + specific_magnitude + threshold.abs();
            // Include an absolute allowance for gradual underflow or flushing
            // tiny arithmetic results to zero on a device.
            let error = 0.25 * (gamma * magnitude + operations * f64::from(f32::MIN_POSITIVE));
            if !error.is_finite() || error > budget {
                return false;
            }
        }
    }
    true
}

/// GPU closed-form expected raw totals. Inputs must already be validated by
/// the CPU owner (`two_tier_expected_raw_on`).
pub(crate) fn expected_raw_gpu(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Option<Vec<f64>> {
    let p = params.n_primary;
    let n_rows = theta_primary.len() / p;
    if n_rows == 0 || !predictor_precision_is_safe(params, theta_primary, theta_specific) {
        return None;
    }
    let a_primary = checked_f32(&params.a_primary)?;
    let a_specific = checked_f32(&params.a_specific)?;
    let thresholds = checked_f32(&params.thresholds)?;
    let nodes = checked_f32(theta_specific)?;
    let weights = checked_f32(weights_specific)?;
    let free: Vec<u32> = params
        .specific_map
        .iter()
        .map(|&s| u32::from(s == -1))
        .collect();
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    if [a_primary.len(), thresholds.len(), nodes.len()]
        .into_iter()
        .any(|len| !fits(&limits, len))
    {
        return None;
    }
    // Chunk rows so theta/out buffers stay within one binding and one dispatch.
    let max_by_buffer = (limits.max_storage_buffer_binding_size as usize / 4) / p.max(1);
    let max_by_dispatch =
        limits.max_compute_workgroups_per_dimension as usize * WORKGROUP_SIZE as usize;
    let chunk = max_by_buffer.min(max_by_dispatch).min(1 << 20);
    if chunk == 0 {
        return None;
    }

    let device = &context.device;
    let queue = &context.queue;
    let storage = wgpu::BufferUsages::STORAGE;
    let a_primary_buf = buffer_init(device, bytemuck::cast_slice(&a_primary), storage);
    let a_specific_buf = buffer_init(device, bytemuck::cast_slice(&a_specific), storage);
    let thresholds_buf = buffer_init(device, bytemuck::cast_slice(&thresholds), storage);
    let free_buf = buffer_init(device, bytemuck::cast_slice(&free), storage);
    let nodes_buf = buffer_init(device, bytemuck::cast_slice(&nodes), storage);
    let weights_buf = buffer_init(device, bytemuck::cast_slice(&weights), storage);

    let mut result = Vec::with_capacity(n_rows);
    for start in (0..n_rows).step_by(chunk) {
        let rows = chunk.min(n_rows - start);
        let theta = checked_f32(&theta_primary[start * p..(start + rows) * p])?;
        let uniforms = Uniforms {
            n_rows: u32::try_from(rows).ok()?,
            n_items: u32::try_from(params.n_items()).ok()?,
            n_primary: u32::try_from(p).ok()?,
            m1: u32::try_from(params.n_cat - 1).ok()?,
            n_nodes: u32::try_from(nodes.len()).ok()?,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = buffer_init(
            device,
            bytemuck::bytes_of(&uniforms),
            wgpu::BufferUsages::UNIFORM,
        );
        let theta_buf = buffer_init(device, bytemuck::cast_slice(&theta), storage);
        let size = (rows * 4) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("two-tier-expected-raw-out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("two-tier-expected-raw-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffers = [
            &uniform_buf,
            &theta_buf,
            &a_primary_buf,
            &a_specific_buf,
            &thresholds_buf,
            &free_buf,
            &nodes_buf,
            &weights_buf,
            &out_buf,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mlsirm-two-tier-score-bind-group"),
            layout: &context.layout,
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&context.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(u32::try_from(rows).ok()?.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let view = readback.slice(..).get_mapped_range().ok()?;
        let values: &[f32] = bytemuck::cast_slice(&view);
        result.extend(values.iter().map(|&v| f64::from(v)));
        drop(view);
        readback.unmap();
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}
