//! wgpu person-score reduction for crossed / multiple-membership IRT.
//!
//! The kernel evaluates the `O(n_persons * n_items)` Bernoulli residual and
//! Fisher information that the MAP estimator of `u_h` consumes. Sparse
//! membership accumulation and the dense Newton solve remain on the CPU.
//! Kernels run in f32; the CPU path is the f64 reference. Missing adapters
//! return `None` so the estimator falls back without failing the fit.
//!
//! Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
//! IRT model. *Psychometrika, 66*, 271-288.
//! <https://doi.org/10.1007/BF02294839>

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    _pad0: u32,
    _pad1: u32,
}

const SHADER: &str = r#"
struct Uniforms {
    n_persons: u32,
    n_items: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> slopes: array<f32>;
@group(0) @binding(3) var<storage, read> intercepts: array<f32>;
@group(0) @binding(4) var<storage, read> locations: array<f32>;
@group(0) @binding(5) var<storage, read_write> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> information: array<f32>;

fn logistic(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let ex = exp(x);
    return ex / (1.0 + ex);
}

@compute @workgroup_size(64)
fn person_score_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let person = gid.x;
    if (person >= U.n_persons) { return; }
    var score = 0.0;
    var weight = 0.0;
    let location = locations[person];
    for (var item = 0u; item < U.n_items; item = item + 1u) {
        let response = y[person * U.n_items + item];
        if (!(response == response) || response < 0.0) { continue; }
        let slope = slopes[item];
        let probability = logistic(slope * location + intercepts[item]);
        score = score + slope * (response - probability);
        weight = weight + slope * slope * probability * (1.0 - probability);
    }
    residual[person] = score;
    information[person] = weight;
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
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
    fn init() -> Option<Self> {
        let instance = crate::gpu_init::new_instance();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-crossed-person-effects"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-crossed-person-effects"),
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
        for binding in 1..=6u32 {
            entries.push(storage_entry(binding, binding <= 4));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlsirm-crossed-person-effects-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mlsirm-crossed-person-effects-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("person_score_pass"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("person_score_pass"),
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

fn as_f32_finite(values: &[f64]) -> Option<Vec<f32>> {
    values
        .iter()
        .map(|&value| {
            let converted = value as f32;
            converted.is_finite().then_some(converted)
        })
        .collect()
}

fn as_f32_responses(values: &[f64]) -> Option<Vec<f32>> {
    values
        .iter()
        .map(|&value| {
            if value.is_infinite() {
                return None;
            }
            Some(value as f32)
        })
        .collect()
}

fn storage(device: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage,
    })
}

fn output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("crossed-person-effect-output"),
        size: (len.max(1) * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Option<Vec<f64>> {
    let size = (len.max(1) * std::mem::size_of::<f32>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("crossed-person-effect-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let view = readback.slice(..).get_mapped_range().ok()?;
    let values: &[f32] = bytemuck::cast_slice(&view);
    let result = values.iter().take(len).map(|&value| value as f64).collect();
    drop(view);
    readback.unmap();
    Some(result)
}

/// Reduce per-person Bernoulli scores on a usable GPU.
///
/// Returns `None` when no adapter is present, a buffer exceeds device limits,
/// or f32 conversion of a finite f64 input overflows.
pub(crate) fn person_irt_scores_gpu(
    y: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    locations: &[f64],
    n_persons: usize,
    n_items: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    let y_len = n_persons.checked_mul(n_items)?;
    if y.len() != y_len || locations.len() != n_persons {
        return None;
    }
    for &len in &[y_len, n_items, n_persons] {
        let bytes = len.checked_mul(std::mem::size_of::<f32>())?;
        if bytes as u64 > limits.max_buffer_size
            || bytes > limits.max_storage_buffer_binding_size as usize
        {
            return None;
        }
    }
    let y_f32 = as_f32_responses(y)?;
    let slopes = as_f32_finite(item_slopes)?;
    let intercepts = as_f32_finite(item_intercepts)?;
    let locations_f32 = as_f32_finite(locations)?;
    let uniforms = Uniforms {
        n_persons: u32::try_from(n_persons).ok()?,
        n_items: u32::try_from(n_items).ok()?,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buf = storage(
        &context.device,
        bytemuck::bytes_of(&uniforms),
        wgpu::BufferUsages::UNIFORM,
    );
    let y_buf = storage(
        &context.device,
        bytemuck::cast_slice(&y_f32),
        wgpu::BufferUsages::STORAGE,
    );
    let slope_buf = storage(
        &context.device,
        bytemuck::cast_slice(&slopes),
        wgpu::BufferUsages::STORAGE,
    );
    let intercept_buf = storage(
        &context.device,
        bytemuck::cast_slice(&intercepts),
        wgpu::BufferUsages::STORAGE,
    );
    let location_buf = storage(
        &context.device,
        bytemuck::cast_slice(&locations_f32),
        wgpu::BufferUsages::STORAGE,
    );
    let residual_buf = output(&context.device, n_persons);
    let information_buf = output(&context.device, n_persons);
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crossed-person-effect-bind-group"),
            layout: &context.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slope_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: intercept_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: residual_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: information_buf.as_entire_binding(),
                },
            ],
        });
    let groups = n_persons.div_ceil(WORKGROUP_SIZE as usize) as u32;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("crossed-person-effect-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("person_score_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&context.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    context.queue.submit([encoder.finish()]);
    let residual = read_f32(&context.device, &context.queue, &residual_buf, n_persons)?;
    let information = read_f32(&context.device, &context.queue, &information_buf, n_persons)?;
    if residual.iter().any(|value| !value.is_finite())
        || information.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some((residual, information))
}
