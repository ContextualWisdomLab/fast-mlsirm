//! wgpu kernels for fixed-bank scoring diagnostics.
//!
//! This module owns the accelerator implementation of item and test
//! information and empirical reliability. The parallel f64 implementation in
//! `scoring.rs` remains the hardware-independent fallback and numerical
//! reference.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InformationUniforms {
    n_points: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    free_alpha: u32,
    interaction_kind: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReliabilityUniforms {
    n_persons: u32,
    n_dims: u32,
    _pad0: u32,
    _pad1: u32,
}

const SHADER: &str = r#"
struct InformationUniforms {
    n_points: u32,
    n_items: u32,
    n_dims: u32,
    latent_dim: u32,
    free_alpha: u32,
    interaction_kind: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> U: InformationUniforms;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> intercept: array<f32>;
@group(0) @binding(3) var<storage, read> zeta: array<f32>;
// [gamma, eps_distance]
@group(0) @binding(4) var<storage, read> scalars: array<f32>;
@group(0) @binding(5) var<storage, read> factor_id: array<u32>;
@group(0) @binding(6) var<storage, read> theta: array<f32>;
@group(0) @binding(7) var<storage, read> xi: array<f32>;
@group(0) @binding(8) var<storage, read_write> item_info: array<f32>;
@group(0) @binding(9) var<storage, read_write> test_info: array<f32>;

fn logistic(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let ex = exp(x);
    return ex / (1.0 + ex);
}

fn information_at(point: u32, item: u32) -> f32 {
    let dim = factor_id[item];
    var a = 1.0;
    if (U.free_alpha != 0u) {
        a = exp(alpha[item]);
    }
    var eta = a * theta[point * U.n_dims + dim] + intercept[item];
    if (U.interaction_kind == 1u) {
        var dist2 = scalars[1];
        for (var k = 0u; k < U.latent_dim; k = k + 1u) {
            let diff = xi[point * U.latent_dim + k]
                - zeta[item * U.latent_dim + k];
            dist2 = dist2 + diff * diff;
        }
        eta = eta - scalars[0] * sqrt(dist2);
    } else if (U.interaction_kind == 2u) {
        for (var k = 0u; k < U.latent_dim; k = k + 1u) {
            eta = eta + zeta[item * U.latent_dim + k]
                * xi[point * U.latent_dim + k];
        }
    }
    let probability = logistic(eta);
    return a * a * probability * (1.0 - probability);
}

@compute @workgroup_size(64)
fn item_information_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_points * U.n_items;
    if (flat >= count) { return; }
    let point = flat / U.n_items;
    let item = flat % U.n_items;
    item_info[flat] = information_at(point, item);
}

@compute @workgroup_size(64)
fn test_information_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_points * U.n_dims;
    if (flat >= count) { return; }
    let point = flat / U.n_dims;
    let dim = flat % U.n_dims;
    var total = 0.0;
    for (var item = 0u; item < U.n_items; item = item + 1u) {
        if (factor_id[item] == dim) {
            total = total + item_info[point * U.n_items + item];
        }
    }
    test_info[flat] = total;
}
"#;

const RELIABILITY_SHADER: &str = r#"
struct ReliabilityUniforms {
    n_persons: u32,
    n_dims: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> U: ReliabilityUniforms;
@group(0) @binding(1) var<storage, read> theta_eap: array<f32>;
@group(0) @binding(2) var<storage, read> theta_sd: array<f32>;
@group(0) @binding(3) var<storage, read_write> reliability: array<f32>;

@compute @workgroup_size(64)
fn empirical_reliability_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    if (dim >= U.n_dims) { return; }
    let n = f32(U.n_persons);
    var mean = 0.0;
    for (var person = 0u; person < U.n_persons; person = person + 1u) {
        mean = mean + theta_eap[person * U.n_dims + dim];
    }
    mean = mean / n;
    var variance = 0.0;
    var mse = 0.0;
    for (var person = 0u; person < U.n_persons; person = person + 1u) {
        let cell = person * U.n_dims + dim;
        let centered = theta_eap[cell] - mean;
        variance = variance + centered * centered;
        mse = mse + theta_sd[cell] * theta_sd[cell];
    }
    variance = variance / n;
    mse = mse / n;
    let denominator = variance + mse;
    if (denominator > 0.0) {
        reliability[dim] = variance / denominator;
    } else {
        reliability[dim] = bitcast<f32>(0x7fc00000u);
    }
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    item_pipeline: wgpu::ComputePipeline,
    test_pipeline: wgpu::ComputePipeline,
    reliability_pipeline: wgpu::ComputePipeline,
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
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-scoring-information"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-scoring-information"),
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
        for binding in 1..=9u32 {
            entries.push(storage_entry(binding, binding <= 7));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlsirm-scoring-information-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mlsirm-scoring-information-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let reliability_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-empirical-reliability"),
            source: wgpu::ShaderSource::Wgsl(RELIABILITY_SHADER.into()),
        });
        let reliability_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("empirical_reliability_pass"),
                layout: None,
                module: &reliability_module,
                entry_point: Some("empirical_reliability_pass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Some(Self {
            item_pipeline: make_pipeline("item_information_pass"),
            test_pipeline: make_pipeline("test_information_pass"),
            reliability_pipeline,
            device,
            queue,
            layout,
        })
    }

    fn get() -> Option<&'static Self> {
        CONTEXT.get_or_init(Self::init).as_ref()
    }
}

pub(crate) struct GpuInformationInputs<'a> {
    pub n_points: usize,
    pub n_items: usize,
    pub n_dims: usize,
    pub latent_dim: usize,
    pub free_alpha: bool,
    /// 0 = none, 1 = distance, 2 = inner product.
    pub interaction_kind: u32,
    pub gamma: f64,
    pub eps_distance: f64,
    pub alpha: &'a [f64],
    pub b: &'a [f64],
    pub zeta: &'a [f64],
    pub factor_id: &'a [usize],
    pub theta: &'a [f64],
    pub xi: &'a [f64],
}

pub(crate) struct GpuInformationOutputs {
    pub item_info: Vec<f64>,
    pub test_info: Vec<f64>,
}

fn as_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&value| value as f32).collect()
}

fn checked_f32(values: &[f64]) -> Option<Vec<f32>> {
    values
        .iter()
        .map(|&value| {
            let converted = value as f32;
            converted.is_finite().then_some(converted)
        })
        .collect()
}

fn buffer_fits(limits: &wgpu::Limits, len: usize) -> bool {
    let Some(bytes) = len.checked_mul(std::mem::size_of::<f32>()) else {
        return false;
    };
    bytes as u64 <= limits.max_buffer_size
        && bytes <= limits.max_storage_buffer_binding_size as usize
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
        label: Some("scoring-information-output"),
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
        label: Some("scoring-information-readback"),
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

/// Compute item and per-dimension test information on a usable GPU.
/// Returns `None` when no adapter is available, the flattened outputs exceed
/// WGSL's u32 indexing range, or f32 arithmetic produces invalid information.
pub(crate) fn bank_information_gpu(
    inputs: &GpuInformationInputs<'_>,
) -> Option<GpuInformationOutputs> {
    if inputs.n_points == 0 || inputs.n_items == 0 {
        return None;
    }
    let item_count = inputs.n_points.checked_mul(inputs.n_items)?;
    let test_count = inputs.n_points.checked_mul(inputs.n_dims)?;
    if item_count > u32::MAX as usize
        || test_count > u32::MAX as usize
        || inputs.n_items > u32::MAX as usize
        || inputs.n_dims > u32::MAX as usize
        || inputs.latent_dim > u32::MAX as usize
    {
        return None;
    }
    let context = GpuContext::get()?;
    let device = &context.device;
    let queue = &context.queue;
    use wgpu::BufferUsages as BU;

    let uniforms = InformationUniforms {
        n_points: inputs.n_points as u32,
        n_items: inputs.n_items as u32,
        n_dims: inputs.n_dims as u32,
        latent_dim: inputs.latent_dim as u32,
        free_alpha: u32::from(inputs.free_alpha),
        interaction_kind: inputs.interaction_kind,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let alpha = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.alpha)),
        BU::STORAGE,
    );
    let intercept = storage(device, bytemuck::cast_slice(&as_f32(inputs.b)), BU::STORAGE);
    let zeta = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.zeta)),
        BU::STORAGE,
    );
    let scalars = [inputs.gamma as f32, inputs.eps_distance as f32];
    let scalars = storage(device, bytemuck::cast_slice(&scalars), BU::STORAGE);
    let factor_id: Vec<u32> = inputs.factor_id.iter().map(|&dim| dim as u32).collect();
    let factor_id = storage(device, bytemuck::cast_slice(&factor_id), BU::STORAGE);
    let theta = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.theta)),
        BU::STORAGE,
    );
    let xi = storage(
        device,
        bytemuck::cast_slice(&as_f32(inputs.xi)),
        BU::STORAGE,
    );
    let item_info = output(device, item_count);
    let test_info = output(device, test_count);

    let buffers = [
        (0, &uniform_buffer),
        (1, &alpha),
        (2, &intercept),
        (3, &zeta),
        (4, &scalars),
        (5, &factor_id),
        (6, &theta),
        (7, &xi),
        (8, &item_info),
        (9, &test_info),
    ];
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-scoring-information-bind-group"),
        layout: &context.layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.item_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((item_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.test_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((test_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit([encoder.finish()]);

    let item_info = read_f32(device, queue, &item_info, item_count)?;
    let test_info = read_f32(device, queue, &test_info, test_count)?;
    if item_info
        .iter()
        .chain(&test_info)
        .any(|&value| !value.is_finite() || value < 0.0)
    {
        return None;
    }
    Some(GpuInformationOutputs {
        item_info,
        test_info,
    })
}

/// Compute empirical EAP reliability on a usable GPU. Returns `None` when the
/// request exceeds WebGPU bounds, f64 inputs cannot be represented as finite
/// f32 values, no adapter is available, or the kernel produces an invalid
/// reliability value.
pub(crate) fn empirical_reliability_gpu(
    theta_eap: &[f64],
    theta_sd: &[f64],
    n_persons: usize,
    n_dims: usize,
) -> Option<Vec<f64>> {
    let cell_count = n_persons.checked_mul(n_dims)?;
    if n_persons == 0
        || n_dims == 0
        || theta_eap.len() != cell_count
        || theta_sd.len() != cell_count
        || n_persons > u32::MAX as usize
        || n_dims > u32::MAX as usize
    {
        return None;
    }
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    let workgroups = (n_dims as u32).div_ceil(WORKGROUP_SIZE);
    if workgroups > limits.max_compute_workgroups_per_dimension
        || !buffer_fits(&limits, cell_count)
        || !buffer_fits(&limits, n_dims)
    {
        return None;
    }
    let theta_eap = checked_f32(theta_eap)?;
    let theta_sd = checked_f32(theta_sd)?;
    let device = &context.device;
    let queue = &context.queue;
    use wgpu::BufferUsages as BU;
    let uniforms = ReliabilityUniforms {
        n_persons: n_persons as u32,
        n_dims: n_dims as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = storage(device, bytemuck::bytes_of(&uniforms), BU::UNIFORM);
    let theta_eap = storage(device, bytemuck::cast_slice(&theta_eap), BU::STORAGE);
    let theta_sd = storage(device, bytemuck::cast_slice(&theta_sd), BU::STORAGE);
    let reliability = output(device, n_dims);
    let layout = context.reliability_pipeline.get_bind_group_layout(0);
    let buffers = [
        (0, &uniform_buffer),
        (1, &theta_eap),
        (2, &theta_sd),
        (3, &reliability),
    ];
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-empirical-reliability-bind-group"),
        layout: &layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.reliability_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit([encoder.finish()]);

    let values = read_f32(device, queue, &reliability, n_dims)?;
    if values.iter().any(|&value| {
        !(value.is_nan() || (value.is_finite() && (0.0..=1.000_001).contains(&value)))
    }) {
        return None;
    }
    Some(
        values
            .into_iter()
            .map(|value| {
                if value.is_nan() {
                    value
                } else {
                    value.clamp(0.0, 1.0)
                }
            })
            .collect(),
    )
}
