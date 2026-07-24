//! wgpu posterior reduction and sampling for plausible values.
//!
//! The CPU builds the same fixed-bank probability tables used by EAP scoring
//! and the deterministic uniform stream.  The GPU performs the expensive
//! person/grid posterior reductions and categorical selections.  Returning
//! `None` keeps the public path able to fall back to the f64 CPU reference.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlausibleUniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    q_t: u32,
    n_x: u32,
    n_draws: u32,
    _pad0: u32,
    _pad1: u32,
}

const SHADER: &str = r#"
struct PlausibleUniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    q_t: u32,
    n_x: u32,
    n_draws: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> U: PlausibleUniforms;
@group(0) @binding(1) var<storage, read> logp0: array<f32>;
@group(0) @binding(2) var<storage, read> logp1: array<f32>;
@group(0) @binding(3) var<storage, read> c0: array<f32>;
@group(0) @binding(4) var<storage, read> t_logw: array<f32>;
@group(0) @binding(5) var<storage, read> x_logw: array<f32>;
@group(0) @binding(6) var<storage, read> t_nodes: array<f32>;
@group(0) @binding(7) var<storage, read> prior_mean: array<f32>;
@group(0) @binding(8) var<storage, read> prior_sd: array<f32>;
@group(0) @binding(9) var<storage, read> factor_id: array<u32>;
@group(0) @binding(10) var<storage, read> pos_off: array<u32>;
@group(0) @binding(11) var<storage, read> pos_items: array<u32>;
@group(0) @binding(12) var<storage, read> miss_off: array<u32>;
@group(0) @binding(13) var<storage, read> miss_items: array<u32>;
@group(0) @binding(14) var<storage, read> random_uniforms: array<f32>;
@group(0) @binding(15) var<storage, read_write> log_zdx: array<f32>;
@group(0) @binding(16) var<storage, read_write> samples: array<f32>;

fn response_loglik(person: u32, dim: u32, t: u32, x: u32) -> f32 {
    let cell = U.q_t * U.n_x;
    var value = c0[dim * cell + t * U.n_x + x];
    for (var j = miss_off[person]; j < miss_off[person + 1u]; j = j + 1u) {
        let item = miss_items[j];
        if (factor_id[item] == dim) {
            let node = item * cell + t * U.n_x + x;
            value = value - logp0[node];
        }
    }
    for (var j = pos_off[person]; j < pos_off[person + 1u]; j = j + 1u) {
        let item = pos_items[j];
        if (factor_id[item] == dim) {
            let node = item * cell + t * U.n_x + x;
            value = value + logp1[node] - logp0[node];
        }
    }
    return value;
}

@compute @workgroup_size(64)
fn posterior_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_persons * U.n_dims * U.n_x;
    if (flat >= count) { return; }
    let x = flat % U.n_x;
    let pd = flat / U.n_x;
    let dim = pd % U.n_dims;
    let person = pd / U.n_dims;
    var maximum = -3.402823e38;
    for (var t = 0u; t < U.q_t; t = t + 1u) {
        let value = t_logw[t] + response_loglik(person, dim, t, x);
        maximum = max(maximum, value);
    }
    var total = 0.0;
    for (var t = 0u; t < U.q_t; t = t + 1u) {
        let value = t_logw[t] + response_loglik(person, dim, t, x);
        total = total + exp(value - maximum);
    }
    log_zdx[flat] = maximum + log(total);
}

fn x_log_weight(person: u32, x: u32) -> f32 {
    var value = x_logw[x];
    for (var dim = 0u; dim < U.n_dims; dim = dim + 1u) {
        value = value + log_zdx[(person * U.n_dims + dim) * U.n_x + x];
    }
    return value;
}

fn finite(value: f32) -> bool {
    return value == value && abs(value) < 3.402823e38;
}

fn write_invalid(person: u32, draw: u32) {
    let invalid = bitcast<f32>(0x7fc00000u);
    for (var dim = 0u; dim < U.n_dims; dim = dim + 1u) {
        samples[(person * U.n_draws + draw) * U.n_dims + dim] = invalid;
    }
}

@compute @workgroup_size(64)
fn sample_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_persons * U.n_draws;
    if (flat >= count) { return; }
    let draw = flat % U.n_draws;
    let person = flat / U.n_draws;
    let random_base = flat * (U.n_dims + 1u);

    var x_maximum = -3.402823e38;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        x_maximum = max(x_maximum, x_log_weight(person, x));
    }
    if (!finite(x_maximum)) {
        write_invalid(person, draw);
        return;
    }
    var x_total = 0.0;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        x_total = x_total + exp(x_log_weight(person, x) - x_maximum);
    }
    if (!finite(x_total) || x_total <= 0.0) {
        write_invalid(person, draw);
        return;
    }
    let x_target = random_uniforms[random_base] * x_total;
    var x_acc = 0.0;
    var x_sel = U.n_x - 1u;
    for (var x = 0u; x < U.n_x; x = x + 1u) {
        x_acc = x_acc + exp(x_log_weight(person, x) - x_maximum);
        if (x_target <= x_acc) {
            x_sel = x;
            break;
        }
    }

    for (var dim = 0u; dim < U.n_dims; dim = dim + 1u) {
        var t_maximum = -3.402823e38;
        for (var t = 0u; t < U.q_t; t = t + 1u) {
            let value = t_logw[t] + response_loglik(person, dim, t, x_sel);
            t_maximum = max(t_maximum, value);
        }
        if (!finite(t_maximum)) {
            write_invalid(person, draw);
            return;
        }
        var t_total = 0.0;
        for (var t = 0u; t < U.q_t; t = t + 1u) {
            let value = t_logw[t] + response_loglik(person, dim, t, x_sel);
            t_total = t_total + exp(value - t_maximum);
        }
        if (!finite(t_total) || t_total <= 0.0) {
            write_invalid(person, draw);
            return;
        }
        let t_target = random_uniforms[random_base + 1u + dim] * t_total;
        var t_acc = 0.0;
        var t_sel = U.q_t - 1u;
        for (var t = 0u; t < U.q_t; t = t + 1u) {
            let value = t_logw[t] + response_loglik(person, dim, t, x_sel);
            t_acc = t_acc + exp(value - t_maximum);
            if (t_target <= t_acc) {
                t_sel = t;
                break;
            }
        }
        samples[(person * U.n_draws + draw) * U.n_dims + dim] =
            prior_mean[dim] + prior_sd[dim] * t_nodes[t_sel];
    }
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    posterior_pipeline: wgpu::ComputePipeline,
    sample_pipeline: wgpu::ComputePipeline,
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
        let adapter_limits = adapter.limits();
        if adapter_limits.max_storage_buffers_per_shader_stage < 16
            || adapter_limits.max_uniform_buffers_per_shader_stage < 1
        {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-plausible-values"),
            required_limits: adapter_limits,
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-plausible-values"),
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
        for binding in 1..=16u32 {
            entries.push(storage_entry(binding, binding <= 14));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlsirm-plausible-values-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mlsirm-plausible-values-pipeline-layout"),
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
        Some(Self {
            posterior_pipeline: make_pipeline("posterior_pass"),
            sample_pipeline: make_pipeline("sample_pass"),
            device,
            queue,
            layout,
        })
    }

    fn get() -> Option<&'static Self> {
        CONTEXT.get_or_init(Self::init).as_ref()
    }
}

pub(crate) struct GpuPlausibleInputs<'a> {
    pub n_persons: usize,
    pub n_items: usize,
    pub n_dims: usize,
    pub q_t: usize,
    pub n_x: usize,
    pub n_draws: usize,
    pub logp0: &'a [f64],
    pub logp1: &'a [f64],
    pub c0: &'a [f64],
    pub t_logw: &'a [f64],
    pub x_logw: &'a [f64],
    pub t_nodes: &'a [f64],
    pub prior_mean: &'a [f64],
    pub prior_sd: &'a [f64],
    pub factor_id: &'a [usize],
    pub pos_off: &'a [u32],
    pub pos_items: &'a [u32],
    pub miss_off: &'a [u32],
    pub miss_items: &'a [u32],
    pub random_uniforms: &'a [f64],
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

fn storage(device: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage,
    })
}

fn storage_f32(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    let placeholder = [0.0_f32];
    let data = if values.is_empty() {
        &placeholder[..]
    } else {
        values
    };
    storage(
        device,
        bytemuck::cast_slice(data),
        wgpu::BufferUsages::STORAGE,
    )
}

fn storage_u32(device: &wgpu::Device, values: &[u32]) -> wgpu::Buffer {
    let placeholder = [0_u32];
    let data = if values.is_empty() {
        &placeholder[..]
    } else {
        values
    };
    storage(
        device,
        bytemuck::cast_slice(data),
        wgpu::BufferUsages::STORAGE,
    )
}

fn output(device: &wgpu::Device, len: usize, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
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
        label: Some("plausible-values-readback"),
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

fn buffer_fits(limits: &wgpu::Limits, len: usize) -> bool {
    let Some(bytes) = len.checked_mul(std::mem::size_of::<f32>()) else {
        return false;
    };
    bytes as u64 <= limits.max_buffer_size
        && bytes <= limits.max_storage_buffer_binding_size as usize
}

/// Run plausible-value posterior reduction and draws on a usable GPU.
/// Returns `None` for unavailable hardware, unsupported sizes, non-finite f32
/// inputs, or an invalid GPU result so the caller can use the f64 CPU path.
pub(crate) fn plausible_values_gpu(inputs: &GpuPlausibleInputs<'_>) -> Option<Vec<f64>> {
    if inputs.n_persons == 0 || inputs.n_draws == 0 {
        return None;
    }
    let posterior_count = inputs
        .n_persons
        .checked_mul(inputs.n_dims)?
        .checked_mul(inputs.n_x)?;
    let draw_count = inputs.n_persons.checked_mul(inputs.n_draws)?;
    let output_count = draw_count.checked_mul(inputs.n_dims)?;
    let random_count = draw_count.checked_mul(inputs.n_dims.checked_add(1)?)?;
    if posterior_count > u32::MAX as usize
        || draw_count > u32::MAX as usize
        || output_count > u32::MAX as usize
        || inputs.n_persons > u32::MAX as usize
        || inputs.n_items > u32::MAX as usize
        || inputs.n_dims > u32::MAX as usize
        || inputs.q_t > u32::MAX as usize
        || inputs.n_x > u32::MAX as usize
        || inputs.n_draws > u32::MAX as usize
        || inputs.random_uniforms.len() != random_count
    {
        return None;
    }
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    let posterior_workgroups = (posterior_count as u32).div_ceil(WORKGROUP_SIZE);
    let sample_workgroups = (draw_count as u32).div_ceil(WORKGROUP_SIZE);
    if posterior_workgroups > limits.max_compute_workgroups_per_dimension
        || sample_workgroups > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let f64_inputs = [
        inputs.logp0,
        inputs.logp1,
        inputs.c0,
        inputs.t_logw,
        inputs.x_logw,
        inputs.t_nodes,
        inputs.prior_mean,
        inputs.prior_sd,
        inputs.random_uniforms,
    ];
    if f64_inputs
        .iter()
        .any(|values| !buffer_fits(&limits, values.len()))
        || [
            inputs.factor_id.len(),
            inputs.pos_off.len(),
            inputs.pos_items.len(),
            inputs.miss_off.len(),
            inputs.miss_items.len(),
        ]
        .into_iter()
        .any(|len| !buffer_fits(&limits, len))
        || !buffer_fits(&limits, posterior_count)
        || !buffer_fits(&limits, output_count)
    {
        return None;
    }
    let converted: Vec<Vec<f32>> = f64_inputs
        .iter()
        .map(|values| checked_f32(values))
        .collect::<Option<_>>()?;
    let factor_id: Vec<u32> = inputs
        .factor_id
        .iter()
        .map(|&dim| u32::try_from(dim).ok())
        .collect::<Option<_>>()?;

    let device = &context.device;
    let queue = &context.queue;
    let uniforms = PlausibleUniforms {
        n_persons: inputs.n_persons as u32,
        n_items: inputs.n_items as u32,
        n_dims: inputs.n_dims as u32,
        q_t: inputs.q_t as u32,
        n_x: inputs.n_x as u32,
        n_draws: inputs.n_draws as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = storage(
        device,
        bytemuck::bytes_of(&uniforms),
        wgpu::BufferUsages::UNIFORM,
    );
    let logp0 = storage_f32(device, &converted[0]);
    let logp1 = storage_f32(device, &converted[1]);
    let c0 = storage_f32(device, &converted[2]);
    let t_logw = storage_f32(device, &converted[3]);
    let x_logw = storage_f32(device, &converted[4]);
    let t_nodes = storage_f32(device, &converted[5]);
    let prior_mean = storage_f32(device, &converted[6]);
    let prior_sd = storage_f32(device, &converted[7]);
    let factor_id = storage_u32(device, &factor_id);
    let pos_off = storage_u32(device, inputs.pos_off);
    let pos_items = storage_u32(device, inputs.pos_items);
    let miss_off = storage_u32(device, inputs.miss_off);
    let miss_items = storage_u32(device, inputs.miss_items);
    let random_uniforms = storage_f32(device, &converted[8]);
    let log_zdx = output(device, posterior_count, "plausible-values-log-zdx");
    let samples = output(device, output_count, "plausible-values-output");

    let buffers = [
        (0, &uniform_buffer),
        (1, &logp0),
        (2, &logp1),
        (3, &c0),
        (4, &t_logw),
        (5, &x_logw),
        (6, &t_nodes),
        (7, &prior_mean),
        (8, &prior_sd),
        (9, &factor_id),
        (10, &pos_off),
        (11, &pos_items),
        (12, &miss_off),
        (13, &miss_items),
        (14, &random_uniforms),
        (15, &log_zdx),
        (16, &samples),
    ];
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-plausible-values-bind-group"),
        layout: &context.layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.posterior_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(posterior_workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.sample_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(sample_workgroups, 1, 1);
    }
    queue.submit([encoder.finish()]);

    let values = read_f32(device, queue, &samples, output_count)?;
    if values.iter().enumerate().any(|(index, &value)| {
        if !value.is_finite() {
            return true;
        }
        let dim = index % inputs.n_dims;
        !inputs.t_nodes.iter().any(|&node| {
            let expected = inputs.prior_mean[dim] + inputs.prior_sd[dim] * node;
            (value - expected).abs() <= 2.0e-5 * (1.0 + expected.abs())
        })
    }) {
        return None;
    }
    Some(values)
}
