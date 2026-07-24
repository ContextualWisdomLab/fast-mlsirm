//! wgpu Lord-Wingersky recursion, EAPsum moment reduction, and table lookup.
//!
//! Fixed-bank probabilities and request validation stay in the shared Rust
//! core. The GPU performs the score-distribution recursion and posterior
//! reductions, or applies already validated conversion tables. Returning
//! `None` lets the public dispatch use the parallel f64 CPU reference.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scoring::EapSumTable;

const WORKGROUP_SIZE: u32 = 64;
const MAX_ITEMS_PER_DIM: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TableUniforms {
    n_dims: u32,
    q_t: u32,
    n_x: u32,
    cell: u32,
    n_scores: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LookupUniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    _pad0: u32,
}

const TABLE_SHADER: &str = r#"
struct TableUniforms {
    n_dims: u32,
    q_t: u32,
    n_x: u32,
    cell: u32,
    n_scores: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> U: TableUniforms;
@group(0) @binding(1) var<storage, read> dim_item_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> dim_items: array<u32>;
@group(0) @binding(3) var<storage, read> score_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> logp1: array<f32>;
@group(0) @binding(5) var<storage, read> t_logw: array<f32>;
@group(0) @binding(6) var<storage, read> x_logw: array<f32>;
@group(0) @binding(7) var<storage, read> t_nodes: array<f32>;
@group(0) @binding(8) var<storage, read> prior_mean: array<f32>;
@group(0) @binding(9) var<storage, read> prior_sd: array<f32>;
@group(0) @binding(10) var<storage, read> score_dims: array<u32>;
@group(0) @binding(11) var<storage, read> score_values: array<u32>;
@group(0) @binding(12) var<storage, read_write> score_dist: array<f32>;
@group(0) @binding(13) var<storage, read_write> score_prob: array<f32>;
@group(0) @binding(14) var<storage, read_write> eap: array<f32>;
@group(0) @binding(15) var<storage, read_write> posterior_sd: array<f32>;

@compute @workgroup_size(64)
fn recursion_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_dims * U.cell;
    if (flat >= count) { return; }
    let dim = flat / U.cell;
    let cell_index = flat % U.cell;
    let begin = dim_item_offsets[dim];
    let end = dim_item_offsets[dim + 1u];
    let n_items_dim = end - begin;
    var dist: array<f32, 257>;
    for (var score = 0u; score <= n_items_dim; score = score + 1u) {
        dist[score] = 0.0;
    }
    dist[0] = 1.0;
    for (var position = begin; position < end; position = position + 1u) {
        let item = dim_items[position];
        let probability = exp(logp1[item * U.cell + cell_index]);
        let seen = position - begin;
        var score = seen + 1u;
        loop {
            let stay = dist[score] * (1.0 - probability);
            var up = 0.0;
            if (score > 0u) {
                up = dist[score - 1u] * probability;
            }
            dist[score] = stay + up;
            if (score == 0u) { break; }
            score = score - 1u;
        }
    }
    let base = score_offsets[dim];
    for (var score = 0u; score <= n_items_dim; score = score + 1u) {
        score_dist[(base + score) * U.cell + cell_index] = dist[score];
    }
}

@compute @workgroup_size(64)
fn reduction_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    if (output_index >= U.n_scores) { return; }
    let dim = score_dims[output_index];
    let score = score_values[output_index];
    var p0 = 0.0;
    var m1 = 0.0;
    var m2 = 0.0;
    for (var cell_index = 0u; cell_index < U.cell; cell_index = cell_index + 1u) {
        let t = cell_index / U.n_x;
        let x = cell_index % U.n_x;
        let weight = exp(t_logw[t] + x_logw[x]);
        let theta = prior_mean[dim] + prior_sd[dim] * t_nodes[t];
        let value = weight * score_dist[(score_offsets[dim] + score) * U.cell + cell_index];
        p0 = p0 + value;
        m1 = m1 + value * theta;
        m2 = m2 + value * theta * theta;
    }
    score_prob[output_index] = p0;
    if (p0 > 0.0) {
        let mean = m1 / p0;
        eap[output_index] = mean;
        posterior_sd[output_index] = sqrt(max(0.0, m2 / p0 - mean * mean));
    } else {
        eap[output_index] = prior_mean[dim];
        posterior_sd[output_index] = prior_sd[dim];
    }
}
"#;

const LOOKUP_SHADER: &str = r#"
struct LookupUniforms {
    n_persons: u32,
    n_items: u32,
    n_dims: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> U: LookupUniforms;
@group(0) @binding(1) var<storage, read> responses: array<f32>;
@group(0) @binding(2) var<storage, read> factor_id: array<u32>;
@group(0) @binding(3) var<storage, read> table_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> table_eap: array<f32>;
@group(0) @binding(5) var<storage, read> table_sd: array<f32>;
@group(0) @binding(6) var<storage, read_write> theta_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> sd_out: array<f32>;

@compute @workgroup_size(64)
fn lookup_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let count = U.n_persons * U.n_dims;
    if (flat >= count) { return; }
    let person = flat / U.n_dims;
    let dim = flat % U.n_dims;
    var score = 0u;
    for (var item = 0u; item < U.n_items; item = item + 1u) {
        if (factor_id[item] == dim) {
            score = score + u32(responses[person * U.n_items + item]);
        }
    }
    let table_index = table_offsets[dim] + score;
    theta_out[flat] = table_eap[table_index];
    sd_out[flat] = table_sd[table_index];
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    table_layout: wgpu::BindGroupLayout,
    recursion_pipeline: wgpu::ComputePipeline,
    reduction_pipeline: wgpu::ComputePipeline,
    lookup_layout: wgpu::BindGroupLayout,
    lookup_pipeline: wgpu::ComputePipeline,
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

fn layout(
    device: &wgpu::Device,
    label: &'static str,
    storage_count: u32,
    writable_from: u32,
) -> wgpu::BindGroupLayout {
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
    for binding in 1..=storage_count {
        entries.push(storage_entry(binding, binding < writable_from));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl GpuContext {
    fn init() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let adapter_limits = adapter.limits();
        if adapter_limits.max_storage_buffers_per_shader_stage < 15
            || adapter_limits.max_uniform_buffers_per_shader_stage < 1
        {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mlsirm-eapsum"),
            required_limits: adapter_limits,
            ..Default::default()
        }))
        .ok()?;
        let table_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-eapsum-table"),
            source: wgpu::ShaderSource::Wgsl(TABLE_SHADER.into()),
        });
        let lookup_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mlsirm-eapsum-lookup"),
            source: wgpu::ShaderSource::Wgsl(LOOKUP_SHADER.into()),
        });
        let table_layout = layout(&device, "mlsirm-eapsum-table-layout", 15, 12);
        let lookup_layout = layout(&device, "mlsirm-eapsum-lookup-layout", 7, 6);
        let recursion_pipeline = pipeline(
            &device,
            &table_module,
            &table_layout,
            "mlsirm-eapsum-recursion",
            "recursion_pass",
        );
        let reduction_pipeline = pipeline(
            &device,
            &table_module,
            &table_layout,
            "mlsirm-eapsum-reduction",
            "reduction_pass",
        );
        let lookup_pipeline = pipeline(
            &device,
            &lookup_module,
            &lookup_layout,
            "mlsirm-eapsum-lookup",
            "lookup_pass",
        );
        Some(Self {
            device,
            queue,
            table_layout,
            recursion_pipeline,
            reduction_pipeline,
            lookup_layout,
            lookup_pipeline,
        })
    }

    fn get() -> Option<&'static Self> {
        CONTEXT.get_or_init(Self::init).as_ref()
    }
}

pub(crate) struct GpuEapSumTableInputs<'a> {
    pub n_items: usize,
    pub n_dims: usize,
    pub q_t: usize,
    pub n_x: usize,
    pub factor_id: &'a [usize],
    pub logp1: &'a [f64],
    pub t_logw: &'a [f64],
    pub x_logw: &'a [f64],
    pub t_nodes: &'a [f64],
    pub prior_mean: &'a [f64],
    pub prior_sd: &'a [f64],
}

pub(crate) struct GpuEapSumLookupInputs<'a> {
    pub y: &'a [f64],
    pub n_persons: usize,
    pub n_items: usize,
    pub n_dims: usize,
    pub factor_id: &'a [usize],
    pub tables: &'a [EapSumTable],
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
    storage(
        device,
        bytemuck::cast_slice(if values.is_empty() {
            &placeholder
        } else {
            values
        }),
        wgpu::BufferUsages::STORAGE,
    )
}

fn storage_u32(device: &wgpu::Device, values: &[u32]) -> wgpu::Buffer {
    let placeholder = [0_u32];
    storage(
        device,
        bytemuck::cast_slice(if values.is_empty() {
            &placeholder
        } else {
            values
        }),
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
        label: Some("eapsum-readback"),
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

pub(crate) fn eapsum_tables_gpu(inputs: &GpuEapSumTableInputs<'_>) -> Option<Vec<EapSumTable>> {
    let cell = inputs.q_t.checked_mul(inputs.n_x)?;
    let work_count = inputs.n_dims.checked_mul(cell)?;
    let mut dim_item_offsets = Vec::with_capacity(inputs.n_dims + 1);
    let mut dim_items = Vec::with_capacity(inputs.n_items);
    let mut score_offsets = Vec::with_capacity(inputs.n_dims + 1);
    let mut score_dims = Vec::new();
    let mut score_values = Vec::new();
    dim_item_offsets.push(0_u32);
    score_offsets.push(0_u32);
    for dim in 0..inputs.n_dims {
        let items: Vec<usize> = (0..inputs.n_items)
            .filter(|&item| inputs.factor_id[item] == dim)
            .collect();
        if items.len() > MAX_ITEMS_PER_DIM {
            return None;
        }
        for item in items {
            dim_items.push(u32::try_from(item).ok()?);
        }
        dim_item_offsets.push(u32::try_from(dim_items.len()).ok()?);
        let n_scores_dim = dim_items.len() - dim_item_offsets[dim] as usize + 1;
        for score in 0..n_scores_dim {
            score_dims.push(u32::try_from(dim).ok()?);
            score_values.push(u32::try_from(score).ok()?);
        }
        score_offsets.push(u32::try_from(score_dims.len()).ok()?);
    }
    let n_scores = score_dims.len();
    let score_dist_len = n_scores.checked_mul(cell)?;
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    let recursion_groups = u32::try_from(work_count).ok()?.div_ceil(WORKGROUP_SIZE);
    let reduction_groups = u32::try_from(n_scores).ok()?.div_ceil(WORKGROUP_SIZE);
    if recursion_groups > limits.max_compute_workgroups_per_dimension
        || reduction_groups > limits.max_compute_workgroups_per_dimension
        || [
            inputs.factor_id.len(),
            inputs.logp1.len(),
            inputs.t_logw.len(),
            inputs.x_logw.len(),
            inputs.t_nodes.len(),
            inputs.prior_mean.len(),
            inputs.prior_sd.len(),
            dim_item_offsets.len(),
            dim_items.len(),
            score_offsets.len(),
            score_dims.len(),
            score_values.len(),
            score_dist_len,
            n_scores,
        ]
        .into_iter()
        .any(|len| !buffer_fits(&limits, len))
    {
        return None;
    }
    let converted = [
        inputs.logp1,
        inputs.t_logw,
        inputs.x_logw,
        inputs.t_nodes,
        inputs.prior_mean,
        inputs.prior_sd,
    ]
    .into_iter()
    .map(checked_f32)
    .collect::<Option<Vec<_>>>()?;
    let device = &context.device;
    let queue = &context.queue;
    let uniforms = TableUniforms {
        n_dims: u32::try_from(inputs.n_dims).ok()?,
        q_t: u32::try_from(inputs.q_t).ok()?,
        n_x: u32::try_from(inputs.n_x).ok()?,
        cell: u32::try_from(cell).ok()?,
        n_scores: u32::try_from(n_scores).ok()?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniform_buffer = storage(
        device,
        bytemuck::bytes_of(&uniforms),
        wgpu::BufferUsages::UNIFORM,
    );
    let dim_item_offsets = storage_u32(device, &dim_item_offsets);
    let dim_items = storage_u32(device, &dim_items);
    let score_offsets_buffer = storage_u32(device, &score_offsets);
    let logp1 = storage_f32(device, &converted[0]);
    let t_logw = storage_f32(device, &converted[1]);
    let x_logw = storage_f32(device, &converted[2]);
    let t_nodes = storage_f32(device, &converted[3]);
    let prior_mean = storage_f32(device, &converted[4]);
    let prior_sd = storage_f32(device, &converted[5]);
    let score_dims = storage_u32(device, &score_dims);
    let score_values = storage_u32(device, &score_values);
    let score_dist = output(device, score_dist_len, "eapsum-score-dist");
    let score_prob = output(device, n_scores, "eapsum-score-prob");
    let eap = output(device, n_scores, "eapsum-eap");
    let posterior_sd = output(device, n_scores, "eapsum-sd");
    let buffers = [
        (0, &uniform_buffer),
        (1, &dim_item_offsets),
        (2, &dim_items),
        (3, &score_offsets_buffer),
        (4, &logp1),
        (5, &t_logw),
        (6, &x_logw),
        (7, &t_nodes),
        (8, &prior_mean),
        (9, &prior_sd),
        (10, &score_dims),
        (11, &score_values),
        (12, &score_dist),
        (13, &score_prob),
        (14, &eap),
        (15, &posterior_sd),
    ];
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-eapsum-table-bind-group"),
        layout: &context.table_layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.recursion_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(recursion_groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.reduction_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(reduction_groups, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let probabilities = read_f32(device, queue, &score_prob, n_scores)?;
    let means = read_f32(device, queue, &eap, n_scores)?;
    let sds = read_f32(device, queue, &posterior_sd, n_scores)?;
    let mut result = Vec::with_capacity(inputs.n_dims);
    for dim in 0..inputs.n_dims {
        let start = score_offsets[dim] as usize;
        let end = score_offsets[dim + 1] as usize;
        let mut score_prob = probabilities[start..end].to_vec();
        let eap = means[start..end].to_vec();
        let sd = sds[start..end].to_vec();
        let total = score_prob.iter().sum::<f64>();
        if !total.is_finite()
            || (total - 1.0).abs() > 5.0e-4
            || score_prob
                .iter()
                .any(|&value| !value.is_finite() || value < 0.0)
            || eap.iter().any(|&value| !value.is_finite())
            || sd.iter().any(|&value| !value.is_finite() || value < 0.0)
        {
            return None;
        }
        // Preserve the public probability-table invariant after f32 reduction.
        // The recursion and moment work stays on the GPU; this bounded host
        // packaging step removes only accumulated roundoff from the score
        // marginal and pins the final entry to the remaining unit mass.
        for value in &mut score_prob {
            *value /= total;
        }
        if let Some((last, prefix)) = score_prob.split_last_mut() {
            *last = 1.0 - prefix.iter().sum::<f64>();
        }
        if score_prob.iter().any(|&value| value < 0.0) {
            return None;
        }
        result.push(EapSumTable {
            dim,
            n_items_dim: end - start - 1,
            score_prob,
            eap,
            sd,
        });
    }
    Some(result)
}

pub(crate) fn score_eapsum_gpu(inputs: &GpuEapSumLookupInputs<'_>) -> Option<(Vec<f64>, Vec<f64>)> {
    let output_len = inputs.n_persons.checked_mul(inputs.n_dims)?;
    if output_len == 0 {
        return None;
    }
    let mut table_offsets = vec![0_u32; inputs.n_dims + 1];
    let mut table_eap = Vec::new();
    let mut table_sd = Vec::new();
    let mut by_dim: Vec<Option<&EapSumTable>> = (0..inputs.n_dims).map(|_| None).collect();
    for table in inputs.tables {
        by_dim[table.dim] = Some(table);
    }
    for dim in 0..inputs.n_dims {
        let table = by_dim[dim]?;
        table_eap.extend_from_slice(&table.eap);
        table_sd.extend_from_slice(&table.sd);
        table_offsets[dim + 1] = u32::try_from(table_eap.len()).ok()?;
    }
    let responses = checked_f32(inputs.y)?;
    let table_eap_f32 = checked_f32(&table_eap)?;
    let table_sd_f32 = checked_f32(&table_sd)?;
    let factor_id: Vec<u32> = inputs
        .factor_id
        .iter()
        .map(|&dim| u32::try_from(dim).ok())
        .collect::<Option<_>>()?;
    let context = GpuContext::get()?;
    let limits = context.device.limits();
    let workgroups = u32::try_from(output_len).ok()?.div_ceil(WORKGROUP_SIZE);
    if workgroups > limits.max_compute_workgroups_per_dimension
        || [
            responses.len(),
            factor_id.len(),
            table_offsets.len(),
            table_eap_f32.len(),
            table_sd_f32.len(),
            output_len,
        ]
        .into_iter()
        .any(|len| !buffer_fits(&limits, len))
    {
        return None;
    }
    let device = &context.device;
    let queue = &context.queue;
    let uniforms = LookupUniforms {
        n_persons: u32::try_from(inputs.n_persons).ok()?,
        n_items: u32::try_from(inputs.n_items).ok()?,
        n_dims: u32::try_from(inputs.n_dims).ok()?,
        _pad0: 0,
    };
    let uniform_buffer = storage(
        device,
        bytemuck::bytes_of(&uniforms),
        wgpu::BufferUsages::UNIFORM,
    );
    let responses = storage_f32(device, &responses);
    let factor_id = storage_u32(device, &factor_id);
    let table_offsets = storage_u32(device, &table_offsets);
    let table_eap = storage_f32(device, &table_eap_f32);
    let table_sd = storage_f32(device, &table_sd_f32);
    let theta_out = output(device, output_len, "eapsum-theta");
    let sd_out = output(device, output_len, "eapsum-theta-sd");
    let buffers = [
        (0, &uniform_buffer),
        (1, &responses),
        (2, &factor_id),
        (3, &table_offsets),
        (4, &table_eap),
        (5, &table_sd),
        (6, &theta_out),
        (7, &sd_out),
    ];
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlsirm-eapsum-lookup-bind-group"),
        layout: &context.lookup_layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&context.lookup_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let theta = read_f32(device, queue, &theta_out, output_len)?;
    let sd = read_f32(device, queue, &sd_out, output_len)?;
    if theta.iter().any(|value| !value.is_finite())
        || sd.iter().any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }
    Some((theta, sd))
}
