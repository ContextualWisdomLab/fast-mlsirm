use wgpu::{Backends, InstanceDescriptor, InstanceFlags, RequestAdapterOptions};

const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 8;
const REQUIRED_UNIFORM_BUFFERS_PER_STAGE: u32 = 1;
const REQUIRED_BUFFERS_PER_STAGE: u32 =
    REQUIRED_STORAGE_BUFFERS_PER_STAGE + REQUIRED_UNIFORM_BUFFERS_PER_STAGE;
const REQUIRED_WORKGROUP_SIZE_X: u32 = 64;

fn main() {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::PRIMARY;
    descriptor.flags = InstanceFlags::default();
    descriptor
        .flags
        .insert(InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);

    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("controlled software-Vulkan evidence must expose a wgpu adapter");
    let info = adapter.get_info();
    let limits = adapter.limits();

    eprintln!(
        "gpu adapter name={:?} backend={:?} device_type={:?} storage_buffers_per_stage={} uniform_buffers_per_stage={} buffers_and_acceleration_structures_per_stage={} storage_buffer_binding_size={} max_buffer_size={} storage_buffer_offset_alignment={} workgroup_size_x={} invocations_per_workgroup={}",
        info.name,
        info.backend,
        info.device_type,
        limits.max_storage_buffers_per_shader_stage,
        limits.max_uniform_buffers_per_shader_stage,
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.min_storage_buffer_offset_alignment,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_invocations_per_workgroup,
    );

    assert!(
        limits.max_storage_buffers_per_shader_stage >= REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        "packed marginal GPU requires at least {REQUIRED_STORAGE_BUFFERS_PER_STAGE} storage buffers per shader stage; adapter {:?} exposes {}",
        info.name,
        limits.max_storage_buffers_per_shader_stage,
    );
    assert!(
        limits.max_uniform_buffers_per_shader_stage >= REQUIRED_UNIFORM_BUFFERS_PER_STAGE,
        "packed marginal GPU requires at least {REQUIRED_UNIFORM_BUFFERS_PER_STAGE} uniform buffer per shader stage; adapter {:?} exposes {}",
        info.name,
        limits.max_uniform_buffers_per_shader_stage,
    );
    assert!(
        limits.max_buffers_and_acceleration_structures_per_shader_stage >= REQUIRED_BUFFERS_PER_STAGE,
        "packed marginal GPU requires at least {REQUIRED_BUFFERS_PER_STAGE} combined buffer/acceleration-structure bindings per shader stage; adapter {:?} exposes {}",
        info.name,
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
    );
    assert!(
        limits.max_compute_workgroup_size_x >= REQUIRED_WORKGROUP_SIZE_X,
        "marginal GPU requires max_compute_workgroup_size_x >= {REQUIRED_WORKGROUP_SIZE_X}; adapter {:?} exposes {}",
        info.name,
        limits.max_compute_workgroup_size_x,
    );
    assert!(
        limits.max_compute_invocations_per_workgroup >= REQUIRED_WORKGROUP_SIZE_X,
        "marginal GPU requires max_compute_invocations_per_workgroup >= {REQUIRED_WORKGROUP_SIZE_X}; adapter {:?} exposes {}",
        info.name,
        limits.max_compute_invocations_per_workgroup,
    );
}
