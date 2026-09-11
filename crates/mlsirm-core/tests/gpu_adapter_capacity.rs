#![cfg(feature = "gpu")]

use wgpu::{Backends, InstanceDescriptor, InstanceFlags, RequestAdapterOptions};

const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 18;
const REQUIRED_WORKGROUP_SIZE_X: u32 = 64;

#[test]
fn software_vulkan_adapter_meets_marginal_capacity_contract() {
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
        "gpu adapter name={:?} backend={:?} device_type={:?} storage_buffers_per_stage={} workgroup_size_x={} invocations_per_workgroup={}",
        info.name,
        info.backend,
        info.device_type,
        limits.max_storage_buffers_per_shader_stage,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_invocations_per_workgroup,
    );

    assert!(
        limits.max_storage_buffers_per_shader_stage >= REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        "marginal GPU requires at least {REQUIRED_STORAGE_BUFFERS_PER_STAGE} storage buffers per shader stage; adapter {:?} exposes {}",
        info.name,
        limits.max_storage_buffers_per_shader_stage,
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
