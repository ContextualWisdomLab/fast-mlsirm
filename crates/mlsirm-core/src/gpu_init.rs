//! Shared wgpu instance construction for all GPGPU modules.
//!
//! Restricted sandboxes often expose a broken `/dev/dri` node that makes the
//! GL/EGL backend log `libEGL warning: failed to open /dev/dri/...` and then
//! SIGSEGV inside the native stack. That bypasses Rust's `Result` fallback and
//! kills the process before CPU paths can run.
//!
//! We therefore build the instance with [`wgpu::Backends::PRIMARY`] only
//! (Vulkan / Metal / DX12 / WebGPU) — never GL — and treat missing adapters as
//! a soft `None` so callers fall back to the f64 CPU reference.

/// Construct a wgpu instance that avoids the GL/EGL backend.
pub(crate) fn new_instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    // PRIMARY = Vulkan | Metal | DX12 | BrowserWebGPU — never GL/EGL.
    desc.backends = wgpu::Backends::PRIMARY;
    wgpu::Instance::new(desc)
}
