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
//!
//! Required CI evidence may intentionally exercise the image-bundled Chrome
//! SwiftShader Vulkan driver. wgpu hides underlying drivers whose Vulkan
//! compliance version is not trustworthy by default, so that evidence path
//! must opt in explicitly through the fast-mlsirm-owned environment boundary;
//! production callers keep the compliant-adapter default unless they make the
//! same explicit choice.

const SOFTWARE_VULKAN_OPT_IN: &str = "FAST_MLSIRM_ALLOW_NONCOMPLIANT_SOFTWARE_VULKAN";

/// Return whether the process explicitly opted into non-compliant software Vulkan evidence.
fn software_vulkan_evidence_opt_in() -> bool {
    std::env::var(SOFTWARE_VULKAN_OPT_IN).as_deref() == Ok("1")
}

/// Build instance flags while preserving wgpu's production-safe defaults.
fn instance_flags(allow_noncompliant_software_vulkan: bool) -> wgpu::InstanceFlags {
    let mut flags = wgpu::InstanceFlags::default();
    if allow_noncompliant_software_vulkan {
        flags.insert(wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER);
    }
    flags
}

/// Construct a wgpu instance that avoids the GL/EGL backend.
pub(crate) fn new_instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    // PRIMARY = Vulkan | Metal | DX12 | BrowserWebGPU — never GL/EGL.
    desc.backends = wgpu::Backends::PRIMARY;
    desc.flags = instance_flags(software_vulkan_evidence_opt_in());
    wgpu::Instance::new(desc)
}

#[cfg(test)]
mod tests {
    use super::instance_flags;

    #[test]
    fn software_vulkan_noncompliance_is_opt_in_only() {
        let default_flags = instance_flags(false);
        let evidence_flags = instance_flags(true);

        assert!(!default_flags.contains(
            wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
        ));
        assert!(evidence_flags.contains(
            wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
        ));
    }
}
