"""Repository-owned CI contracts for deterministic GPU smoke provisioning."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).parents[1]
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_SOFTWARE_VULKAN_HELPER = _ROOT / "scripts" / "configure_software_vulkan.sh"


def _gpu_job() -> str:
    """Return only the ``gpu-smoke`` job from the CI workflow."""
    text = _CI_WORKFLOW.read_text(encoding="utf-8")
    start_marker = "  gpu-smoke:\n"
    end_marker = "\n  fuzz:\n"
    assert start_marker in text, "gpu-smoke job is missing"
    _, remainder = text.split(start_marker, 1)
    assert end_marker in remainder, "gpu-smoke job boundary is missing"
    job, _ = remainder.split(end_marker, 1)
    return job


def test_gpu_smoke_uses_repository_owned_image_local_vulkan_helper() -> None:
    """Required GPU evidence must not depend on a live package mirror."""
    job = _gpu_job()
    assert "bash scripts/configure_software_vulkan.sh" in job
    assert "sudo apt-get" not in job
    assert "mesa-vulkan-drivers" not in job
    assert "vulkan-tools" not in job


def test_software_vulkan_helper_fails_closed_and_records_exact_identity() -> None:
    """The helper must bind the hosted image manifest to readable driver bytes."""
    script = _SOFTWARE_VULKAN_HELPER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "vk_swiftshader_icd.json" in script
    assert "libvk_swiftshader.so" in script
    assert "libvulkan.so.1" in script
    assert "resolve(strict=True)" in script
    assert "resolved_driver != expected_driver" in script
    assert 'sha256sum "$icd_manifest" "$swiftshader_driver" "$vulkan_loader"' in script
    assert "sudo" not in script
    assert "apt-get" not in script


def test_software_vulkan_helper_exports_bounded_wgpu_evidence_environment() -> None:
    """Only the controlled evidence lane may opt into bundled SwiftShader."""
    script = _SOFTWARE_VULKAN_HELPER.read_text(encoding="utf-8")
    assert "GITHUB_ENV must be provided by GitHub Actions" in script
    assert "VK_DRIVER_FILES=%s" in script
    assert "XDG_RUNTIME_DIR=%s" in script
    assert "FAST_MLSIRM_ALLOW_NONCOMPLIANT_SOFTWARE_VULKAN=1" in script
    assert "LD_LIBRARY_PATH=%s" in script


def test_gpu_smoke_requires_real_parity_execution_without_skip() -> None:
    """CPU fallback must never be accepted as successful GPU evidence."""
    job = _gpu_job()
    assert "tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely" in job
    assert "--junitxml=gpu-junit.xml" in job
    assert 'skips = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)' in job
    assert "GPU evidence contained {skips} skipped test(s)" in job
