"""Repository-owned CI contracts for deterministic GPU smoke provisioning."""

from __future__ import annotations

from pathlib import Path
import re


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _gpu_install_script() -> str:
    """Return the software-Vulkan provisioning shell from the CI workflow."""
    text = _CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"- name: Install software Vulkan adapter\n"
        r"\s+run: \|\n(?P<script>(?:\s{8,}.*\n)+?)"
        r"\s+- name: Prove Vulkan compute adapter availability",
        text,
    )
    assert match is not None, "gpu-smoke Vulkan provisioning step is missing"
    return match.group("script")


def _logical_shell(script: str) -> str:
    """Join explicit shell continuations so command contracts stay readable."""
    return script.replace("\\\n", " ")


def test_gpu_smoke_apt_network_work_has_hard_deadlines() -> None:
    """Runner mirror stalls must fail boundedly before the job-level timeout."""
    script = _logical_shell(_gpu_install_script())

    assert re.search(r"timeout\s+\d+s\s+sudo\s+apt-get\b.*\bupdate\b", script)
    assert re.search(r"timeout\s+\d+s\s+sudo\s+apt-get\b.*\binstall\b", script)
    assert "Acquire::http::Timeout=" in script
    assert "Acquire::https::Timeout=" in script
    assert "Acquire::Retries=" in script


def test_gpu_smoke_apt_lock_wait_is_bounded() -> None:
    """Package-manager lock contention must not consume the full GPU job budget."""
    script = _gpu_install_script()

    assert "DPkg::Lock::Timeout=" in script
