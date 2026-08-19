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


def _apt_commands() -> tuple[tuple[str, ...], ...]:
    """Return tokenized apt commands from only the GPU provisioning step."""
    script = _logical_shell(_gpu_install_script())
    return tuple(
        tuple(line.split())
        for line in script.splitlines()
        if "sudo apt-get" in line
    )


def test_gpu_smoke_apt_network_work_has_hard_deadlines() -> None:
    """Each apt operation must retain its exact bounded network contract."""
    update = (
        "timeout",
        "120s",
        "sudo",
        "apt-get",
        "-o",
        "Acquire::Retries=2",
        "-o",
        "Acquire::http::Timeout=10",
        "-o",
        "Acquire::https::Timeout=10",
        "-o",
        "DPkg::Lock::Timeout=30",
        "update",
    )
    install = (
        "timeout",
        "180s",
        "sudo",
        "apt-get",
        "-o",
        "Acquire::Retries=2",
        "-o",
        "Acquire::http::Timeout=10",
        "-o",
        "Acquire::https::Timeout=10",
        "-o",
        "DPkg::Lock::Timeout=30",
        "install",
        "--yes",
        "mesa-vulkan-drivers",
        "vulkan-tools",
    )

    assert _apt_commands() == (update, install)


def test_gpu_smoke_apt_lock_wait_is_bounded() -> None:
    """Both package-manager operations keep the exact 30-second lock bound."""
    for command in _apt_commands():
        assert "DPkg::Lock::Timeout=30" in command
