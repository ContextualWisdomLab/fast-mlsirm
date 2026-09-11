"""Repository-owned Linux workflows must use stable hosted-runner contracts.

A floating ``ubuntu-latest`` selector has repeatedly remained pre-checkout with
``runner_id=0`` while explicit ``ubuntu-24.04`` jobs in sibling repositories
have acquired GitHub-hosted runners. GPU evidence also must not depend on a live
Ubuntu mirror merely to obtain a software Vulkan adapter: the hosted image
already carries Chrome's SwiftShader Vulkan driver, so repository workflows use
that local driver and fail closed if the image contract disappears.
"""

from pathlib import Path
import os
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
PR_WORKFLOWS = (
    WORKFLOW_DIRECTORY / "ci.yml",
    WORKFLOW_DIRECTORY / "codeql.yml",
)
QUEUE_SENSITIVE_PR_WORKFLOWS = (WORKFLOW_DIRECTORY / "codeql.yml",)
GPU_EVIDENCE_WORKFLOWS = (
    WORKFLOW_DIRECTORY / "ci.yml",
    WORKFLOW_DIRECTORY / "statistical-studies.yml",
)
SOFTWARE_VULKAN_SETUP = REPOSITORY_ROOT / "scripts" / "configure_software_vulkan.sh"
PR_LIFECYCLE_TYPES = (
    "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, closed]"
)
FLOATING_RUNNER_ASSIGNMENT = re.compile(
    r"(?m)^\s*runs-on:\s*ubuntu-latest\s*(?:#.*)?$"
)


def _workflow_paths(directory: Path) -> tuple[Path, ...]:
    """Return every GitHub workflow regardless of the accepted YAML extension."""
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".yml", ".yaml"}
        )
    )


def test_required_pr_workflows_use_explicit_ubuntu_2404() -> None:
    """Require every Linux runner in the repository-owned PR gates to be pinned."""
    for workflow in PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert FLOATING_RUNNER_ASSIGNMENT.search(source) is None, workflow
        assert "runs-on: ubuntu-24.04" in source, workflow


def test_workflow_inventory_includes_yml_and_yaml(tmp_path: Path) -> None:
    """A workflow cannot evade runner policy by choosing the other YAML suffix."""
    yml = tmp_path / "first.yml"
    yaml = tmp_path / "second.yaml"
    ignored = tmp_path / "README.md"
    for path in (yml, yaml, ignored):
        path.write_text("name: fixture\n", encoding="utf-8")

    assert _workflow_paths(tmp_path) == (yml, yaml)


def test_repository_workflows_do_not_float_ubuntu_runner_identity() -> None:
    """Reject only active floating runner assignments, not harmless prose/data."""
    workflow_paths = _workflow_paths(WORKFLOW_DIRECTORY)
    assert workflow_paths, "repository workflow inventory must not be empty"

    offenders = [
        workflow.relative_to(REPOSITORY_ROOT).as_posix()
        for workflow in workflow_paths
        if FLOATING_RUNNER_ASSIGNMENT.search(workflow.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"floating Ubuntu runner selectors remain: {offenders}"


def test_queue_sensitive_pr_workflows_cancel_predecessor_heads() -> None:
    """A superseded PR head must not retain scarce hosted-runner queue capacity."""
    expected_group = (
        "group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}"
    )
    for workflow in QUEUE_SENSITIVE_PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "\nconcurrency:\n" in source, workflow
        concurrency = source.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert expected_group in concurrency, workflow
        assert "cancel-in-progress: true" in concurrency, workflow


def test_queue_sensitive_pr_workflows_reacquire_after_draft_lifecycle() -> None:
    """Ready transitions must create fresh evidence without a no-op source commit."""
    inactive_guard = (
        "if: ${{ github.event_name != 'pull_request' || "
        "(!github.event.pull_request.draft && github.event.action != 'closed') }}"
    )
    for workflow in QUEUE_SENSITIVE_PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        pull_request = source.split("\n  pull_request:\n", 1)[1].split(
            "\n  workflow_dispatch:\n", 1
        )[0]
        assert PR_LIFECYCLE_TYPES in pull_request, workflow
        analyze_actions = source.split("\n  analyze-actions:\n", 1)[1].split(
            "\n  analyze-python:\n", 1
        )[0]
        assert inactive_guard in analyze_actions, workflow


def test_required_python_context_cannot_pass_without_gpu_parity() -> None:
    """The protected ``python`` context must fail when explicit GPU parity fails."""
    source = (WORKFLOW_DIRECTORY / "ci.yml").read_text(encoding="utf-8")
    python_job = source.split("\n  python:\n", 1)[1].split("\n\n  rust:\n", 1)[0]

    assert "needs: [python-matrix, gpu-smoke]" in python_job
    assert 'test "${{ needs.python-matrix.result }}" = "success"' in python_job
    assert 'test "${{ needs.gpu-smoke.result }}" = "success"' in python_job


def test_gpu_evidence_uses_local_swiftshader_without_live_apt() -> None:
    """GPU proof must not fail before execution because an Ubuntu mirror stalls."""
    assert SOFTWARE_VULKAN_SETUP.is_file()
    for workflow in GPU_EVIDENCE_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "bash scripts/configure_software_vulkan.sh" in source, workflow
        assert "sudo apt-get" not in source, workflow
        assert "VK_ICD_FILENAMES" not in source, workflow


def test_software_vulkan_setup_writes_modern_loader_environment(tmp_path: Path) -> None:
    """The setup helper must bind the loader to the image-local SwiftShader ICD."""
    chrome_root = tmp_path / "chrome"
    chrome_root.mkdir()
    (chrome_root / "libvk_swiftshader.so").write_bytes(b"driver")
    (chrome_root / "libvulkan.so.1").write_bytes(b"loader")
    (chrome_root / "vk_swiftshader_icd.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"./libvk_swiftshader.so","api_version":"1.0.5"}}',
        encoding="utf-8",
    )
    github_env = tmp_path / "github-env"
    runtime_dir = tmp_path / "runtime"
    env = os.environ | {
        "FAST_MLSIRM_SWIFTSHADER_ROOT": str(chrome_root),
        "GITHUB_ENV": str(github_env),
        "XDG_RUNTIME_DIR": str(runtime_dir),
    }

    result = subprocess.run(
        ["bash", str(SOFTWARE_VULKAN_SETUP)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    exported = github_env.read_text(encoding="utf-8")
    assert f"VK_DRIVER_FILES={chrome_root / 'vk_swiftshader_icd.json'}\n" in exported
    assert f"LD_LIBRARY_PATH={chrome_root}" in exported
    assert "VK_ICD_FILENAMES" not in exported
    assert runtime_dir.is_dir()


def test_software_vulkan_setup_rejects_manifest_driver_escape(tmp_path: Path) -> None:
    """A mutable manifest may not redirect the trusted runner path to another driver."""
    chrome_root = tmp_path / "chrome"
    chrome_root.mkdir()
    (chrome_root / "libvk_swiftshader.so").write_bytes(b"driver")
    (chrome_root / "libvulkan.so.1").write_bytes(b"loader")
    escaped_driver = tmp_path / "outside.so"
    escaped_driver.write_bytes(b"other-driver")
    (chrome_root / "vk_swiftshader_icd.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"../outside.so","api_version":"1.0.5"}}',
        encoding="utf-8",
    )
    env = os.environ | {
        "FAST_MLSIRM_SWIFTSHADER_ROOT": str(chrome_root),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
    }

    result = subprocess.run(
        ["bash", str(SOFTWARE_VULKAN_SETUP)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SwiftShader manifest driver mismatch" in result.stderr
