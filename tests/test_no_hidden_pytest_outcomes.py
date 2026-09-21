"""Fail-closed policy tests for hidden pytest outcomes.

These tests prove that any pytest invocation ending with a skip,
import-or-skip, skipif, xfail, or xpass outcome exits non-zero, while a
clean all-executed invocation keeps its success status. Each case runs a
REAL child pytest subprocess in an isolated ``tmp_path`` session dir, so
this file itself never uses skip/xfail marks.

Implementation basis: fail-closed outcome policy of Issue #1732
(ContextualWisdomLab, 2026).

References:
    ContextualWisdomLab. (2026). Test governance: fail closed on skipped
    and expected-failure outcomes (Issue #1732).
    https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_TESTS_DIR = Path(__file__).resolve().parent
ALLOWLIST_NAME = "allowed_non_execution.txt"
POLICY_MODULE_NAME = "_policy_under_test.py"


def _install_enforcement(dest: Path, mode: str) -> None:
    """Install the enforcement plugin (or a reporter-less stub) into a child session dir.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    source = REPO_TESTS_DIR / "conftest.py"
    allowlist = REPO_TESTS_DIR / ALLOWLIST_NAME
    if mode == "enforced":
        # Copy the real session-level plugin (and its allowlist) when they
        # exist. Before the GREEN implementation lands, the copy is a no-op
        # and the child runs bare, which is exactly the RED demonstration.
        if source.exists():
            shutil.copy(source, dest / "conftest.py")
        if allowlist.exists():
            shutil.copy(allowlist, dest / ALLOWLIST_NAME)
    elif mode == "missing-reporter":
        if source.exists():
            # Simulate accounting that can never be observed: register ONLY
            # the real session-finish verdict hook, without the configure
            # marker or the collection/run reporters that initialize it.
            shutil.copy(source, dest / POLICY_MODULE_NAME)
            stub = textwrap.dedent(
                """\
                from _policy_under_test import pytest_sessionfinish  # noqa: F401
                """
            )
            (dest / "conftest.py").write_text(stub, encoding="utf-8")
        else:
            # Pre-GREEN: no accounting exists at all; a bare conftest makes
            # the defect visible (the child stays GREEN despite no reporter).
            (dest / "conftest.py").write_text("", encoding="utf-8")
    else:  # pragma: no cover - defensive against helper misuse
        raise ValueError(f"unknown conftest mode: {mode!r}")


def _run_child(dest: Path, files: dict[str, str], mode: str) -> subprocess.CompletedProcess[str]:
    """Run a real isolated child pytest session and return its completed process.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    _install_enforcement(dest, mode)
    for name, body in files.items():
        (dest / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=dest,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_clean_all_executed_keeps_success(tmp_path: Path) -> None:
    """Assert that a child session with every test executed exits zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {"test_clean.py": "def test_executes():\n    assert 1 + 1 == 2\n"},
        mode="enforced",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


def test_call_time_skip_exits_nonzero(tmp_path: Path) -> None:
    """Assert that a child session with a call-time skip exits non-zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {
            "test_skippy.py": (
                "import pytest\n"
                "def test_runs():\n"
                "    assert True\n"
                "def test_skipped():\n"
                "    pytest.skip('injected capability skip')\n"
            )
        },
        mode="enforced",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_collection_time_skip_exits_nonzero(tmp_path: Path) -> None:
    """Assert that a child session with a collection-time skip exits non-zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {
            "test_collected.py": (
                "def test_executes():\n"
                "    assert 1 + 1 == 2\n"
            ),
            "test_skipped_module.py": (
                "import pytest\n"
                "pytest.skip('injected collection skip', allow_module_level=True)\n"
                "def test_never_collected():\n"
                "    assert True\n"
            ),
        },
        mode="enforced",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_xfail_exits_nonzero(tmp_path: Path) -> None:
    """Assert that a child session with an xfail outcome exits non-zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {
            "test_expected_fail.py": (
                "import pytest\n"
                "@pytest.mark.xfail(reason='injected expected failure')\n"
                "def test_fails_as_expected():\n"
                "    assert False\n"
            )
        },
        mode="enforced",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_xpass_exits_nonzero(tmp_path: Path) -> None:
    """Assert that a child session with a non-strict xpass outcome exits non-zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {
            "test_unexpected_pass.py": (
                "import pytest\n"
                "@pytest.mark.xfail(reason='injected unexpected pass', strict=False)\n"
                "def test_passes_unexpectedly():\n"
                "    assert True\n"
            )
        },
        mode="enforced",
    )
    assert "xpass" in (proc.stdout + proc.stderr).lower()
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_missing_reporter_fails_closed(tmp_path: Path) -> None:
    """Assert that a session without observable accounting fails closed.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(
        tmp_path,
        {"test_clean.py": "def test_executes():\n    assert 1 + 1 == 2\n"},
        mode="missing-reporter",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_zero_collected_stays_nonzero(tmp_path: Path) -> None:
    """Assert that a child session collecting zero tests stays non-zero.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    proc = _run_child(tmp_path, {}, mode="enforced")
    assert proc.returncode != 0, proc.stdout + proc.stderr


def _workflow_job_body(workflow: str, job_name: str) -> str:
    """Return one top-level GitHub Actions job without borrowing sibling evidence."""
    lines = workflow.splitlines()
    header = f"  {job_name}:"
    try:
        start_index = lines.index(header)
    except ValueError as exc:
        raise AssertionError(f"missing workflow job {job_name!r}") from exc

    job_lines = [lines[start_index]]
    for line in lines[start_index + 1 :]:
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            break
        job_lines.append(line)
    return "\n".join(job_lines)


def test_allowlisted_capability_nodes_have_exact_ci_owners() -> None:
    """Require exact non-execution entries and evidence in their owning CI jobs."""
    allowlist = (REPO_TESTS_DIR / ALLOWLIST_NAME).read_text(encoding="utf-8")
    ci_workflow = (
        REPO_TESTS_DIR.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    gpu_smoke_job = _workflow_job_body(ci_workflow, "gpu-smoke")
    fuzz_job = _workflow_job_body(ci_workflow, "fuzz")
    gpu_owned_nodes = (
        "tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely",
        "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q121",
        "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q241",
        "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q481",
        "tests/test_bifactor_gpu_high_q.py::"
        "test_bifactor_gpu_parity_q241_wide_items_metal_workgroups",
        "tests/test_bifactor_gpu_high_q.py::test_bifactor_cpu_q121_vs_q241_agree",
        "tests/test_bifactor_bootstrap_benchmark.py::"
        "test_joint_bootstrap_cpu_vs_gpu_wall_time_q121",
    )

    assert "tests/test_bifactor_gpu_high_q.py::*" not in allowlist
    assert "tests/test_fuzz_properties.py #" not in allowlist
    for node in gpu_owned_nodes:
        assert node in allowlist
        assert node in gpu_smoke_job
    assert "pytest tests/test_fuzz_properties.py" in fuzz_job
    assert 'ElementTree.parse(Path("gpu-junit.xml"))' in gpu_smoke_job
    assert 'ElementTree.parse(Path("fuzz-properties-junit.xml"))' in fuzz_job
