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

import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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
    normalized = _plain_workflow_mapping(workflow)
    structural = _plain_workflow_mapping(workflow, structural_only=True)
    if normalized is None or structural is None:
        raise AssertionError("unsupported workflow mapping")
    lines = normalized.splitlines()
    structure = structural.splitlines()
    jobs = [i for i, line in enumerate(structure) if line == "jobs:"]
    if len(jobs) != 1:
        raise AssertionError("missing or duplicate jobs mapping")
    start = jobs[0] + 1
    end = next((i for i in range(start, len(structure))
                if structure[i].strip() and not structure[i].lstrip().startswith("#")
                and not structure[i].startswith(" ")), len(structure))
    candidates = [i for i in range(start, end) if structure[i] == f"  {job_name}:"]
    if len(candidates) != 1:
        raise AssertionError(f"missing or duplicate workflow job {job_name!r}")
    start_index = candidates[0]
    stop = next((i for i in range(start_index + 1, end)
                 if structure[i].strip() and not structure[i].lstrip().startswith("#")
                 and len(structure[i]) - len(structure[i].lstrip()) <= 2), end)
    return "\n".join(lines[start_index:stop])


# Test modules whose skips are capability/opt-in gates evidenced by a
# dedicated CI job; any allowlist entry for them must name that job.
CAPABILITY_MODULES = (
    "tests/test_bifactor_gpu_high_q.py",
    "tests/test_bifactor_bootstrap_benchmark.py",
    "tests/test_marginal_parity.py",
)
_OWNER_MARKER = "owned by "


def _default_run_setting(source: str, indent: int, setting: str) -> str | None:
    """Read one plain defaults.run setting; reject other mapping syntax.

    Basis: GitHub (n.d.), Workflow syntax for GitHub Actions, defaults.run
    and jobs.<job_id>.defaults.run sections. Job defaults override workflow
    defaults. This restricted static parser is not a runtime execution proof.

    References:
        GitHub. (n.d.). Workflow syntax for GitHub Actions. GitHub Docs.
        https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
    """
    structural = _plain_workflow_mapping(source, structural_only=True)
    if structural is None:
        return ""
    lines = structural.splitlines()
    for key, depth in (("defaults", indent), ("run", indent + 2)):
        entries = [i for i, line in enumerate(lines)
                   if re.match(rf"^{' ' * depth}{key}:", line)]
        if not entries:
            return None
        if len(entries) != 1:
            return ""
        start = entries[0]
        if lines[start].split(":", 1)[1].strip():
            return ""
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].strip() and not lines[i].lstrip().startswith("#")
                    and len(lines[i]) - len(lines[i].lstrip()) <= depth), len(lines))
        lines = lines[start + 1:end]
    shells = [i for i, line in enumerate(lines)
              if re.match(rf"^{' ' * (indent + 4)}{re.escape(setting)}:", line)]
    if not shells:
        return None
    if len(shells) != 1:
        return ""
    index = shells[0]
    continuation = next((line for line in lines[index + 1:] if line.strip()
                         and not line.lstrip().startswith("#")), "")
    if continuation and len(continuation) - len(continuation.lstrip()) > indent + 4:
        return ""
    return lines[index].split(":", 1)[1].strip()


def _plain_workflow_mapping(source: str, *, structural_only: bool = False) -> str | None:
    """Normalize literal mapping keys, rejecting unsupported structural syntax.

    Basis: YAML Language Development Team (2021), YAML 1.2.2, Sections
    6.9 and 8.2.2. Node properties and explicit keys are not plain implicit
    mappings. Scalar block contents are preserved and never parsed as keys.

    References:
        YAML Language Development Team. (2021). YAML Ain't Markup Language
        (YAML) version 1.2.2. https://yaml.org/spec/1.2.2/
    """
    normalized: list[str] = []
    scalar_indent: int | None = None
    for line in source.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped or stripped.startswith("#"):
            normalized.append(line)
            continue
        if scalar_indent is not None:
            if indent > scalar_indent:
                normalized.append(" " if structural_only else line)
                continue
            scalar_indent = None
        match = re.fullmatch(
            r"([ ]*(?:- )?)(['\"]?)([A-Za-z_][A-Za-z0-9_.-]*)\2[ ]*:(.*)",
            line,
        )
        if match is None:
            return None
        prefix, _, key, value = match.groups()
        normalized.append(f"{prefix}{key}:{value}")
        if re.fullmatch(r"[ ]*[|>][+-]?[ ]*(?:#.*)?", value):
            scalar_indent = indent + (2 if stripped.startswith("- ") else 0)
    return "\n".join(normalized)


def _pytest_owner_nodes(body: str, workflow: str = "") -> set[str]:
    """Recognize only the workflow's plain run scalars and direct pytest grammar.

    This is static command ownership, not proof of runtime or GPU execution.
    Unsupported YAML or shell syntax supplies no evidence.
    """
    normalized_body = _plain_workflow_mapping(body)
    normalized_workflow = _plain_workflow_mapping(workflow)
    if normalized_body is None or normalized_workflow is None:
        return set()
    body, workflow = normalized_body, normalized_workflow
    structure = _plain_workflow_mapping(body, structural_only=True)
    if structure is None:
        return set()
    conditions = [line.split(":", 1)[1].strip() for line in structure.splitlines()
                  if line.startswith("    if:")]
    if len(conditions) > 1:
        return set()
    if conditions:
        condition = re.split(r"[ \t]+#", conditions[0], maxsplit=1)[0].strip()
        event_admission = (
            "${{ github.event_name != 'pull_request' || "
            "(!github.event.pull_request.draft && github.event.action != 'closed') }}"
        )
        if condition not in {"true", "True", "TRUE", "!!bool true", "${{ true }}", event_admission}:
            return set()
    structure_lines = structure.splitlines()
    body_lines = body.splitlines()
    starts = [i for i, line in enumerate(structure_lines) if line == "    steps:"]
    if len(starts) != 1:
        return set()
    steps_start = starts[0] + 1
    steps_end = next((i for i in range(steps_start, len(structure_lines))
                      if structure_lines[i].strip()
                      and not structure_lines[i].lstrip().startswith("#")
                      and len(structure_lines[i]) - len(structure_lines[i].lstrip()) <= 4),
                     len(structure_lines))
    inherited_shell = _default_run_setting(body, 4, "shell")
    if inherited_shell is None:
        inherited_shell = _default_run_setting(workflow, 0, "shell")
    inherited_directory = _default_run_setting(body, 4, "working-directory")
    if inherited_directory is None:
        inherited_directory = _default_run_setting(workflow, 0, "working-directory")
    nodes: set[str] = set()
    step_starts = [i for i in range(steps_start, steps_end)
                   if structure_lines[i].startswith("      - ")]
    for position, start in enumerate(step_starts):
        end = step_starts[position + 1] if position + 1 < len(step_starts) else steps_end
        lines = body_lines[start:end]
        metadata = structure_lines[start:end]
        # Conditional steps are not unconditional owners. Job-level event
        # admission remains a separate runtime contract.
        if any(re.match(r"^(?:      - |        )if:", line) for line in metadata):
            continue
        shells = [i for i, line in enumerate(metadata)
                  if re.match(r"^(?:      - |        )shell:", line)]
        if len(shells) > 1:
            continue
        effective_shell = inherited_shell
        if shells:
            shell_index = shells[0]
            effective_shell = lines[shell_index].split("shell:", 1)[1].strip()
            continuation = next((line for line in lines[shell_index + 1:]
                                 if line.strip() and not line.lstrip().startswith("#")), "")
            if continuation.startswith("          "):
                continue
        if effective_shell is not None and effective_shell not in {"bash", "sh"}:
            continue
        directories = [i for i, line in enumerate(metadata)
                       if re.match(r"^(?:      - |        )working-directory:", line)]
        if len(directories) > 1:
            continue
        effective_directory = inherited_directory
        if directories:
            directory_index = directories[0]
            effective_directory = lines[directory_index].split("working-directory:", 1)[1].strip()
            continuation = next((line for line in lines[directory_index + 1:]
                                 if line.strip() and not line.lstrip().startswith("#")), "")
            if continuation.startswith("          "):
                continue
        if effective_directory is not None and effective_directory not in {".", "./"}:
            continue
        runs = [i for i, line in enumerate(metadata)
                if re.match(r"^(?:      - |        )run: ", line)]
        if len(runs) != 1:
            continue
        index = runs[0]
        value = lines[index].split("run: ", 1)[1]
        if value == "|":
            block = []
            for line in lines[index + 1:]:
                if line.strip() and not line.startswith("          "):
                    break
                block.append(line[10:])
            command = "\n".join(block).strip()
        else:
            continuation = next((line for line in lines[index + 1:] if line.strip()), "")
            if continuation.startswith("          "):
                continue
            command = value
        # A deliberately restricted grammar avoids borrowing selector tokens
        # from comments, other commands, substitutions or option values.
        command = command.replace("\\\n", " ")
        if not re.fullmatch(r"[A-Za-z0-9_./:=\[\] -]+", command):
            continue
        tokens = shlex.split(command)
        if not tokens or tokens[0] != "pytest":
            continue
        positional = []
        for token in tokens[1:]:
            if token == "-q" or (token.startswith("--junitxml=") and token != "--junitxml="):
                continue
            if not token.startswith("tests/"):
                break
            positional.append(token)
        else:
            nodes.update(positional)
    return nodes


def _capability_ownership_violations(allowlist: str, ci_workflow: str) -> list[str]:
    """Return every capability allowlist entry that lacks an executing CI owner.

    An entry is a capability entry when its reason declares ``owned by <job>``,
    names a GPU/``STAGE5_HIGH_Q`` gate, or its node lives in
    ``CAPABILITY_MODULES``. Each must be one exact node (no glob), declare its
    owning job, and be an exact positional selector in its direct pytest run,
    so a newly allowlisted
    node without an executing job fails instead of silently widening the
    waiver.
    """
    violations: list[str] = []
    for raw in allowlist.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        node, _, reason = line.partition(" # ")
        node = node.strip()
        is_capability = (
            _OWNER_MARKER in reason
            or "GPU" in reason
            or "STAGE5_HIGH_Q" in reason
            or node.split("::", 1)[0] in CAPABILITY_MODULES
        )
        if not is_capability:
            continue
        if "*" in node:
            violations.append(f"{node}: capability entry must be one exact node, not a glob")
            continue
        if _OWNER_MARKER not in reason:
            violations.append(f"{node}: capability entry does not declare 'owned by <job>'")
            continue
        job = reason.split(_OWNER_MARKER, 1)[1].split()[0].strip("`.,;()")
        try:
            body = _workflow_job_body(ci_workflow, job)
        except AssertionError:
            violations.append(f"{node}: declared owner job {job!r} does not exist")
            continue
        if node not in _pytest_owner_nodes(body, ci_workflow):
            violations.append(f"{node}: owner job {job!r} lacks a supported pytest command for this exact node")
    return violations


def test_allowlisted_capability_nodes_have_exact_ci_owners() -> None:
    """Require every capability allowlist entry to be executed by its owning CI job."""
    allowlist = (REPO_TESTS_DIR / ALLOWLIST_NAME).read_text(encoding="utf-8")
    ci_workflow = (
        REPO_TESTS_DIR.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    gpu_smoke_job = _workflow_job_body(ci_workflow, "gpu-smoke")
    fuzz_job = _workflow_job_body(ci_workflow, "fuzz")

    assert _capability_ownership_violations(allowlist, ci_workflow) == []
    owned = [
        line for line in allowlist.splitlines()
        if not line.lstrip().startswith("#") and _OWNER_MARKER + "gpu-smoke" in line
    ]
    assert len(owned) == 7, owned
    assert "tests/test_fuzz_properties.py" not in "\n".join(
        line for line in allowlist.splitlines() if not line.lstrip().startswith("#")
    )
    assert "pytest tests/test_fuzz_properties.py" in fuzz_job
    assert 'ElementTree.parse(Path("gpu-junit.xml"))' in gpu_smoke_job
    assert 'ElementTree.parse(Path("fuzz-properties-junit.xml"))' in fuzz_job


def test_capability_ownership_rejects_unowned_or_widened_entries() -> None:
    """A new capability node without an executing owner job must be reported."""
    ci_workflow = (
        REPO_TESTS_DIR.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    allowlist = (REPO_TESTS_DIR / ALLOWLIST_NAME).read_text(encoding="utf-8")
    unowned = "tests/test_bifactor_gpu_high_q.py::test_unowned_future_capability"
    cases = {
        "claims gpu-smoke but not executed": f"{unowned} # GPU adapter owned by gpu-smoke (reviewed 2026-09-22)",
        "no owner declared": f"{unowned} # GPU adapter absent (reviewed 2026-09-22)",
        "module glob": "tests/test_bifactor_gpu_high_q.py::* # GPU owned by gpu-smoke (reviewed 2026-09-22)",
        "missing owner job": f"{unowned} # GPU owned by no-such-job (reviewed 2026-09-22)",
    }
    for label, entry in cases.items():
        violations = _capability_ownership_violations(allowlist + "\n" + entry + "\n", ci_workflow)
        assert violations, f"{label}: injected entry was not rejected"


_CAPABILITY_NODE = "tests/test_bifactor_gpu_high_q.py::test_future_capability"


@pytest.mark.parametrize("step", [
    f"      # {_CAPABILITY_NODE}\n      - run: true",
    f"      - name: {_CAPABILITY_NODE}\n        run: true",
    f"      - run: true\n        env:\n          NODE: {_CAPABILITY_NODE}",
    f"      - run: |\n          # pytest {_CAPABILITY_NODE}\n          true",
    f"      - run: echo pytest {_CAPABILITY_NODE}",
    f"      - run: pytest -k {_CAPABILITY_NODE}",
    f"      - run: pytest {_CAPABILITY_NODE}_other",
    f"      - run: |\n          pytest tests/other.py\n          echo {_CAPABILITY_NODE}",
    f"      - run: |\n          cat <<EOF\n          pytest {_CAPABILITY_NODE}\n          EOF",
    f"      - run: pytest {_CAPABILITY_NODE}; true",
    f"      - run: pytest --collect-only {_CAPABILITY_NODE}",
    f"      - run: pytest {_CAPABILITY_NODE}\n          --collect-only",
    f"      - run: pytest {_CAPABILITY_NODE}\n\n          --collect-only",
    f"      - if: false\n        run: pytest {_CAPABILITY_NODE}",
    f"      - run: pytest {_CAPABILITY_NODE}\n        if: false",
    f"      - run: |\n          pytest tests/other.py # comment \\\n          {_CAPABILITY_NODE}",
])
def test_capability_ownership_rejects_non_command_evidence(step: str) -> None:
    workflow = f"jobs:\n  gpu-smoke:\n    steps:\n{step}\n"
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("step", [
    f"      - run: pytest {_CAPABILITY_NODE}",
    f"      - name: Evidence\n        run: |\n          pytest -q --junitxml=gpu.xml \\\n            {_CAPABILITY_NODE}",
])
def test_capability_ownership_accepts_exact_pytest_arguments(step: str) -> None:
    workflow = f"jobs:\n  gpu-smoke:\n    steps:\n{step}\n"
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


@pytest.mark.parametrize("condition", [
    "false", "0", "'false'", '"false"', "${{ false }}",
    "false # disabled job cannot own evidence",
    " false",
    " ${{ false }}",
    "  false # disabled job cannot own evidence",
    "${{ null }}", "${{ -0 }}", "${{ '' }}", "${{ 0.0 }}",
    "null", "-0", "''", '""', "False", "FALSE", "!!bool false",
    "${{ false && true }}", "${{ github.event.does_not_exist }}",
    "0 # disabled job cannot own evidence",
    "'false' # disabled job cannot own evidence",
    '"false" # disabled job cannot own evidence',
    "${{ false }} # disabled job cannot own evidence",
])
def test_capability_ownership_rejects_disabled_jobs(condition: str) -> None:
    workflow = (
        f"jobs:\n  gpu-smoke:\n    if: {condition}\n    steps:\n"
        f"      - run: pytest {_CAPABILITY_NODE}\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("condition", [
    "true # false", "${{ true }} # false", " true",
    "${{ github.event_name != 'pull_request' || "
    "(!github.event.pull_request.draft && github.event.action != 'closed') }}",
])
def test_capability_ownership_accepts_enabled_job_with_comment(condition: str) -> None:
    workflow = (
        f"jobs:\n  gpu-smoke:\n    if: {condition}\n    steps:\n"
        f"      - run: pytest {_CAPABILITY_NODE}\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


@pytest.mark.parametrize("shell", ["echo {0}", "python"])
@pytest.mark.parametrize("scope", ["step-first", "step-last", "job", "workflow"])
def test_capability_ownership_rejects_unsupported_shell(shell: str, scope: str) -> None:
    defaults = f"defaults:\n  run:\n    shell: {shell}\n"
    prefix = defaults if scope == "workflow" else ""
    job = textwrap.indent(defaults, "    ") if scope == "job" else ""
    if scope == "step-first":
        step = f"      - shell: {shell}\n        run: pytest {_CAPABILITY_NODE}\n"
    else:
        step = f"      - run: pytest {_CAPABILITY_NODE}\n"
        if scope == "step-last":
            step += f"        shell: {shell}\n"
    workflow = prefix + f"jobs:\n  gpu-smoke:\n{job}    steps:\n" + step
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("workflow_shell,job_shell,step_shell", [
    (None, None, "bash"),
    (None, "sh", None),
    ("bash", None, None),
    ("echo {0}", "sh", None),
    ("python", None, "bash"),
    ("bash", "echo {0}", "sh"),
])
def test_capability_ownership_accepts_supported_effective_shell(
    workflow_shell: str | None, job_shell: str | None, step_shell: str | None,
) -> None:
    workflow = (
        f"defaults:\n  run:\n    shell: {workflow_shell}\n" if workflow_shell else ""
    )
    workflow += "jobs:\n  gpu-smoke:\n"
    if job_shell:
        workflow += f"    defaults:\n      run:\n        shell: {job_shell}\n"
    workflow += f"    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
    if step_shell:
        workflow += f"        shell: {step_shell}\n"
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


@pytest.mark.parametrize("directory", ["other", "..", "${{ github.workspace }}", "'other'"])
@pytest.mark.parametrize("scope", ["step-first", "step-last", "job", "workflow"])
def test_capability_ownership_rejects_nonroot_working_directory(
    directory: str, scope: str,
) -> None:
    defaults = f"defaults:\n  run:\n    working-directory: {directory}\n"
    prefix = defaults if scope == "workflow" else ""
    job = textwrap.indent(defaults, "    ") if scope == "job" else ""
    if scope == "step-first":
        step = f"      - working-directory: {directory}\n        run: pytest {_CAPABILITY_NODE}\n"
    else:
        step = f"      - run: pytest {_CAPABILITY_NODE}\n"
        if scope == "step-last":
            step += f"        working-directory: {directory}\n"
    workflow = prefix + f"jobs:\n  gpu-smoke:\n{job}    steps:\n" + step
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("workflow_directory,job_directory,step_directory", [
    (None, None, "."),
    (None, "./", None),
    (".", None, None),
    ("other", ".", None),
    ("other", None, "./"),
    (".", "other", "."),
])
def test_capability_ownership_accepts_effective_root_directory(
    workflow_directory: str | None, job_directory: str | None, step_directory: str | None,
) -> None:
    workflow = (
        f"defaults:\n  run:\n    working-directory: {workflow_directory}\n"
        if workflow_directory else ""
    )
    workflow += "jobs:\n  gpu-smoke:\n"
    if job_directory:
        workflow += f"    defaults:\n      run:\n        working-directory: {job_directory}\n"
    workflow += f"    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
    if step_directory:
        workflow += f"        working-directory: {step_directory}\n"
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


@pytest.mark.parametrize("quote", ["'", '"', ""])
@pytest.mark.parametrize("separator", ["", " ", "  "])
@pytest.mark.parametrize("key,value,scope", [
    ("shell", "python", "step"),
    ("shell", "python", "job"),
    ("shell", "python", "workflow"),
    ("working-directory", "other", "step"),
    ("working-directory", "other", "job"),
    ("working-directory", "other", "workflow"),
    ("if", "false", "step"),
    ("if", "false", "job"),
])
def test_capability_ownership_rejects_quoted_metadata_keys(
    quote: str, separator: str, key: str, value: str, scope: str,
) -> None:
    setting = f"{quote}{key}{quote}{separator}: {value}"
    workflow = f"defaults:\n  run:\n    {setting}\n" if scope == "workflow" else ""
    workflow += "jobs:\n  gpu-smoke:\n"
    if scope == "job":
        if key == "if":
            workflow += f"    {setting}\n"
        else:
            workflow += f"    defaults:\n      run:\n        {setting}\n"
    workflow += f"    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
    if scope == "step":
        workflow += f"        {setting}\n"
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("key", [r'"sh\u0065ll"', r'"working\u002ddirectory"'])
def test_capability_ownership_rejects_escaped_metadata_keys(key: str) -> None:
    workflow = (
        f"jobs:\n  gpu-smoke:\n    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
        f"        {key}: other\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


@pytest.mark.parametrize("key,value,scope", [
    ("shell", "python", "step"), ("shell", "python", "job"),
    ("shell", "python", "workflow"),
    ("working-directory", "other", "step"),
    ("working-directory", "other", "job"),
    ("working-directory", "other", "workflow"),
    ("if", "false", "step"), ("if", "false", "job"),
])
@pytest.mark.parametrize("form", ["explicit", "tagged", "anchored", "merge"])
def test_capability_ownership_rejects_unsupported_mapping_structure(
    key: str, value: str, scope: str, form: str,
) -> None:
    depth = 8 if scope == "step" or (scope == "job" and key != "if") else 4
    indent = " " * depth
    settings = {
        "explicit": f"{indent}? {key}\n{indent}: {value}\n",
        "tagged": f"{indent}!!str {key}: {value}\n",
        "anchored": f"{indent}&setting {key}: {value}\n",
        "merge": f"{indent}<<: {{{key}: {value}}}\n",
    }
    setting = settings[form]
    workflow = "defaults:\n  run:\n" + setting if scope == "workflow" else ""
    workflow += "jobs:\n  gpu-smoke:\n"
    if scope == "job":
        workflow += setting if key == "if" else "    defaults:\n      run:\n" + setting
    workflow += f"    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
    if scope == "step":
        workflow += setting
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


def test_capability_ownership_does_not_rewrite_literal_run_content() -> None:
    workflow = (
        "jobs:\n  gpu-smoke:\n    steps:\n      - run: |\n"
        "          # !!str shell: python\n"
        f"          pytest {_CAPABILITY_NODE}\n"
        f"      - run: pytest {_CAPABILITY_NODE}\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


@pytest.mark.parametrize("actual_job", ["actual", "gpu-smoke"])
def test_capability_ownership_cannot_borrow_job_from_scalar(actual_job: str) -> None:
    workflow = (
        "name: |\n  gpu-smoke:\n    steps:\n"
        f"      - run: pytest {_CAPABILITY_NODE}\n"
        f"jobs:\n  {actual_job}:\n    steps:\n      - run: true\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )


def test_capability_ownership_real_job_with_scalar_lookalike() -> None:
    workflow = (
        "name: |\n  gpu-smoke:\n    steps:\n      - run: true\n"
        "jobs:\n  gpu-smoke:\n    steps:\n"
        f"      - run: pytest {_CAPABILITY_NODE}\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    ) == []


def test_capability_ownership_cannot_borrow_sibling_command() -> None:
    workflow = (
        "jobs:\n  gpu-smoke:\n    steps:\n      - run: true\n"
        f"  sibling:\n    steps:\n      - run: pytest {_CAPABILITY_NODE}\n"
    )
    assert _capability_ownership_violations(
        f"{_CAPABILITY_NODE} # GPU owned by gpu-smoke", workflow
    )
