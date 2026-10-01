"""Contract tests for Statistical Studies dedicated-job execution evidence.

Implementation basis: the fail-open defect documented in
ContextualWisdomLab/fast-mlsirm#1869, which was read in full via the GitHub
API before implementation. That issue proves ``cargo test ... --exact <name>``
exits 0 while reporting ``0 passed`` when the filter matches no test, so each
dedicated recovery-study step must assert its test actually executed instead of
relying on the process exit status alone (Issue #1869, 2026).

References
----------
ContextualWisdomLab. (2026). *Test governance: dedicated Statistical Studies
jobs pass when their exact test filter matches nothing* (Issue #1869).
https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1869
"""

from __future__ import annotations

from pathlib import Path

_WORKFLOW = (
    Path(__file__).parents[1] / ".github" / "workflows" / "statistical-studies.yml"
)

# The four scientific-validity recovery studies that run in dedicated jobs
# rather than in the ignored-test shard (Issue #1869, 2026).
_DEDICATED_TESTS = (
    "kang_jeon_2025_minimum_cell_recovers_true_parameters",
    "higher_order_dina_recovery_respects_monte_carlo_tolerance",
    "mc_grm_recovery_500",
    "gpu_recovery_matches_cpu_on_paper_design",
)

# Every dedicated step must require exactly one executed passing test in the
# captured cargo output; ``0 passed`` (zero-match filter) must fail the job.
_EXECUTION_ASSERTION = "test result: ok. 1 passed"


def _workflow_text() -> str:
    """Return the Statistical Studies workflow text under contract."""
    return _WORKFLOW.read_text(encoding="utf-8")


def test_dedicated_recovery_steps_reference_their_exact_tests():
    """Each dedicated job still selects its recovery study by exact filter."""
    workflow = _workflow_text()
    for test_name in _DEDICATED_TESTS:
        assert test_name in workflow, (
            f"dedicated Statistical Studies job lost its exact filter: {test_name}"
        )


def test_dedicated_recovery_steps_assert_exactly_one_passed_test():
    """A zero-match exact filter (``0 passed``) cannot report success.

    Basis: Issue #1869 proves ``cargo test --exact`` exits 0 when the filter
    matches nothing, so each dedicated step must assert ``1 passed`` in its
    captured output (Issue #1869, 2026).
    """
    workflow = _workflow_text()
    occurrences = workflow.count(_EXECUTION_ASSERTION)
    assert occurrences >= len(_DEDICATED_TESTS), (
        "each of the four dedicated recovery steps must assert "
        f"'{_EXECUTION_ASSERTION}'; found {occurrences} assertion(s)"
    )


def test_grm_recovery_study_log_is_still_published():
    """The GRM recovery study log remains a published evidence artifact."""
    workflow = _workflow_text()
    assert "grm-recovery-study.log" in workflow
    assert "upload-artifact" in workflow


def test_two_tier_reference_uses_shlex_and_shell_free_runner_invocation():
    """호출 인자는 셸 명령이 되지 않고 고정 출력 경로를 바꾸지 못한다."""
    workflow = _workflow_text()
    _, job = workflow.split("  two-tier-reference-gpu:\n", maxsplit=1)
    assert "REFERENCE_ARGS: ${{ inputs.reference_args }}" in job
    assert "shlex.split(os.environ[\"REFERENCE_ARGS\"])" in job
    assert "subprocess.run(" in job
    assert "shell=False" in job
    assert '"--device", "both",' in job
    assert '"--out", str(root / "reference-gpu-benchmark.json"),' in job
    assert "shell=True" not in job
    assert "secrets." not in job
    assert "GH_TOKEN" not in job
    assert "run_owned(command, 1200" in job
    assert "run_owned(build, 1800" in job
    assert "timeout-minutes: 240" in job
    assert "timeout-minutes: 60" in job
    assert "without JSON evidence" in job
    assert "stdout.log" in job
    assert "stderr.log" in job
    assert "reference-gpu-benchmark.json" in job
    assert "if: always()" in job
    assert "reference-gpu-resource-probe.log" in job


def test_public_reference_route_binds_one_wheel_and_external_receipt():
    """격리 정상 경로의 로그·wheel·실제 import가 source 검사 전에 결속된다."""
    job = _workflow_text().split("  two-tier-reference-gpu:\n", 1)[1]
    benchmark = job.split("      - name: 호출자가 정한 조건으로 two-tier 기준 적합 측정", 1)[1]
    assert 'Path(os.environ["RUNNER_TEMP"]) / "g1-public-proof"' in benchmark
    assert '".venv/bin/maturin", "build", "--release", "--locked"' in benchmark
    assert '"--interpreter", ".venv/bin/python"' in benchmark
    assert '"--build-receipt", str(root / "build-receipt.json")' in benchmark
    assert 'root / "reference-gpu-benchmark.json"' in benchmark
    assert 'start_new_session=True' in benchmark
    assert 'os.killpg(child.pid, signal.SIGTERM)' in benchmark
    assert '            ${{ runner.temp }}/g1-public-proof/\n' in job


def test_public_reference_wheel_identity_rejects_ambiguous_or_stale_import(tmp_path, monkeypatch):
    """비실행 wheel/core stub 대조이며 native build나 GPU 증거가 아니다."""
    import ast
    import hashlib
    import sys
    import textwrap
    import types
    import zipfile
    import pytest

    block = _workflow_text().split("      - name: 호출자가 정한 조건으로 two-tier 기준 적합 측정", 1)[1]
    source = textwrap.dedent(block.split("          .venv/bin/python - <<'PY'\n", 1)[1].split("          PY", 1)[0])
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "wheel_identity")
    namespace = dict(Path=Path, hashlib=hashlib, zipfile=zipfile, sys=sys)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "workflow-wheel-contract", "exec"), namespace)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    prefix = tmp_path / "venv"
    core = prefix / "fast_mlsirm" / "_core.stub.so"
    core.parent.mkdir(parents=True)
    core.write_bytes(b"unit-test native member stub")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    module = types.ModuleType("fast_mlsirm")
    module._core = types.SimpleNamespace(__file__=str(core))
    monkeypatch.setitem(sys.modules, "fast_mlsirm", module)
    def wheel(path, content):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("fast_mlsirm/_core.stub.so", content)
    first = wheels / "one.whl"
    wheel(first, core.read_bytes())
    identity = namespace["wheel_identity"](wheels)
    assert identity["core_sha256"] == hashlib.sha256(core.read_bytes()).hexdigest()
    assert identity["wheel_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    second = wheels / "two.whl"
    wheel(second, core.read_bytes())
    with pytest.raises(RuntimeError, match="one wheel"):
        namespace["wheel_identity"](wheels)
    second.unlink()
    core.write_bytes(b"old installed extension")
    with pytest.raises(RuntimeError, match="wheel.*extension"):
        namespace["wheel_identity"](wheels)


def test_owned_process_resource_stop_signals_only_matching_group(tmp_path):
    """소유 group과 start identity를 확인하며 외부 프로세스에는 signal을 보내지 않는다."""
    import ast
    import json
    import signal
    import subprocess
    import textwrap
    import time
    import types
    import pytest

    block = _workflow_text().split("      - name: 호출자가 정한 조건으로 two-tier 기준 적합 측정", 1)[1]
    source = textwrap.dedent(block.split("          .venv/bin/python - <<'PY'\n", 1)[1].split("          PY", 1)[0])
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_owned")
    for case in ("term", "kill", "identity_mismatch", "start_error"):
        root = tmp_path / case
        root.mkdir()
        signals = []
        starts = []
        class Child:
            pid = 42
            returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                if case == "kill" and len(signals) == 1:
                    raise subprocess.TimeoutExpired("unit-test-only", timeout)
                self.returncode = -15 if case == "term" else -9
                return self.returncode
        child = Child()
        def launch(*args, **kwargs):
            assert kwargs["shell"] is False and kwargs["start_new_session"] is True
            return child
        def start(pid):
            starts.append(pid)
            if case == "start_error":
                child.returncode = 1
                raise FileNotFoundError("unit-test process already exited")
            return "changed" if case == "identity_mismatch" and len(starts) > 1 else "unit-test-start"
        namespace = dict(root=root, time=time, json=json, signal=signal, process_start=start,
            resources=lambda: {"cgroup_current_bytes":7*1024**3,"host_available_bytes":32*1024**3,"gpu_used_mib":28},
            os=types.SimpleNamespace(getuid=lambda:1000, getpgid=lambda pid:pid, getpgrp=lambda:99,
                                     killpg=lambda pid, sig:signals.append((pid,sig))),
            subprocess=types.SimpleNamespace(Popen=launch, check_output=lambda *a, **k:"42 100\n99 200\n", TimeoutExpired=subprocess.TimeoutExpired))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "workflow-owned-process-contract", "exec"), namespace)
        if case in ("identity_mismatch", "start_error"):
            exception = RuntimeError if case == "identity_mismatch" else FileNotFoundError
            with pytest.raises(exception):
                namespace["run_owned"](["unit-test-only"], 1200, "stub")
            assert signals == []
        else:
            assert namespace["run_owned"](["unit-test-only"], 1200, "stub") == 1
            assert signals[0] == (42, signal.SIGTERM)
            assert signals == [(42,signal.SIGTERM)] + ([(42,signal.SIGKILL)] if case == "kill" else [])
        receipt = json.loads((root / "stub-process.json").read_text())
        assert receipt["pid"] == receipt["pgid"] == 42
        assert receipt["reason"] in ("resource_limit", "monitor_error")


def test_public_reference_step_has_cleanup_and_receipt_time_budget():
    import re
    step = _workflow_text().split("      - name: 호출자가 정한 조건으로 two-tier 기준 적합 측정", 1)[1]
    step = step.split("      - name: Two-tier 기준 적합 측정 증거 보관", 1)[0]
    outer = int(re.search(r"timeout-minutes: (\d+)", step).group(1))*60
    owned = sum(int(value) for value in re.findall(r"run_owned\((?:build|command), (\d+)", step))
    assert outer >= owned + 120 + 120


def test_two_tier_resource_probe_is_not_a_fit_acceptance_receipt():
    """Adapter probe는 수치 적합 또는 수렴 수용으로 계산하지 않는다."""
    workflow = _workflow_text()
    _, job = workflow.split("  two-tier-reference-gpu:\n", maxsplit=1)
    assert "reference_resource_probe:" not in job
    assert 'if: ${{ inputs.reference_resource_probe == true }}' in job
    assert 'if: ${{ inputs.reference_resource_probe != true && inputs.reference_kernel_probe != true }}' in job
    assert 'G1_GPU_RESOURCE_PROBE: "1"' in job
    assert "cargo test -p mlsirm-core --lib" in job
    assert (
        "two_tier_grm::tests::reference_fit_actual_gpu_matches_cpu "
        "\\\n            -- --ignored --exact --nocapture"
    ) in job
    assert "tee reference-gpu-resource-probe.log" in job
    assert 'grep -q "test result: ok. 1 passed" reference-gpu-resource-probe.log' in job
    assert "adapter 연결 증거일 뿐 수치 적합 또는 과학적 수용이 아닙니다." in job
    assert '"--device", "both",' in job
    assert "reference-gpu-resource-probe.log" in job


def test_two_tier_reference_keeps_all_numerical_controls_caller_required():
    """호출자가 모든 수치 제어값을 정하며 q121 수렴을 지름길로 인정하지 않는다."""
    workflow = _workflow_text()
    _, job = workflow.split("  two-tier-reference-gpu:\n", maxsplit=1)
    benchmark = job.split("      - name: 호출자가 정한 조건으로 two-tier 기준 적합 측정", 1)[1]
    for option in (
        "--persons",
        "--q-primary",
        "--q-specific",
        "--max-iter",
        "--tol",
        "--n-starts",
        "--seed",
        "--gpu-memory-budget-bytes",
    ):
        assert option not in benchmark
    assert "--device both" not in benchmark
    assert "--out reference-gpu-benchmark.json" not in benchmark


def test_fixed_bank_kernel_probe_is_explicit_and_has_no_fit_receipt():
    """단일 E-step 검사와 adapter·전체 적합의 완료 판정을 혼합하지 않는다."""
    text = _workflow_text()
    job = text.split("  two-tier-reference-gpu:\n", 1)[1]
    assert "reference_kernel_probe:" in text and "default: false" in text
    assert "resource-only와 kernel probe는 한 번에 선택할 수 없습니다" in job
    assert "inputs.reference_kernel_probe == true" in job
    assert 'argparse.ArgumentParser(allow_abbrev=False)' in job
    assert 'parser.add_argument("--" + name, required=True, type=int)' in job
    assert 'parser.parse_args(shlex.split(os.environ["REFERENCE_ARGS"]))' in job
    assert "G1_KERNEL_Q_PRIMARY" in job and "G1_KERNEL_Q_SPECIFIC" in job
    assert "G1_KERNEL_GPU_BUDGET_BYTES" in job and "G1_KERNEL_HOST_BUDGET_BYTES" in job
    assert "G1_EXECUTION_HEAD: ${{ github.sha }}" in job
    assert "two_tier_grm::tests::reference_gpu_fixed_bank_kernel_profile" in job
    assert "reference-gpu-kernel-probe.log" in job and "test result: ok. 1 passed" in job
    assert "--skip mlsirm-core/lib/mlsirm_core::two_tier_grm::tests::reference_gpu_fixed_bank_kernel_profile" in text
    assert "수렴 적합·모수 갱신·연구 수용이 아닙니다" in job
    source = (_WORKFLOW.parents[2] / "tests/unit/two_tier_grm_tests.rs").read_text()
    profile = source.split("fn reference_gpu_fixed_bank_kernel_profile()", 1)[1].split("\n#[test]", 1)[0]
    assert "required_timing_keys" in profile
    assert "seconds.is_finite()" in profile
    assert "seconds >= 0.0" in profile
    assert "timings.len()" in profile
    assert "prepare_compute_pipeline_seconds" in profile
    assert "prepare_table_map_copy_seconds" in profile
    assert "sweep_readback_buffer_preparation_seconds" in profile
    assert "timing_receipt_negative_cases=5" in profile


def test_two_tier_success_requires_converged_cpu_gpu_and_gpu_adapter_evidence():
    """성공 상태에는 두 장치의 수렴과 실제 GPU adapter 기록이 필요하다."""
    workflow = _workflow_text()
    _, job = workflow.split("  two-tier-reference-gpu:\n", maxsplit=1)
    for condition in (
        'json.loads(report_path.read_text(encoding="utf-8"))',
        'set(runs) != {"cpu", "gpu"}',
        'report.get("exit_status")',
        'runs[device].get("converged") is not True',
        'gpu.get("backend") != "gpu"',
        'gpu.get("gpu_adapter_name")',
        'gpu.get("gpu_adapter_backend")',
    ):
        assert condition in job
    assert "tolerance" not in job.lower()
