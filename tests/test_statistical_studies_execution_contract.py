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
    assert '"--out", "reference-gpu-benchmark.json",' in job
    assert "shell=True" not in job
    assert "secrets." not in job
    assert "GH_TOKEN" not in job
    assert "timeout=" not in job
    assert "timeout-minutes: 240" in job
    assert "without JSON evidence" in job
    assert "stdout.log" in job
    assert "stderr.log" in job
    assert "reference-gpu-benchmark.json" in job
    assert "if: always()" in job


def test_two_tier_reference_keeps_all_numerical_controls_caller_required():
    """호출자가 모든 수치 제어값을 정하며 q121 수렴을 지름길로 인정하지 않는다."""
    workflow = _workflow_text()
    _, job = workflow.split("  two-tier-reference-gpu:\n", maxsplit=1)
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
        assert option not in job
    assert "--device both" not in job
    assert "--out reference-gpu-benchmark.json" not in job


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
