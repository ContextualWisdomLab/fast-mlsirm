"""Workflow-contract tests separating bounded PR CI from exhaustive studies."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PR_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_STUDIES = _ROOT / ".github" / "workflows" / "statistical-studies.yml"
_WORKSPACE = _ROOT / "Cargo.toml"
_SHARD_RUNNER = _ROOT / "scripts" / "run_ignored_rust_shard.py"


def test_pull_request_ci_keeps_exhaustive_studies_out_of_the_queue():
    """PR CI retains one GPU smoke while excluding all exhaustive sweeps."""
    text = _PR_CI.read_text(encoding="utf-8")
    assert "gpu-smoke:" in text
    assert "rust-ignored:" not in text
    assert "rust-pyo3-ignored:" not in text
    assert "rust-recovery:" not in text
    assert "grm-recovery:" not in text
    assert "gpu-recovery:" not in text
    assert "gpu_recovery_matches_cpu_on_paper_design" not in text
    assert "kang_jeon_2025_minimum_cell_recovers_true_parameters" not in text
    assert "mc_grm_recovery_500" not in text


def test_exhaustive_studies_are_scheduled_manual_and_release_triggered():
    """The heavy evidence suite is reproducible without blocking every PR."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert 'cron: "17 2 * * *"' in text
    assert '      - "v*"' in text
    assert "rust-ignored:" in text
    assert "rust-pyo3-ignored:" in text
    assert "rust-recovery:" in text
    assert "grm-recovery:" in text
    assert "gpu-recovery:" in text


def test_manual_two_tier_profile_preserves_default_study_ownership():
    """수동으로 새 프로필을 고른 경우에만 기존 연구 job을 건너뛴다."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "study:" in text
    assert "default: all" in text
    assert "two-tier-reference-gpu" in text
    assert "reference_args:" in text
    assert "reference_resource_probe:" in text
    assert "type: boolean" in text
    assert "default: false" in text
    assert "default: \"\"" in text
    expected_guard = (
        "if: ${{ github.event_name != 'workflow_dispatch' "
        "|| inputs.study != 'two-tier-reference-gpu' }}"
    )
    for job in (
        "rust-ignored:",
        "rust-pyo3-ignored:",
        "rust-recovery:",
        "grm-recovery:",
        "gpu-recovery:",
    ):
        assert f"  {job}\n    {expected_guard}" in text
    assert "two-tier-reference-gpu:" in text
    assert "runs-on: [self-hosted, linux, x64, focal-gpu-isolated]" in text
    assert "timeout-minutes: 240" in text
    assert "contents: write" not in text


def test_reference_gpu_checkout_is_reviewed_and_credentials_are_not_persisted():
    """격리 runner는 고정된 checkout을 쓰고 인증 정보를 남기지 않는다."""
    text = _STUDIES.read_text(encoding="utf-8")
    checkout = "uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    assert text.count(checkout) == 6
    block = text.split("  two-tier-reference-gpu:\n", 1)[1]
    checkout_block = block.split(checkout, 1)[1].split("\n      - ", 1)[0]
    assert "persist-credentials: false" in checkout_block
    assert "python-version: \"3.12\"" in block
    assert "toolchain: 1.97.1" in block
    assert "requirements/ci.txt" in block
    assert ".venv" in block
    assert 'export PATH="$PWD/.venv/bin:$PATH"' in block
    assert "scripts/benchmark_two_tier_reference_gpu.py" in block
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in block


def test_reference_gpu_fit_has_one_hardware_execution_owner():
    """GPU 기준 적합은 CPU shard 대신 격리 hardware 잡에서 실제 실행한다."""
    name = "two_tier_grm::tests::reference_fit_actual_gpu_matches_cpu"
    assert f"--skip mlsirm-core/lib/mlsirm_core::{name}" in _STUDIES.read_text()
    text = _PR_CI.read_text()
    hardware = text.split("  focal-gpu-native:", 1)[1].split("\n  rust:", 1)[0]
    assert f"cargo test -p mlsirm-core --lib {name}" in hardware
    assert "--ignored --exact --nocapture | tee reference-gpu-fit.log" in hardware
    assert 'grep -q "test result: ok. 1 passed" reference-gpu-fit.log' in hardware
    assert "G1_GPU_RESOURCE_PROBE" not in text


def test_ignored_rust_shards_allow_long_recovery_evidence_to_finish():
    """The shard deadline exceeds the historical 30-minute study timeout."""
    text = _STUDIES.read_text(encoding="utf-8")
    rust_ignored = text[text.index("  rust-ignored:"):text.index("\n  rust-pyo3-ignored:")]
    assert "timeout-minutes: 180" in rust_ignored
    assert 'FAST_MLSIRM_STATISTICAL_TEST_TIMEOUT_SECONDS: "7200"' in rust_ignored


def test_statistical_studies_declare_the_secret_boundary():
    """The evidence workflow documents the reviewed secret-input policy."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "Secret boundary:" in text
    assert "${{ secrets.NAME }}" in text
    assert "never hardcode" in text


def test_statistical_studies_are_read_only_and_never_rewrite_source():
    """Scientific evidence runs reviewed code and cannot commit replacements."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "write_text(" not in text
    assert "--exact" in text


def test_grm_recovery_checkout_does_not_persist_credentials():
    """The dedicated GRM study keeps checkout credentials away from Rust tests."""
    text = _STUDIES.read_text(encoding="utf-8")
    _, grm_block = text.split("\n  grm-recovery:\n", maxsplit=1)
    grm_block, _ = grm_block.split("\n  gpu-recovery:\n", maxsplit=1)
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in grm_block
    assert "persist-credentials: false" in grm_block


def test_statistical_studies_checkouts_do_not_persist_credentials():
    """Every study checkout withholds the Actions token from executed code."""
    text = _STUDIES.read_text(encoding="utf-8")
    checkout = "uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    checkout_tails = text.split(checkout)[1:]
    assert len(checkout_tails) == 6
    for checkout_tail in checkout_tails:
        checkout_block = checkout_tail.split("\n      - ", maxsplit=1)[0]
        assert "\n        with:\n" in checkout_block
        assert "\n          persist-credentials: false" in checkout_block


def test_grm_recovery_publishes_a_durable_study_log():
    """Buyers can download bias/RMSE/convergence lines after the job log expires."""
    text = _STUDIES.read_text(encoding="utf-8")
    _, grm_block = text.split("\n  grm-recovery:\n", maxsplit=1)
    grm_block, _ = grm_block.split("\n  gpu-recovery:\n", maxsplit=1)
    assert "timeout-minutes: 120" in grm_block
    assert "grm::tests::mc_grm_recovery_500" in grm_block
    assert "--ignored" in grm_block
    assert "--exact" in grm_block
    assert "--test-threads=1" in grm_block
    assert "set -euo pipefail" in grm_block
    assert "tee grm-recovery-study.log" in grm_block
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
        in grm_block
    )
    assert "path: grm-recovery-study.log" in grm_block
    assert "if-no-files-found: error" in grm_block
    assert "retention-days: 90" in grm_block
    assert "if: always()" in grm_block


def test_general_and_pyo3_jobs_follow_the_declared_workspace_boundary():
    """The general inventory uses workspace metadata; excluded PyO3 is separate."""
    workflow = _STUDIES.read_text(encoding="utf-8")
    workspace = _WORKSPACE.read_text(encoding="utf-8")
    runner = _SHARD_RUNNER.read_text(encoding="utf-8")
    assert 'members = ["crates/mlsirm-core"]' in workspace
    assert 'exclude = ["crates/fast-mlsirm-py"]' in workspace
    assert "cargo_metadata_command" in runner
    assert '"metadata"' in runner
    assert '"workspace_members"' in runner
    assert "--exclude-package" not in workflow
    assert (
        "cargo test --release --manifest-path crates/fast-mlsirm-py/Cargo.toml"
        in workflow
    )


def test_dedicated_study_exclusions_are_target_qualified_and_executed_elsewhere():
    """Every shard exclusion identifies one package, target, and test function."""
    text = _STUDIES.read_text(encoding="utf-8")
    identifiers = (
        "mlsirm-core/test/literature_true_parameter_recovery::"
        "kang_jeon_2025_minimum_cell_recovers_true_parameters",
        "mlsirm-core/test/literature_true_parameter_recovery::"
        "gpu_recovery_matches_cpu_on_paper_design",
        "mlsirm-core/test/higher_order_mc_recovery::"
        "higher_order_dina_recovery_respects_monte_carlo_tolerance",
        "mlsirm-core/lib/mlsirm_core::grm::tests::mc_grm_recovery_500",
    )
    raw_names = (
        "kang_jeon_2025_minimum_cell_recovers_true_parameters",
        "gpu_recovery_matches_cpu_on_paper_design",
        "higher_order_dina_recovery_respects_monte_carlo_tolerance",
        "mc_grm_recovery_500",
    )
    for identifier in identifiers:
        assert f"--skip {identifier}" in text
    for name in raw_names:
        assert text.count(name) == 2
        assert f"--skip {name}" not in text
    assert "mc_ho_recovery_500" not in text
