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
    assert "gpu_recovery_matches_cpu_on_paper_design" not in text
    assert "kang_jeon_2025_minimum_cell_recovers_true_parameters" not in text


def test_exhaustive_studies_are_scheduled_manual_and_release_triggered():
    """The heavy evidence suite is reproducible without blocking every PR."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert 'cron: "17 2 * * *"' in text
    assert '      - "v*"' in text
    assert "rust-ignored:" in text
    assert "rust-pyo3-ignored:" in text
    assert "rust-recovery:" in text
    assert "gpu-recovery:" in text


def test_ignored_rust_shards_allow_long_recovery_evidence_to_finish():
    """The shard deadline exceeds the historical 30-minute study timeout."""
    text = _STUDIES.read_text(encoding="utf-8")
    rust_ignored = text[text.index("  rust-ignored:"):text.index("\n  rust-pyo3-ignored:")]
    assert "timeout-minutes: 180" in rust_ignored
    assert 'FAST_MLSIRM_STATISTICAL_TEST_TIMEOUT_SECONDS: "7200"' in rust_ignored


def test_statistical_studies_are_read_only_and_never_rewrite_source():
    """Scientific evidence runs reviewed code and cannot commit replacements."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "write_text(" not in text
    assert "--exact" in text


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
    )
    raw_names = (
        "kang_jeon_2025_minimum_cell_recovers_true_parameters",
        "gpu_recovery_matches_cpu_on_paper_design",
        "higher_order_dina_recovery_respects_monte_carlo_tolerance",
    )
    for identifier in identifiers:
        assert f"--skip {identifier}" in text
    for name in raw_names:
        assert text.count(name) == 2
        assert f"--skip {name}" not in text
    assert "mc_ho_recovery_500" not in text
