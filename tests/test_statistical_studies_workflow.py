"""Workflow-contract tests separating bounded PR CI from exhaustive studies."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PR_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_STUDIES = _ROOT / ".github" / "workflows" / "statistical-studies.yml"


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


def test_statistical_studies_are_read_only_and_never_rewrite_source():
    """Scientific evidence runs reviewed code and cannot commit replacements."""
    text = _STUDIES.read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "write_text(" not in text
    assert "--exact" in text


def test_dedicated_study_exclusions_are_exact_and_executed_elsewhere():
    """Every shard exclusion is an exact name with a matching dedicated command."""
    text = _STUDIES.read_text(encoding="utf-8")
    dedicated = (
        "kang_jeon_2025_minimum_cell_recovers_true_parameters",
        "gpu_recovery_matches_cpu_on_paper_design",
        "higher_order_dina_recovery_respects_monte_carlo_tolerance",
    )
    for name in dedicated:
        assert f"--skip {name}" in text
        assert text.count(name) == 2
    assert "mc_ho_recovery_500" not in text
