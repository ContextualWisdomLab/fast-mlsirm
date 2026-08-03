"""Source-level contract for the historical higher-order DINA Monte Carlo gate."""

from __future__ import annotations

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CDM_TESTS = _REPOSITORY_ROOT / "tests" / "unit" / "cdm_tests.rs"
_STATISTICAL_WORKFLOW = (
    _REPOSITORY_ROOT / ".github" / "workflows" / "statistical-studies.yml"
)


def test_historical_higher_order_gate_accounts_for_finite_replication_error() -> None:
    """The original Rust study must use an explicit Monte Carlo uncertainty floor."""
    source = _CDM_TESTS.read_text(encoding="utf-8")
    assert "fn mc_ho_recovery_500()" in source
    assert "conv_rate >= 0.95" not in source
    assert "nominal_convergence" in source
    assert "monte_carlo_se" in source
    assert "convergence_floor" in source


def test_statistical_sweep_executes_the_historical_study_without_an_exception() -> None:
    """The exhaustive ignored-test inventory must not skip the corrected study."""
    workflow = _STATISTICAL_WORKFLOW.read_text(encoding="utf-8")
    assert "--skip mc_ho_recovery_500" not in workflow


def test_no_source_mutating_statistical_patch_workflow_is_tracked() -> None:
    """Reviewed source changes cannot be delegated to a write-capable CI patcher."""
    forbidden = (
        _REPOSITORY_ROOT
        / ".github"
        / "workflows"
        / "apply-reviewed-ho-mc-source-fix.yml"
    )
    assert not forbidden.exists()
