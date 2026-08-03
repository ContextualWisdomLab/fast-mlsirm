"""Contracts for the quarantined historical higher-order DINA Monte Carlo gate.

The historical unit test is retained as provenance, but its exact finite-sample
``0.95`` assertion is superseded by the reviewed integration study.  The
statistical workflow must therefore quarantine only that fully qualified test
name and execute the replacement study in a dedicated job.
"""

from __future__ import annotations

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CDM_TESTS = _REPOSITORY_ROOT / "tests" / "unit" / "cdm_tests.rs"
_REPLACEMENT_STUDY = (
    _REPOSITORY_ROOT
    / "crates"
    / "mlsirm-core"
    / "tests"
    / "higher_order_mc_recovery.rs"
)
_STATISTICAL_WORKFLOW = (
    _REPOSITORY_ROOT / ".github" / "workflows" / "statistical-studies.yml"
)


def test_historical_exact_threshold_is_quarantined_by_exact_name() -> None:
    """Only the known superseded test may be excluded from the exhaustive shard."""
    source = _CDM_TESTS.read_text(encoding="utf-8")
    workflow = _STATISTICAL_WORKFLOW.read_text(encoding="utf-8")

    assert "fn mc_ho_recovery_500()" in source
    assert "conv_rate >= 0.95" in source
    exact_skip = "--skip cdm::tests::mc_ho_recovery_500"
    assert workflow.count(exact_skip) == 1
    assert "--skip mc_ho_recovery_500" not in workflow


def test_replacement_gate_owns_finite_monte_carlo_acceptance() -> None:
    """The dedicated reviewed study must implement and execute the MCSE floor."""
    source = _REPLACEMENT_STUDY.read_text(encoding="utf-8")
    workflow = _STATISTICAL_WORKFLOW.read_text(encoding="utf-8")

    assert "fn higher_order_dina_recovery_respects_monte_carlo_tolerance()" in source
    assert "nominal_convergence" in source
    assert "monte_carlo_se" in source
    assert "convergence_floor" in source
    test_name = "higher_order_dina_recovery_respects_monte_carlo_tolerance"
    assert workflow.count(test_name) == 2
    assert f"--skip {test_name}" in workflow


def test_no_source_mutating_statistical_patch_workflow_is_tracked() -> None:
    """Reviewed source changes cannot be delegated to a write-capable CI patcher."""
    forbidden = (
        _REPOSITORY_ROOT
        / ".github"
        / "workflows"
        / "apply-reviewed-ho-mc-source-fix.yml"
    )
    assert not forbidden.exists()
