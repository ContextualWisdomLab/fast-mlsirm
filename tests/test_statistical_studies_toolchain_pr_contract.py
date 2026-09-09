"""Contracts for exact-head scientific evidence on compiler-baseline changes."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_STUDIES = REPO_ROOT / ".github" / "workflows" / "statistical-studies.yml"
CORR_MIRT_TEST = "mlsirm-core/lib/mlsirm_core::twopl::tests::mc_corr_mirt_recovery_500"
QMC_MIRT_TEST = "mlsirm-core/lib/mlsirm_core::twopl::tests::mc_qmc_mirt_recovery_500"
GPCM_MHRM_TEST = "mlsirm-core/lib/mlsirm_core::mhrm::tests::mc_gpcm_mhrm_recovery_500"
MHRM_TEST = "mlsirm-core/lib/mlsirm_core::mhrm::tests::mc_mhrm_recovery_500"
TESTLET_NORMAL_TEST = "mlsirm-core/lib/mlsirm_core::testlet::tests::mc_testlet_recovery_500_normal"
TESTLET_SKEW_TEST = "mlsirm-core/lib/mlsirm_core::testlet::tests::mc_testlet_recovery_500_skew"


def _event_block(workflow: str) -> str:
    """Return the workflow trigger block without parsing GitHub's ``on`` key as YAML 1.1."""

    return workflow.split("\npermissions:", 1)[0]


def _concurrency_block(workflow: str) -> str:
    """Return the workflow concurrency block that governs same-ref study runs."""

    return workflow.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]


def _job_block(workflow: str, job: str, next_job: str) -> str:
    """Return one top-level job block delimited by the next named job."""

    return workflow.split(f"\n  {job}:\n", 1)[1].split(f"\n  {next_job}:\n", 1)[0]


def test_rust_toolchain_change_runs_scientific_studies_on_pull_request() -> None:
    """Compiler-baseline PRs must execute recovery/parity studies before landing."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    events = _event_block(workflow)

    assert "\n  pull_request:\n" in events
    assert "    paths:\n      - \"rust-toolchain.toml\"\n" in events


def test_new_toolchain_head_supersedes_stale_same_ref_scientific_run() -> None:
    """A synchronized PR head must not wait behind obsolete Monte Carlo evidence."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    concurrency = _concurrency_block(workflow)

    assert "  group: statistical-studies-${{ github.ref }}\n" in concurrency
    assert "  cancel-in-progress: true\n" in concurrency


def test_correlated_mirt_recovery_has_dedicated_bounded_evidence_job() -> None:
    """The multi-hour correlated-MIRT study must not inherit the generic 2 h child bound."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    rust_ignored = _job_block(workflow, "rust-ignored", "rust-pyo3-ignored")
    corr_mirt = _job_block(workflow, "corr-mirt-recovery", "qmc-mirt-recovery")

    assert f"--skip {CORR_MIRT_TEST}" in rust_ignored
    assert "    timeout-minutes: 360\n" in corr_mirt
    assert "twopl::tests::mc_corr_mirt_recovery_500" in corr_mirt
    assert "2>&1 | tee corr-mirt-recovery-study.log" in corr_mirt
    assert "if: always()" in corr_mirt
    assert "retention-days: 90" in corr_mirt


def test_qmc_mirt_recovery_has_dedicated_bounded_evidence_job() -> None:
    """The multi-hour QMC-MIRT study must not inherit the generic 2 h child bound."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    rust_ignored = _job_block(workflow, "rust-ignored", "rust-pyo3-ignored")
    qmc_mirt = _job_block(workflow, "qmc-mirt-recovery", "gpcm-mhrm-recovery")

    assert f"--skip {QMC_MIRT_TEST}" in rust_ignored
    assert "    timeout-minutes: 360\n" in qmc_mirt
    assert "twopl::tests::mc_qmc_mirt_recovery_500" in qmc_mirt
    assert "2>&1 | tee qmc-mirt-recovery-study.log" in qmc_mirt
    assert "if: always()" in qmc_mirt
    assert "retention-days: 90" in qmc_mirt


def test_gpcm_mhrm_recovery_has_dedicated_bounded_evidence_job() -> None:
    """The multi-hour GPCM-MHRM study must not inherit the generic 2 h child bound."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    rust_ignored = _job_block(workflow, "rust-ignored", "rust-pyo3-ignored")
    gpcm_mhrm = _job_block(workflow, "gpcm-mhrm-recovery", "mhrm-recovery")

    assert f"--skip {GPCM_MHRM_TEST}" in rust_ignored
    assert "    timeout-minutes: 360\n" in gpcm_mhrm
    assert "mhrm::tests::mc_gpcm_mhrm_recovery_500" in gpcm_mhrm
    assert "2>&1 | tee gpcm-mhrm-recovery-study.log" in gpcm_mhrm
    assert "if: always()" in gpcm_mhrm
    assert "retention-days: 90" in gpcm_mhrm


def test_mhrm_recovery_has_dedicated_bounded_evidence_job() -> None:
    """The multi-hour MHRM study must not inherit the generic 2 h child bound."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    rust_ignored = _job_block(workflow, "rust-ignored", "rust-pyo3-ignored")
    mhrm = _job_block(workflow, "mhrm-recovery", "gpu-recovery")

    assert f"--skip {MHRM_TEST}" in rust_ignored
    assert "    timeout-minutes: 360\n" in mhrm
    assert "mhrm::tests::mc_mhrm_recovery_500" in mhrm
    assert "2>&1 | tee mhrm-recovery-study.log" in mhrm
    assert "if: always()" in mhrm
    assert "retention-days: 90" in mhrm


def test_testlet_recovery_conditions_have_independent_bounded_evidence_jobs() -> None:
    """The two 500-rep testlet cells must not share the generic shard wall clock."""

    workflow = STATISTICAL_STUDIES.read_text(encoding="utf-8")
    rust_ignored = _job_block(workflow, "rust-ignored", "rust-pyo3-ignored")
    normal = _job_block(workflow, "testlet-normal-recovery", "testlet-skew-recovery")
    skew = _job_block(workflow, "testlet-skew-recovery", "corr-mirt-recovery")

    assert f"--skip {TESTLET_NORMAL_TEST}" in rust_ignored
    assert f"--skip {TESTLET_SKEW_TEST}" in rust_ignored

    for block, target, log_name in (
        (normal, "testlet::tests::mc_testlet_recovery_500_normal", "testlet-normal-recovery-study.log"),
        (skew, "testlet::tests::mc_testlet_recovery_500_skew", "testlet-skew-recovery-study.log"),
    ):
        assert "    timeout-minutes: 360\n" in block
        assert target in block
        assert f"2>&1 | tee {log_name}" in block
        assert "if: always()" in block
        assert "retention-days: 90" in block
