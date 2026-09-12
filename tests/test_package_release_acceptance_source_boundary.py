from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _package_job_source() -> str:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    marker = "\n  package:\n"
    assert marker in workflow
    return workflow.split(marker, 1)[1]


def test_package_release_acceptance_keeps_build_outputs_out_of_source_seal() -> None:
    package_job = _package_job_source()

    assert "CARGO_TARGET_DIR: /tmp/fast-mlsirm-package-target" in package_job
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in package_job
    assert "--distribution-root dist" in package_job


def test_package_release_acceptance_does_not_disable_source_seal() -> None:
    package_job = _package_job_source()

    assert "scripts/release_acceptance.py" in package_job
    assert "--distribution-root ." not in package_job
    assert "git clean" not in package_job
    assert "git reset --hard" not in package_job
