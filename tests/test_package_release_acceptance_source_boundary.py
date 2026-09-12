from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _job_source(name: str, next_name: str) -> str:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    marker = f"\n  {name}:\n"
    next_marker = f"\n  {next_name}:\n"
    assert marker in workflow
    body = workflow.split(marker, 1)[1]
    assert next_marker in body
    return body.split(next_marker, 1)[0]


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


def test_python_matrix_keeps_test_build_outputs_out_of_source_seal() -> None:
    python_matrix = _job_source("python-matrix", "python")

    assert "CARGO_TARGET_DIR: /tmp/fast-mlsirm-python-${{ matrix.python-version }}-target" in python_matrix
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in python_matrix
    assert 'PYTEST_ADDOPTS: "-p no:cacheprovider"' in python_matrix


def test_package_release_acceptance_does_not_disable_source_seal() -> None:
    package_job = _package_job_source()

    assert "scripts/release_acceptance.py" in package_job
    assert "--distribution-root ." not in package_job
    assert "git clean" not in package_job
    assert "git reset --hard" not in package_job
