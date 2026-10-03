"""Source-only contracts for the isolated CI routing requirement.

These contracts do not attest runner registration, physical isolation or actual
native/GPU execution. Those require the existing runner operator's receipts.
"""
from pathlib import Path
import re

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_JOBS = ("python-matrix", "python", "rust", "gpu-smoke", "fuzz", "package")
_CHECKOUT_JOBS = tuple(job for job in _JOBS if job != "python")


def _job(workflow: str, name: str) -> str:
    """Read one literal top-level job from the retained CI workflow grammar."""
    pattern = rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
    matches = re.findall(pattern, workflow, re.MULTILINE | re.DOTALL)
    assert len(matches) == 1, f"expected exactly one CI job: {name}"
    return matches[0]


@pytest.mark.parametrize("name", _JOBS)
def test_ci_jobs_require_isolated_linux_x64_self_hosted_routing(name: str) -> None:
    """Keep every current CI job off hosted/control/scanner fallback routes."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    selectors = re.findall(r"^    runs-on: (.+)$", job, re.MULTILINE)
    assert selectors == ["[self-hosted, Linux, X64, cwlab-ci-isolated]"], (
        f"{name} must request the operator-provisioned isolated pool, without fallback"
    )


@pytest.mark.parametrize("name", _CHECKOUT_JOBS)
def test_ci_checkouts_do_not_persist_credentials(name: str) -> None:
    """Executable PR code must not inherit a checkout-persisted Git credential."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    checkouts = re.findall(
        r"^      - uses: actions/checkout@[^\n]+\n(.*?)(?=^      - |\Z)",
        job,
        re.MULTILINE | re.DOTALL,
    )
    assert len(checkouts) == 1
    assert re.search(r"^        with:\n          persist-credentials: false$", checkouts[0], re.MULTILINE)


def test_ci_keeps_all_existing_execution_job_identities() -> None:
    """Do not remove a failing required producer to satisfy the routing check."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert tuple(re.findall(r"^  ([A-Za-z0-9_-]+):$", workflow.split("jobs:\n", 1)[1], re.MULTILINE)) == _JOBS
