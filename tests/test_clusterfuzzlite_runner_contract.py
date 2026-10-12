"""Source routing contracts; runner isolation and fuzz execution need receipts."""
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "cflite_pr.yml"


def test_clusterfuzzlite_requires_isolated_linux_x64_self_hosted_routing() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    selectors = re.findall(r"^    runs-on: (.+)$", workflow, re.MULTILINE)
    assert selectors == ["[self-hosted, Linux, X64, cwlab-ci-isolated]"], (
        "ClusterFuzzLite must request the operator-provisioned isolated pool, "
        "not hosted or privileged control/scanner workers"
    )


def test_clusterfuzzlite_retains_event_guard_and_single_sanitizer_job() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    jobs = workflow.split("jobs:\n", 1)[1]
    assert re.findall(r"^  ([A-Za-z0-9_-]+):$", jobs, re.MULTILINE) == ["PR"]
    assert "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, closed]" in workflow
    assert "if: ${{ !github.event.pull_request.draft && github.event.action != 'closed' }}" in jobs
    assert "sanitizer: [address]" in jobs
    assert "cancel-in-progress: true" in jobs
    for path in (".clusterfuzzlite/**", ".github/workflows/cflite_pr.yml", "Cargo.toml", "crates/**", "fuzz/**"):
        assert f'      - "{path}"' in workflow


def test_clusterfuzzlite_retains_pins_permissions_and_fuzz_budget() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    pin = "884713a6c30a92e5e8544c39945cd7cb630abcd1"
    assert f"google/clusterfuzzlite/actions/build_fuzzers@{pin}" in workflow
    assert f"google/clusterfuzzlite/actions/run_fuzzers@{pin}" in workflow
    assert workflow.count("github-token: ${{ secrets.GITHUB_TOKEN }}") == 2
    assert "permissions:\n  contents: read\n" in workflow
    assert "    permissions:\n      contents: read\n      security-events: write\n" in workflow
    assert "fuzz-seconds: 120" in workflow
    assert "mode: code-change" in workflow
    assert "output-sarif: true" in workflow
