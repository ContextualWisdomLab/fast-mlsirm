"""Test table-wrapper focus contracts in script reports."""

from pathlib import Path
import re
import subprocess
import sys

import pytest

from scripts.build_benchmark_report import _render_report_html as render_benchmark_report
from scripts.build_benchmark_report import _report_css as benchmark_css
from scripts.build_buyer_packet import _report_css as buyer_css
from scripts.build_pr_queue_governance import _report_css as pr_queue_css
from scripts.build_procurement_due_diligence import _report_css as due_diligence_css
from scripts.build_release_evidence_index import _report_css as release_index_css


@pytest.mark.parametrize(
    "css_factory",
    [
        benchmark_css,
        buyer_css,
        pr_queue_css,
        due_diligence_css,
        release_index_css,
    ],
)
def test_table_wrap_focus_accessibility(css_factory):
    """Pointer focus suppression must not remove keyboard focus visibility."""
    css = css_factory()

    assert ".table-wrap:focus:not(:focus-visible)" in css
    assert ".table-wrap:focus-visible" in css
    assert ".table-wrap:focus {" not in css

    not_visible_block = re.search(
        r"\.table-wrap:focus:not\(:focus-visible\)\s*\{([^}]*)\}", css
    )
    assert not_visible_block is not None
    assert "outline: none;" in not_visible_block.group(1)

    visible_block = re.search(r"\.table-wrap:focus-visible\s*\{([^}]*)\}", css)
    assert visible_block is not None
    assert "outline: 3px solid" in visible_block.group(1)


def test_benchmark_skip_link_has_static_and_browser_contract():
    """Benchmark evidence must expose a motion-free keyboard bypass path."""
    html = render_benchmark_report({})
    css = benchmark_css()
    repo_root = Path(__file__).resolve().parents[1]
    verifier = (repo_root / "scripts" / "verify_report_browser_e2e.py").read_text(
        encoding="utf-8"
    )

    assert '<a href="#main-content" class="skip-link">Skip to main content</a>' in html
    assert '<main id="main-content" tabindex="-1">' in html
    assert ".skip-link:focus-visible" in css
    assert "main:focus-visible" in css
    assert "transition:" not in css
    assert "_assert_skip_link_focus" in verifier
    assert "_assert_skip_target_focus" in verifier
    assert "session.press_enter()" in verifier


def test_jules_palette_focus_guidance_stays_on_canonical_path():
    """The lowercase agent-learning owner must receive this report contract."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_path = repo_root / ".jules" / "palette.md"
    legacy_case_path = repo_root / ".Jules" / "palette.md"
    canonical = canonical_path.read_text(encoding="utf-8")

    assert "마우스 사용자를 위한 표 컨테이너 초점 표시 관리" in canonical
    if legacy_case_path.exists():
        assert legacy_case_path.read_text(encoding="utf-8") == canonical


def test_report_focus_contract_has_real_browser_e2e_lane():
    """Material report focus evidence must execute in a real browser on PR heads."""
    repo_root = Path(__file__).resolve().parents[1]
    verifier = repo_root / "scripts" / "verify_report_browser_e2e.py"
    workflow = repo_root / ".github" / "workflows" / "report-browser-e2e.yml"

    assert verifier.is_file()
    assert workflow.is_file()
    workflow_text = workflow.read_text(encoding="utf-8")
    assert "python -m scripts.verify_report_browser_e2e" in workflow_text
    assert "report-browser-e2e.json" in workflow_text
    assert "actions/upload-artifact@" in workflow_text
    assert '- ".jules/palette.md"' in workflow_text
    assert '- ".Jules/palette.md"' in workflow_text


def test_report_browser_verifier_module_entrypoint_loads():
    """The workflow's module invocation must reach argparse before browser use."""
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.verify_report_browser_e2e", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Verify commercial report focus" in completed.stdout


def test_report_browser_e2e_is_bound_to_exact_pr_head():
    """Browser evidence must reject GitHub's synthetic pull-request merge checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (
        repo_root / ".github" / "workflows" / "report-browser-e2e.yml"
    ).read_text(encoding="utf-8")

    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "EXPECTED_SOURCE_COMMIT: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "evidence[\"source_commit\"] == expected" in workflow
