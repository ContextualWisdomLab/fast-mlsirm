"""Test table-wrapper focus contracts in script reports."""

from pathlib import Path
import re

import pytest

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


def test_jules_palette_focus_guidance_stays_on_canonical_path():
    """The lowercase agent-learning owner must receive this report contract."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_path = repo_root / ".jules" / "palette.md"
    legacy_case_path = repo_root / ".Jules" / "palette.md"
    canonical = canonical_path.read_text(encoding="utf-8")

    assert "마우스 사용자를 위한 표 컨테이너 초점 표시 관리" in canonical
    if legacy_case_path.exists():
        assert legacy_case_path.read_text(encoding="utf-8") == canonical
