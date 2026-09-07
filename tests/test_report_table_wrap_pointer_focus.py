"""Test CSS behavior of table-wrap pointer focus in script reports."""

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
    """
    Ensure the table wrapper provides pointer focus suppression
    but strictly preserves keyboard accessibility and avoids blanket outline none.
    """
    css = css_factory()

    # Must have pointer focus suppression rule
    assert ".table-wrap:focus:not(:focus-visible)" in css

    # Must explicitly set a keyboard focus ring
    assert ".table-wrap:focus-visible" in css

    # Must not blanket disable all focus outlines
    assert ".table-wrap:focus {" not in css

    # Extract blocks to check contents
    import re
    not_visible_block = re.search(r'\.table-wrap:focus:not\(:focus-visible\)\s*\{([^}]*)\}', css)
    assert not_visible_block is not None
    assert "outline: none;" in not_visible_block.group(1)

    visible_block = re.search(r'\.table-wrap:focus-visible\s*\{([^}]*)\}', css)
    assert visible_block is not None
    assert "outline: 3px solid" in visible_block.group(1)
