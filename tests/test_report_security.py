"""Security regression tests for standalone diagnostics report rendering."""

from html import escape
import json
import base64
import hashlib
import re

from fast_mlsirm.report import _metric_section, _table_section


def test_report_section_heading_ids_reject_html_attribute_injection():
    """Keep report section IDs inert when headings contain active HTML syntax."""
    heading = '"><script>alert("xss")</script>'
    rendered_sections = [
        _metric_section(heading, {"loglik": -1.0}),
        _table_section(heading, [{"value": 1.0}]),
    ]

    for section in rendered_sections:
        assert section is not None
        assert "<script>" not in section
        assert 'aria-labelledby=""><' not in section
        assert 'id=""><' not in section
        assert escape(heading) in section

def test_report_csp_style_hash_matches_content(tmp_path):
    """Verify the CSP style-src hash exactly matches the rendered <style> content."""
    from fast_mlsirm.report import render_diagnostics_report

    # Generate a minimal fit report
    diag_path = tmp_path / "diag.json"
    diag_path.write_text(json.dumps({
        "metadata": {
            "version": "1",
            "run_id": "test",
            "engine": "test"
        },
        "model_fit": {
            "loglik": -1.0,
            "aic": 1.0,
            "bic": 1.0,
            "aicc": 1.0,
            "deviance": 1.0,
            "df": 1
        },
        "summary": {
            "person_count": 1,
            "item_count": 1,
            "response_count": 1
        },
        "metrics": {},
        "exact_values": {}
    }))

    out_path = tmp_path / "out.html"
    render_diagnostics_report(diag_path, out_path)

    html = out_path.read_text(encoding="utf-8")

    # Extract the style element's text content (excluding tags)
    style_match = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    assert style_match is not None, "Report must contain a <style> element"
    style_content = style_match.group(1)

    # Extract the CSP style-src hash
    csp_match = re.search(r"style-src (?:'|&#x27;)sha256-([^'&#]+)(?:'|&#x27;)", html)
    assert csp_match is not None, "Report must declare a style-src hash in CSP"
    csp_hash = csp_match.group(1)

    # Compute the expected hash
    expected_hash = base64.b64encode(hashlib.sha256(style_content.encode("utf-8")).digest()).decode("utf-8")

    # They must exactly match, including any whitespace
    assert expected_hash == csp_hash, f"CSP hash does not match <style> content. Hash: {csp_hash} Content starts with: {repr(style_content[:10])}"
