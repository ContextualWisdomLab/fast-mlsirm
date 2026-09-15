"""The report CSP hash must match the style element it authorizes.

`_content_security_policy` hashes `_css()` while the document embeds that CSS
in a `<style>` element. A browser computes the hash over the element's exact
text content, so any whitespace, extra block, or inline `style=` attribute the
renderer introduces silently drops every rule -- the CSP is still strict, the
report is simply unstyled. These tests pin that correspondence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from html import unescape

from fast_mlsirm.report import render_diagnostics_report

_CSP_META = re.compile(
    r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">', re.I
)
_STYLE_BLOCK = re.compile(r"<style>(.*?)</style>", re.S | re.I)
_INLINE_STYLE_ATTR = re.compile(r"<[^>]*\sstyle=", re.I)


def _diagnostics_html(tmp_path) -> str:
    """Render one populated diagnostics report and return its HTML."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "itemfit": {
                    "item_id": [0, 1],
                    "outfit_mnsq": [1.0, 1.2],
                    "observed_count": [4, 4],
                },
                "personfit": {},
                "factorfit": {},
                "categoryfit": {},
                "groupfit": {},
                "clusterfit": {},
                "group_itemfit": {},
                "cluster_itemfit": {},
                "model_fit": {"loglik": -3.2, "deviance": 6.4},
            }
        ),
        encoding="utf-8",
    )
    render_diagnostics_report(source, out, title="CSP Contract")
    return out.read_text(encoding="utf-8")


def test_style_source_hash_matches_the_embedded_style_element(tmp_path) -> None:
    """The sha256 source in style-src equals the digest of the style text."""
    html = _diagnostics_html(tmp_path)

    policy_match = _CSP_META.search(html)
    assert policy_match is not None, "report must embed a Content-Security-Policy"
    policy = unescape(policy_match.group(1))

    style_blocks = _STYLE_BLOCK.findall(html)
    assert len(style_blocks) == 1, "exactly one hashed style element is authorized"

    declared = re.search(r"style-src 'sha256-([^']+)'", policy)
    assert declared is not None, "style-src must carry the style element's hash"
    expected = base64.b64encode(
        hashlib.sha256(style_blocks[0].encode("utf-8")).digest()
    ).decode("utf-8")
    assert declared.group(1) == expected


def test_policy_admits_no_inline_style_and_no_unsafe_source(tmp_path) -> None:
    """No element carries a style attribute the hashed policy would block."""
    html = _diagnostics_html(tmp_path)

    policy_match = _CSP_META.search(html)
    assert policy_match is not None
    policy = unescape(policy_match.group(1))
    assert "'unsafe-inline'" not in policy
    assert "'unsafe-hashes'" not in policy

    offender = _INLINE_STYLE_ATTR.search(html)
    assert offender is None, f"inline style attribute is unreachable under {policy!r}"
