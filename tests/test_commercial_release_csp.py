"""Security regressions for the self-contained commercial release report."""

from __future__ import annotations

import base64
import hashlib

from scripts import build_commercial_release as commercial_release


def test_commercial_release_csp_binds_exact_rendered_stylesheet() -> None:
    """The CSP hash must authorize the exact bytes inside the style element."""
    rendered = commercial_release._render_html({"stages": [], "artifacts": {}})
    style_text = rendered.split("<style>", 1)[1].split("</style>", 1)[0]
    digest = hashlib.sha256(style_text.encode("utf-8")).digest()
    encoded = base64.b64encode(digest).decode("ascii")
    policy = commercial_release._content_security_policy()

    assert style_text == commercial_release._report_css()
    assert "'unsafe-inline'" not in policy
    assert policy.count("style-src ") == 1
    assert f"style-src 'sha256-{encoded}'" in policy
