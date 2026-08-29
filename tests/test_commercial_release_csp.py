"""Security regressions for the self-contained commercial release report."""

from __future__ import annotations

import base64
import hashlib

from scripts import build_commercial_release as commercial_release


def test_commercial_release_csp_binds_exact_inline_stylesheet() -> None:
    """The standalone report must authorize only its package-owned CSS bytes."""
    css = commercial_release._report_css()
    digest = hashlib.sha256(css.encode("utf-8")).digest()
    encoded = base64.b64encode(digest).decode("ascii")
    policy = commercial_release._content_security_policy()

    assert "'unsafe-inline'" not in policy
    assert policy.count("style-src ") == 1
    assert f"style-src 'sha256-{encoded}'" in policy
