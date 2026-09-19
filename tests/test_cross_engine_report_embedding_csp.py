"""Security regression for cross-engine report CSP delivery semantics."""

from __future__ import annotations

from fast_mlsirm.cross_engine_report import _content_security_policy


def test_cross_engine_report_meta_csp_does_not_claim_http_only_anti_framing() -> None:
    """Meta-delivered CSP must not advertise unsupported frame-ancestor policy."""
    csp = _content_security_policy()

    assert "default-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "form-action 'none'" in csp
    assert "frame-src 'none'" in csp
    assert "frame-ancestors" not in csp
    assert "script-src 'none'" in csp
    assert "style-src 'sha256-" in csp
    assert "unsafe-inline" not in csp
