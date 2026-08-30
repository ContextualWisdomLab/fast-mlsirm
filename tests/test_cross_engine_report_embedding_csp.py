"""Security regression for cross-engine report embedding policy."""

from __future__ import annotations

from fast_mlsirm.cross_engine_report import _content_security_policy


def test_cross_engine_report_csp_denies_hostile_embedding() -> None:
    """The standalone report must not be frameable by another origin."""
    csp = _content_security_policy()

    assert "default-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "form-action 'none'" in csp
    assert "frame-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "script-src 'none'" in csp
    assert "style-src 'sha256-" in csp
    assert "unsafe-inline" not in csp
