"""Security regressions for the standalone essay-report CSP."""

import base64
import hashlib

import fast_mlsirm.scoring.essay.report_html as report_html


def test_essay_report_csp_authorizes_only_the_exact_stylesheet() -> None:
    """The portable report must hash its exact package-owned stylesheet bytes."""
    expected_hash = base64.b64encode(
        hashlib.sha256(report_html._css().encode("utf-8")).digest()
    ).decode("ascii")

    policy = report_html._content_security_policy()

    assert f"style-src 'sha256-{expected_hash}'" in policy
    assert "'unsafe-inline'" not in policy
    assert "frame-ancestors" not in policy
