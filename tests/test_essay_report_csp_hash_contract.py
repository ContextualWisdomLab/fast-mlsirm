"""Regression contracts for standalone-report CSP style hashes."""

from __future__ import annotations

import ast
import base64
import hashlib
import inspect

import fast_mlsirm.report as general_report
from fast_mlsirm.scoring.essay import (
    calibration_report_html,
    report_html,
    validation_report_html,
)


def _expected_hash(css: str) -> str:
    """Return the CSP SHA-256 source expression for exact inline CSS bytes."""
    digest = hashlib.sha256(css.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def test_report_csp_helpers_bind_exact_inline_css() -> None:
    """Both report families must authorize only the exact emitted style bytes."""
    for module in (general_report, report_html):
        css = module._css()
        policy = module._content_security_policy(css)
        assert "'unsafe-inline'" not in policy
        assert _expected_hash(css) in policy


def test_general_report_bar_chart_uses_csp_safe_value_markup() -> None:
    """Data-dependent bar widths must not rely on blocked style attributes."""
    chart = general_report._bar_chart(
        [
            {"item_id": "item-a", "outfit_mnsq": 0.5},
            {"item_id": "item-b", "outfit_mnsq": 1.5},
        ],
        "outfit_mnsq",
    )
    assert " style=" not in chart
    assert chart.count('<progress class="bar-track"') == 2
    assert 'max="100"' in chart
    assert 'value="8.0"' in chart
    assert 'value="100.0"' in chart


def test_shared_essay_renderers_pass_css_to_csp_helper() -> None:
    """Shared essay renderers must follow the CSP helper's CSS-bound signature."""
    for module in (validation_report_html, calibration_report_html):
        tree = ast.parse(inspect.getsource(module))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_content_security_policy"
        ]
        assert calls, f"{module.__name__} must emit a Content-Security-Policy"
        for call in calls:
            assert len(call.args) == 1
            arg = call.args[0]
            assert isinstance(arg, ast.Call)
            assert isinstance(arg.func, ast.Name)
            assert arg.func.id == "_css"
            assert not arg.args
            assert not arg.keywords
