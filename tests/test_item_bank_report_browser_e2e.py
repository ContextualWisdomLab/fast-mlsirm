"""Real-browser evidence for the governed item-bank HTML report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shutil

from fast_mlsirm.rubric.item_bank_report import render_item_bank_report_html
from scripts.verify_report_browser_e2e import (
    ChromeSession,
    _assert_skip_link_focus,
    _assert_skip_target_focus,
    _focus_snapshot,
    _sha256,
    _source_commit,
)

_WEBDRIVER_ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)


def _assert_report_shell(
    session: ChromeSession,
    width: int,
    height: int,
) -> dict[str, object]:
    """Require the standalone report shell to remain inside the viewport."""
    session.set_viewport(width, height)
    snapshot = session.execute(
        """
        const main = document.querySelector('main#main-content');
        if (!main) return null;
        const rect = main.getBoundingClientRect();
        return {
          documentScrollWidth: document.documentElement.scrollWidth,
          viewportWidth: innerWidth,
          mainLeft: rect.left,
          mainRight: rect.right,
          captionCount: document.querySelectorAll('table caption').length,
          headingCount: document.querySelectorAll('main h1, main h2').length,
        };
        """
    )
    assert isinstance(snapshot, dict)
    assert snapshot["documentScrollWidth"] <= snapshot["viewportWidth"] + 1
    assert snapshot["mainLeft"] >= -1
    assert snapshot["mainRight"] <= snapshot["viewportWidth"] + 1
    assert snapshot["captionCount"] == 2
    assert snapshot["headingCount"] >= 4
    return snapshot


def _pointer_click_main(session: ChromeSession) -> None:
    """Focus the main landmark with an explicit WebDriver mouse action sequence."""
    located = session.command(
        "POST",
        "/element",
        {"using": "css selector", "value": "main#main-content"},
    )
    element = located.get("value", {})
    element_id = element.get(_WEBDRIVER_ELEMENT_KEY)
    assert element_id
    session.command(
        "POST",
        "/actions",
        {
            "actions": [
                {
                    "type": "pointer",
                    "id": "mouse",
                    "parameters": {"pointerType": "mouse"},
                    "actions": [
                        {
                            "type": "pointerMove",
                            "duration": 0,
                            "origin": {_WEBDRIVER_ELEMENT_KEY: element_id},
                            "x": 0,
                            "y": 0,
                        },
                        {"type": "pointerDown", "button": 0},
                        {"type": "pointerUp", "button": 0},
                    ],
                }
            ]
        },
    )
    session.command("DELETE", "/actions")


def test_item_bank_report_skip_link_focus_is_browser_verified(tmp_path: Path) -> None:
    """Keyboard bypass and pointer-only suppression must hold in real Chrome."""
    repo_root = Path(__file__).parents[1]
    lifecycle = _REPORT_FIXTURES["_lifecycle"]()
    report_path = tmp_path / "item-bank-report.html"
    report_path.write_text(
        render_item_bank_report_html(lifecycle, title="Item bank browser evidence"),
        encoding="utf-8",
    )

    chromedriver = shutil.which("chromedriver")
    assert chromedriver is not None, "chromedriver is required for browser evidence"
    session = ChromeSession(chromedriver, tmp_path)
    evidence: dict[str, object] = {
        "status": "failed",
        "source_commit": _source_commit(repo_root),
        "report_sha256": _sha256(report_path),
        "layouts": [],
    }
    try:
        evidence["browser_version"] = session.browser_version
        session.set_viewport(1280, 900)
        session.navigate(report_path)
        session.execute("if (document.activeElement) document.activeElement.blur();")

        session.press_tab()
        skip_link = _focus_snapshot(session)
        _assert_skip_link_focus(skip_link)
        evidence["skip_link"] = skip_link

        session.press_enter()
        skip_target = _focus_snapshot(session)
        _assert_skip_target_focus(skip_target)
        evidence["skip_target"] = skip_target

        # Reacquire focus after blur so the assertion observes a pointer-acquired
        # focus state rather than the keyboard-acquired state from the skip link.
        session.execute("if (document.activeElement) document.activeElement.blur();")
        _pointer_click_main(session)
        pointer_target = _focus_snapshot(session)
        assert pointer_target["tag"] == "MAIN"
        assert pointer_target["id"] == "main-content"
        assert pointer_target["focusVisible"] is False
        assert pointer_target["outlineStyle"] == "none"
        evidence["pointer_target"] = pointer_target

        layouts: list[dict[str, object]] = []
        for width, height in ((375, 800), (768, 900), (1280, 900)):
            snapshot = _assert_report_shell(session, width, height)
            layouts.append({"width": width, "height": height, **snapshot})
        evidence["layouts"] = layouts
        evidence["status"] = "ok"
    finally:
        session.close()
        evidence_out = os.environ.get("ITEM_BANK_BROWSER_EVIDENCE_OUT")
        if evidence_out:
            Path(evidence_out).write_text(
                json.dumps(evidence, indent=2, sort_keys=True),
                encoding="utf-8",
            )
