"""Real-browser evidence for the governed item-bank HTML report."""

from __future__ import annotations

from pathlib import Path
import runpy
import shutil

from fast_mlsirm.rubric.item_bank_report import render_item_bank_report_html
from scripts.verify_report_browser_e2e import (
    ChromeSession,
    _assert_skip_link_focus,
    _assert_skip_target_focus,
    _focus_snapshot,
)

_WEBDRIVER_ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)


def _assert_report_shell(session: ChromeSession, width: int, height: int) -> None:
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


def _click_body(session: ChromeSession) -> None:
    """Establish pointer input modality without inventing a product control."""
    located = session.command(
        "POST", "/element", {"using": "css selector", "value": "body"}
    )
    element = located.get("value", {})
    element_id = element.get(_WEBDRIVER_ELEMENT_KEY)
    assert element_id
    session.command("POST", f"/element/{element_id}/click", {})


def test_item_bank_report_skip_link_focus_is_browser_verified(tmp_path: Path) -> None:
    """Keyboard bypass and pointer-only suppression must hold in real Chrome."""
    lifecycle = _REPORT_FIXTURES["_lifecycle"]()
    report_path = tmp_path / "item-bank-report.html"
    report_path.write_text(
        render_item_bank_report_html(lifecycle, title="Item bank browser evidence"),
        encoding="utf-8",
    )

    chromedriver = shutil.which("chromedriver")
    assert chromedriver is not None, "chromedriver is required for browser evidence"
    session = ChromeSession(chromedriver, tmp_path)
    try:
        session.set_viewport(1280, 900)
        session.navigate(report_path)
        session.execute("if (document.activeElement) document.activeElement.blur();")

        session.press_tab()
        skip_link = _focus_snapshot(session)
        _assert_skip_link_focus(skip_link)

        session.press_enter()
        skip_target = _focus_snapshot(session)
        _assert_skip_target_focus(skip_target)

        _click_body(session)
        session.execute("document.querySelector('main#main-content').focus();")
        pointer_programmatic_target = _focus_snapshot(session)
        assert pointer_programmatic_target["tag"] == "MAIN"
        assert pointer_programmatic_target["id"] == "main-content"
        assert pointer_programmatic_target["focusVisible"] is False
        assert pointer_programmatic_target["outlineStyle"] == "none"

        for width, height in ((375, 800), (768, 900), (1280, 900)):
            _assert_report_shell(session, width, height)
    finally:
        session.close()
