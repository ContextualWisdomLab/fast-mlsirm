#!/usr/bin/env python
"""Verify commercial report focus and overflow behavior in a real browser."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import time
from http.client import HTTPConnection, HTTPException
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_report import _render_report_html as render_benchmark_report
from scripts.build_buyer_packet import _render_report_html as render_buyer_report
from scripts.build_commercial_release import _render_html as render_release_report
from scripts.build_figma_evidence_sync import _render_report as render_figma_report
from scripts.build_pr_queue_governance import _render_report as render_pr_queue_report
from scripts.build_procurement_due_diligence import _render_report as render_due_diligence_report
from scripts.build_release_evidence_index import _render_report_html as render_release_index

_WEBDRIVER_ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
_TAB = "\ue004"
_ENTER = "\ue007"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one generated report artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit(repo_root: Path) -> str:
    """Return the exact checked-out Git commit used to render browser evidence."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    candidate = completed.stdout.strip()
    if len(candidate) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise RuntimeError("git rev-parse returned an invalid source commit")
    return candidate


def _free_port() -> int:
    """Reserve and release one loopback port for the short-lived ChromeDriver."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ChromeSession:
    """Small dependency-free W3C WebDriver client for GitHub-hosted Chrome."""

    def __init__(self, chromedriver: str, work_dir: Path) -> None:
        self._port = _free_port()
        self._driver_log = work_dir / "chromedriver.log"
        self._process = subprocess.Popen(
            [
                chromedriver,
                f"--port={self._port}",
                f"--log-path={self._driver_log}",
                "--log-level=INFO",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait_until_ready()
        payload = self._request(
            "POST",
            "/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "chrome",
                        "goog:chromeOptions": {
                            "args": [
                                "--headless=new",
                                "--no-sandbox",
                                "--disable-dev-shm-usage",
                                "--window-size=1280,900",
                            ]
                        },
                    }
                }
            },
        )
        value = payload.get("value", {})
        self.session_id = str(value.get("sessionId", ""))
        if not self.session_id:
            raise RuntimeError(f"ChromeDriver did not return a session id: {payload}")
        capabilities = value.get("capabilities", {})
        self.browser_version = str(capabilities.get("browserVersion", "unknown"))

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 15
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"ChromeDriver exited before accepting sessions; log={self._driver_log}"
                )
            try:
                status = self._request("GET", "/status")
                if status.get("value", {}).get("ready") is True:
                    return
            except (OSError, HTTPException, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            time.sleep(0.1)
        raise RuntimeError(
            f"ChromeDriver did not become ready; last_error={last_error}; log={self._driver_log}"
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send one WebDriver command only to the loopback ChromeDriver HTTP server."""
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        connection = HTTPConnection("127.0.0.1", self._port, timeout=30)
        try:
            connection.request(  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                method,
                path,
                body=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            response = connection.getresponse()
            decoded = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
        if not isinstance(decoded, dict):
            raise RuntimeError(f"unexpected WebDriver response for {path}: {decoded!r}")
        value = decoded.get("value")
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(f"WebDriver command failed for {path}: {value}")
        return decoded

    def command(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run one command against the active WebDriver session."""
        return self._request(method, f"/session/{self.session_id}{path}", payload)

    def navigate(self, path: Path) -> None:
        """Navigate to one generated local report and wait for document completion."""
        self.command("POST", "/url", {"url": path.resolve().as_uri()})
        state = self.execute("return document.readyState")
        if state != "complete":
            raise AssertionError(f"report document did not finish loading: {state!r}")

    def execute(self, script: str, *args: Any) -> Any:
        """Execute JavaScript in the active report and return its JSON value."""
        response = self.command(
            "POST", "/execute/sync", {"script": script, "args": list(args)}
        )
        return response.get("value")

    def set_viewport(self, width: int, height: int) -> None:
        """Set the top-level browser window to one responsive evidence size."""
        self.command(
            "POST",
            "/window/rect",
            {"x": 0, "y": 0, "width": width, "height": height},
        )

    def press_tab(self) -> None:
        """Dispatch a genuine WebDriver keyboard Tab action."""
        self.command(
            "POST",
            "/actions",
            {
                "actions": [
                    {
                        "type": "key",
                        "id": "keyboard",
                        "actions": [
                            {"type": "keyDown", "value": _TAB},
                            {"type": "keyUp", "value": _TAB},
                        ],
                    }
                ]
            },
        )

    def press_enter(self) -> None:
        """Activate the current focus target through a genuine WebDriver Enter key."""
        self.command(
            "POST",
            "/actions",
            {
                "actions": [
                    {
                        "type": "key",
                        "id": "keyboard",
                        "actions": [
                            {"type": "keyDown", "value": _ENTER},
                            {"type": "keyUp", "value": _ENTER},
                        ],
                    }
                ]
            },
        )

    def click_first_table_wrap(self) -> None:
        """Click the first focusable table region through WebDriver pointer semantics."""
        located = self.command(
            "POST", "/element", {"using": "css selector", "value": ".table-wrap"}
        )
        element = located.get("value", {})
        element_id = element.get(_WEBDRIVER_ELEMENT_KEY)
        if not element_id:
            raise AssertionError("report has no .table-wrap region")
        self.command("POST", f"/element/{element_id}/click", {})

    def close(self) -> None:
        """Delete the WebDriver session and terminate its short-lived server."""
        session_id = getattr(self, "session_id", "")
        if session_id:
            try:
                self._request("DELETE", f"/session/{session_id}")
            except Exception:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


_RENDERERS: tuple[tuple[str, Callable[[dict[str, Any]], str], dict[str, Any]], ...] = (
    (
        "benchmark",
        render_benchmark_report,
        {
            "status": "ok",
            "runtime_budget_seconds": 20,
            "total_duration_seconds": 12.5,
            "budget_ok": True,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "source_commit": "0" * 40,
            "scenario_coverage": {"observed_backends": ["rust"]},
            "artifact_coverage": {
                "required": ["fit.json"],
                "present": ["fit.json"],
            },
            "command_durations": [
                {
                    "index": 1,
                    "command": "fit",
                    "backend": "rust",
                    "duration_seconds": 12.5,
                    "out": "release/fit.json",
                }
            ],
            "caveats": ["Browser evidence fixture for report interaction only."],
        },
    ),
    (
        "buyer",
        render_buyer_report,
        {
            "coverage": {"acceptance_summary": True, "wheel": True},
            "files": [
                {
                    "archive_path": "dist/fast_mlsirm.whl",
                    "size_bytes": 1024,
                    "sha256": "a" * 64,
                }
            ],
            "contract_value_krw": 2_000_000_000,
            "artifact_count": 1,
            "source_commit": "0" * 40,
            "generated_at": "2026-09-09T00:00:00+00:00",
        },
    ),
    (
        "pr_queue",
        render_pr_queue_report,
        {
            "status": "ok",
            "contract_value_krw": 2_000_000_000,
            "repo": "ContextualWisdomLab/fast-mlsirm",
            "base_sha": "0" * 40,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "open_pr_count": 1,
            "risk_counts": {"review_or_check_delay": 0},
            "duplicate_issue_claims": {},
            "pull_requests": [
                {
                    "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1",
                    "number": 1,
                    "title": "Browser evidence fixture",
                    "headRefOid": "0" * 40,
                    "closing_issue_references": [],
                    "reviewDecision": "APPROVED",
                    "mergeStateStatus": "CLEAN",
                    "risk_reasons": [],
                }
            ],
            "changed_file_overlap_warnings": [],
            "issue_claim_history": [],
            "checks": [
                {
                    "name": "report:browser",
                    "category": "accessibility",
                    "ok": True,
                    "detail": "fixture row",
                }
            ],
        },
    ),
    (
        "procurement",
        render_due_diligence_report,
        {
            "status": "ok",
            "contract_value_krw": 2_000_000_000,
            "project": {"version": "0.9.1"},
            "source_commit": "0" * 40,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "checks": [
                {
                    "name": "report:browser",
                    "category": "accessibility",
                    "ok": True,
                    "detail": "fixture row",
                }
            ],
            "failed_checks": [],
        },
    ),
    (
        "release_index",
        render_release_index,
        {
            "status": "ok",
            "contract_value_krw": 2_000_000_000,
            "project_version": "0.9.1",
            "source_commit": "0" * 40,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "coverage": {"acceptance": True, "sbom": True},
            "files": [
                {
                    "role": "wheel",
                    "name": "fast_mlsirm.whl",
                    "size_bytes": 1024,
                    "sha256": "b" * 64,
                }
            ],
            "dist": {"artifacts": [{"name": "fast_mlsirm.whl"}]},
            "failures": [],
        },
    ),
    (
        "commercial_release",
        render_release_report,
        {
            "status": "ok",
            "generated_at": "2026-09-09T00:00:00+00:00",
            "source_commit": "0" * 40,
            "stages": [{"name": "test", "status": "ok", "duration_seconds": 1.5}],
            "artifacts": {"test": "sha256"},
        },
    ),
    (
        "figma_evidence",
        render_figma_report,
        {
            "status": "ok",
            "generated_at": "2026-09-09T00:00:00+00:00",
            "checks": [{"name": "test", "category": "test", "ok": True, "detail": "test"}],
        },
    ),
)


def _focus_snapshot(session: ChromeSession) -> dict[str, Any]:
    """Return computed focus and visibility state for the active element."""
    value = session.execute(
        """
        const e = document.activeElement;
        const r = e.getBoundingClientRect();
        const s = getComputedStyle(e);
        const cx = Math.min(Math.max(r.left + r.width / 2, 0), innerWidth - 1);
        const cy = Math.min(Math.max(r.top + r.height / 2, 0), innerHeight - 1);
        const hit = document.elementFromPoint(cx, cy);
        return {
          tag: e.tagName,
          id: e.id,
          className: e.className,
          href: e.getAttribute('href'),
          focusVisible: e.matches(':focus-visible'),
          outlineStyle: s.outlineStyle,
          outlineWidth: s.outlineWidth,
          ariaLabel: e.getAttribute('aria-label'),
          role: e.getAttribute('role'),
          tabindex: e.getAttribute('tabindex'),
          rect: {left: r.left, right: r.right, top: r.top, bottom: r.bottom},
          viewport: {width: innerWidth, height: innerHeight},
          centerUnobscured: hit === e || e.contains(hit),
        };
        """
    )
    if not isinstance(value, dict):
        raise AssertionError(f"browser returned invalid focus evidence: {value!r}")
    return value


def _layout_snapshot(session: ChromeSession) -> dict[str, Any]:
    """Return responsive overflow evidence for the first table region and document."""
    value = session.execute(
        """
        const e = document.querySelector('.table-wrap');
        const main = document.querySelector('main');
        if (!e || !main) return null;
        const r = e.getBoundingClientRect();
        const mr = main.getBoundingClientRect();
        return {
          tableClientWidth: e.clientWidth,
          tableScrollWidth: e.scrollWidth,
          documentScrollWidth: document.documentElement.scrollWidth,
          viewportWidth: innerWidth,
          regionLeft: r.left,
          regionRight: r.right,
          mainLeft: mr.left,
          mainRight: mr.right,
        };
        """
    )
    if not isinstance(value, dict):
        raise AssertionError("report is missing a main element or .table-wrap region")
    return value


def _outline_width(snapshot: dict[str, Any]) -> float:
    """Parse one computed CSS outline width from browser evidence."""
    try:
        return float(str(snapshot.get("outlineWidth", "0px")).removesuffix("px"))
    except ValueError as exc:
        raise AssertionError(f"invalid focus outline width: {snapshot}") from exc


def _assert_skip_link_focus(snapshot: dict[str, Any]) -> None:
    """Require the benchmark bypass link to be visible under keyboard focus."""
    if snapshot.get("tag") != "A" or "skip-link" not in str(snapshot.get("className", "")):
        raise AssertionError(f"first benchmark Tab did not focus the skip link: {snapshot}")
    if snapshot.get("href") != "#main-content":
        raise AssertionError(f"skip link does not target the main evidence region: {snapshot}")
    if snapshot.get("focusVisible") is not True:
        raise AssertionError(f"skip link is not :focus-visible: {snapshot}")
    if snapshot.get("outlineStyle") == "none" or _outline_width(snapshot) < 3:
        raise AssertionError(f"skip link has no contract-sized visible outline: {snapshot}")
    if snapshot.get("centerUnobscured") is not True:
        raise AssertionError(f"focused skip link is obscured: {snapshot}")
    rect = snapshot["rect"]
    viewport = snapshot["viewport"]
    if not (rect["bottom"] > 0 and rect["top"] < viewport["height"]):
        raise AssertionError(f"focused skip link is outside the viewport: {snapshot}")


def _assert_skip_target_focus(snapshot: dict[str, Any]) -> None:
    """Require keyboard activation to move focus to the benchmark main region."""
    if snapshot.get("tag") != "MAIN" or snapshot.get("id") != "main-content":
        raise AssertionError(f"skip-link activation did not focus main content: {snapshot}")
    if snapshot.get("tabindex") != "-1":
        raise AssertionError(f"main skip target lost its programmatic focus contract: {snapshot}")
    if snapshot.get("focusVisible") is not True:
        raise AssertionError(f"keyboard-focused main target is not :focus-visible: {snapshot}")
    if snapshot.get("outlineStyle") == "none" or _outline_width(snapshot) < 3:
        raise AssertionError(f"main skip target has no contract-sized visible outline: {snapshot}")
    rect = snapshot["rect"]
    viewport = snapshot["viewport"]
    if not (rect["bottom"] > 0 and rect["top"] < viewport["height"]):
        raise AssertionError(f"main skip target is outside the viewport: {snapshot}")


def _assert_keyboard_focus(snapshot: dict[str, Any]) -> None:
    """Require keyboard focus to be visible and not covered at the focused region."""
    if "table-wrap" not in str(snapshot.get("className", "")):
        raise AssertionError(f"Tab did not focus a table region: {snapshot}")
    if snapshot.get("role") != "region" or snapshot.get("tabindex") != "0":
        raise AssertionError(f"focused table region lost semantics: {snapshot}")
    if not snapshot.get("ariaLabel"):
        raise AssertionError(f"focused table region has no accessible label: {snapshot}")
    if snapshot.get("focusVisible") is not True:
        raise AssertionError(f"keyboard focus is not :focus-visible: {snapshot}")
    if snapshot.get("outlineStyle") == "none":
        raise AssertionError(f"keyboard focus has no visible outline: {snapshot}")
    if _outline_width(snapshot) < 3:
        raise AssertionError(f"keyboard focus outline is thinner than contract: {snapshot}")
    if snapshot.get("centerUnobscured") is not True:
        raise AssertionError(f"focused table region is obscured at its center: {snapshot}")
    rect = snapshot["rect"]
    viewport = snapshot["viewport"]
    if not (rect["bottom"] > 0 and rect["top"] < viewport["height"]):
        raise AssertionError(f"focused table region is outside the viewport: {snapshot}")


def _assert_pointer_focus(snapshot: dict[str, Any]) -> None:
    """Require pointer focus without suppressing the underlying focus target semantics."""
    if "table-wrap" not in str(snapshot.get("className", "")):
        raise AssertionError(f"pointer click did not focus the table region: {snapshot}")
    if snapshot.get("focusVisible") is not False:
        raise AssertionError(f"pointer focus unexpectedly matches :focus-visible: {snapshot}")
    if snapshot.get("outlineStyle") != "none":
        raise AssertionError(f"pointer-only focus outline was not suppressed: {snapshot}")


def _assert_mobile_layout(snapshot: dict[str, Any]) -> None:
    """Require table overflow to stay inside the focusable region on a narrow viewport."""
    if snapshot["tableScrollWidth"] <= snapshot["tableClientWidth"]:
        raise AssertionError(f"narrow viewport did not exercise horizontal table overflow: {snapshot}")
    if snapshot["documentScrollWidth"] > snapshot["viewportWidth"] + 1:
        raise AssertionError(f"table overflow escaped to the page viewport: {snapshot}")
    if snapshot["regionLeft"] < -1 or snapshot["regionRight"] > snapshot["viewportWidth"] + 1:
        raise AssertionError(f"focusable table region itself overflows the viewport: {snapshot}")


def _assert_responsive_layout(snapshot: dict[str, Any]) -> None:
    """Require the report shell to stay inside medium and desktop viewports."""
    if snapshot["documentScrollWidth"] > snapshot["viewportWidth"] + 1:
        raise AssertionError(f"report creates page-level horizontal overflow: {snapshot}")
    if snapshot["mainLeft"] < -1 or snapshot["mainRight"] > snapshot["viewportWidth"] + 1:
        raise AssertionError(f"main report shell overflows the viewport: {snapshot}")


def verify_reports(repo_root: Path, out_path: Path, chromedriver: str) -> dict[str, Any]:
    """Render all five commercial reports and verify real-browser interactions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    driver_version = subprocess.run(
        [chromedriver, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    evidence: dict[str, Any] = {
        "status": "failed",
        "source_commit": _source_commit(repo_root),
        "chromedriver": driver_version,
        "reports": [],
    }
    with tempfile.TemporaryDirectory(prefix="fast-mlsirm-report-e2e-") as temp:
        work_dir = Path(temp)
        session = ChromeSession(chromedriver, work_dir)
        evidence["browser_version"] = session.browser_version
        try:
            for name, renderer, manifest in _RENDERERS:
                report_path = work_dir / f"{name}.html"
                report_path.write_text(renderer(manifest), encoding="utf-8")
                record: dict[str, Any] = {
                    "name": name,
                    "sha256": _sha256(report_path),
                }
                session.set_viewport(1280, 900)
                session.navigate(report_path)
                session.execute("if (document.activeElement) document.activeElement.blur();")
                session.press_tab()
                if name == "benchmark":
                    skip_link = _focus_snapshot(session)
                    _assert_skip_link_focus(skip_link)
                    record["skip_link"] = skip_link
                    session.press_enter()
                    skip_target = _focus_snapshot(session)
                    _assert_skip_target_focus(skip_target)
                    record["skip_target"] = skip_target
                    session.press_tab()
                keyboard = _focus_snapshot(session)
                _assert_keyboard_focus(keyboard)
                record["keyboard"] = keyboard

                session.execute("if (document.activeElement) document.activeElement.blur();")
                session.click_first_table_wrap()
                pointer = _focus_snapshot(session)
                _assert_pointer_focus(pointer)
                record["pointer"] = pointer

                session.set_viewport(375, 800)
                mobile = _layout_snapshot(session)
                _assert_mobile_layout(mobile)
                record["mobile"] = mobile

                session.set_viewport(768, 900)
                medium = _layout_snapshot(session)
                _assert_responsive_layout(medium)
                record["medium"] = medium

                session.set_viewport(1280, 900)
                desktop = _layout_snapshot(session)
                _assert_responsive_layout(desktop)
                record["desktop"] = desktop
                evidence["reports"].append(record)
            evidence["status"] = "ok"
        finally:
            session.close()
            evidence["chromedriver_log"] = (
                work_dir / "chromedriver.log"
            ).read_text(encoding="utf-8", errors="replace")[-12000:]
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for browser evidence verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="report-browser-e2e.json")
    parser.add_argument("--chromedriver", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run browser verification and emit the retained evidence path."""
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    chromedriver = args.chromedriver or shutil.which("chromedriver") or ""
    if not chromedriver:
        raise SystemExit("chromedriver is required for real-browser report evidence")
    try:
        evidence = verify_reports(repo_root, out_path, chromedriver)
    except Exception as exc:
        failure = {
            "status": "failed",
            "source_commit": _source_commit(repo_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "source_commit": evidence["source_commit"],
                "browser_version": evidence.get("browser_version"),
                "report_count": len(evidence["reports"]),
                "out": str(out_path),
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
