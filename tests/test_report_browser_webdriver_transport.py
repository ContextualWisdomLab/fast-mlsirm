"""Security contract for the report-browser WebDriver transport."""

from __future__ import annotations

import json
from typing import Any

import scripts.verify_report_browser_e2e as browser_e2e


class _FakeResponse:
    status = 200
    reason = "OK"

    def read(self) -> bytes:
        return json.dumps({"value": {"ready": True}}).encode("utf-8")


class _FakeConnection:
    calls: list[dict[str, Any]] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        assert host == "127.0.0.1"
        assert port == 9515
        assert timeout == 30

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.calls.append(
            {"method": method, "path": path, "body": body, "headers": headers}
        )

    def getresponse(self) -> _FakeResponse:
        return _FakeResponse()

    def close(self) -> None:
        return None


def test_webdriver_transport_is_fixed_to_loopback_http(monkeypatch) -> None:
    """The verifier must not expose ChromeDriver through a generic URL opener."""
    assert not hasattr(browser_e2e, "urlopen")
    monkeypatch.setattr(browser_e2e, "HTTPConnection", _FakeConnection)
    _FakeConnection.calls.clear()

    session = object.__new__(browser_e2e.ChromeSession)
    session._port = 9515

    result = session._request("GET", "/status")

    assert result == {"value": {"ready": True}}
    assert _FakeConnection.calls == [
        {
            "method": "GET",
            "path": "/status",
            "body": None,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
        }
    ]
