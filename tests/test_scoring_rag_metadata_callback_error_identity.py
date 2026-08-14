"""Fail-first contracts for caller-forged package exception identities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import runpy
from typing import Any

from fast_mlsirm.scoring import AssessmentSpecError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_rag_metadata_callback_safety.py"))
)
_request = _FIXTURES["_request"]
_SECRET = _FIXTURES["_SECRET"]


class _HostileString(str):
    """Raise caller-controlled text from a string validation callback."""

    def strip(self, chars: str | None = None) -> str:
        del chars
        raise RuntimeError(_SECRET)


class _LatePackageErrorKeyTrap(Mapping[str, Any]):
    """Forge a package error after yielding one authorized key."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        yield "evaluation_split"
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata",
            _SECRET,
        )

    def __len__(self) -> int:
        return 2


class _PackageErrorValueTrap(Mapping[str, Any]):
    """Forge a package error from an authorized value callback."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata",
            _SECRET,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split",))

    def __len__(self) -> int:
        return 1


class _HostileStringKeyTrap(Mapping[str, Any]):
    """Expose one string subclass whose validator callback fails."""

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        return iter((_HostileString("evaluation_split"),))

    def __len__(self) -> int:
        return 1


def test_late_caller_package_error_does_not_escape_key_iteration() -> None:
    """Caller-originated package errors remain untrusted callback failures."""
    try:
        _request(_LatePackageErrorKeyTrap())
    except AssessmentSpecError as caught:
        assert caught.code == "invalid_rag_metadata"
        assert _SECRET not in str(caught)
    else:  # pragma: no cover - fail-first assertion aid
        raise AssertionError("hostile key iteration unexpectedly succeeded")


def test_caller_package_error_does_not_escape_value_capture() -> None:
    """Authorized value callbacks cannot forge trusted package evidence."""
    try:
        _request(_PackageErrorValueTrap())
    except AssessmentSpecError as caught:
        assert caught.code == "invalid_rag_metadata"
        assert _SECRET not in str(caught)
    else:  # pragma: no cover - fail-first assertion aid
        raise AssertionError("hostile value capture unexpectedly succeeded")


def test_string_subclass_callback_does_not_escape_key_validation() -> None:
    """A hostile string subclass cannot reflect text from key validation."""
    try:
        _request(_HostileStringKeyTrap())
    except AssessmentSpecError as caught:
        assert caught.code == "invalid_rag_metadata"
        assert _SECRET not in str(caught)
    else:  # pragma: no cover - fail-first assertion aid
        raise AssertionError("hostile string key unexpectedly succeeded")
