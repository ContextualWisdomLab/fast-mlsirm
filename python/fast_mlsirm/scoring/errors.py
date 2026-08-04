"""Structured non-reflective errors for automated-scoring contracts."""

from __future__ import annotations

from typing import NoReturn


class AssessmentSpecError(ValueError):
    """Fail-closed scoring-contract error with stable code and bounded path."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store machine-readable metadata without embedding rejected values."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def contract_error(code: str, path: str, message: str) -> NoReturn:
    """Raise one structured non-reflective scoring-contract error."""
    raise AssessmentSpecError(code, path, message)
