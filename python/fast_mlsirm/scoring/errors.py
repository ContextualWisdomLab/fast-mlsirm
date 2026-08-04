"""Structured redacted errors for automated-scoring contracts."""

from __future__ import annotations

from ..rubric.models import _identifier, _text


class ScoringContractError(ValueError):
    """Stable rejection carrying a machine-readable code and JSON-style path."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store bounded metadata without embedding assessment response content."""
        self.code = _identifier(code, "code")
        if not isinstance(path, str) or not path.startswith("$"):
            raise ValueError("path must be a JSON-style path beginning with '$'")
        self.path = path
        self.message = _text(message, "message", maximum=512)
        super().__init__(f"{self.code} at {self.path}: {self.message}")


def contract_error(code: str, path: str, message: str) -> ScoringContractError:
    """Return one structured scoring-contract error."""
    return ScoringContractError(code, path, message)
