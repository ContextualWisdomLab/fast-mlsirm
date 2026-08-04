#!/usr/bin/env python3
"""Apply the one-time bounded JSON hardening patch to repository scripts."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "build_benchmark_report.py",
    "build_buyer_packet.py",
    "build_commercial_release.py",
    "build_figma_evidence_sync.py",
    "build_pr_queue_governance.py",
    "build_procurement_due_diligence.py",
    "build_release_evidence_index.py",
    "sales_readiness.py",
)

HELPER = '''\
"""Bounded JSON input helpers for repository automation scripts.

The helpers cap bytes and structural nesting before invoking Python's
recursive JSON decoder. Delimiters inside strings are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 128


def _positive_limit(value: object, field_name: str) -> int:
    """Return one strictly positive, non-Boolean integer limit."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_json_depth(
    content: bytes,
    *,
    path: Path,
    max_depth: int,
) -> None:
    """Reject JSON whose object or array nesting exceeds ``max_depth``."""
    depth = 0
    in_string = False
    escaped = False
    for byte in content:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    "JSON nesting exceeds maximum allowed depth "
                    f"{max_depth}: {path}"
                )
        elif byte in (0x5D, 0x7D) and depth:
            depth -= 1


def read_json_object(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
) -> dict[str, Any]:
    """Read one size- and depth-bounded UTF-8 JSON object from ``path``."""
    byte_limit = _positive_limit(max_bytes, "max_bytes")
    depth_limit = _positive_limit(max_depth, "max_depth")
    with path.open("rb") as handle:
        content = handle.read(byte_limit + 1)
    if len(content) > byte_limit:
        raise ValueError(
            f"JSON file exceeds maximum allowed size {byte_limit} bytes: {path}"
        )
    _validate_json_depth(content, path=path, max_depth=depth_limit)
    try:
        payload = json.loads(content.decode("utf-8"))
    except RecursionError as exc:
        raise ValueError(
            f"JSON nesting exceeds decoder recursion capacity: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


__all__ = ["MAX_JSON_BYTES", "MAX_JSON_DEPTH", "read_json_object"]
'''

TESTS = '''\
"""Security and delegation tests for bounded script JSON readers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from scripts import _bounded_json
from scripts._bounded_json import read_json_object

_SCRIPT_MODULES = (
    "scripts.build_benchmark_report",
    "scripts.build_buyer_packet",
    "scripts.build_commercial_release",
    "scripts.build_figma_evidence_sync",
    "scripts.build_pr_queue_governance",
    "scripts.build_procurement_due_diligence",
    "scripts.build_release_evidence_index",
    "scripts.sales_readiness",
)


def _write(path: Path, content: bytes) -> Path:
    """Write exact bytes and return the resulting path."""
    path.write_bytes(content)
    return path


def test_valid_object_preserves_delimiters_escapes_and_boundary(
    tmp_path: Path,
) -> None:
    """String delimiters, escapes, Unicode, and exact limits remain valid."""
    content = json.dumps(
        {"text": "[{}] \\\"quoted\\\" \\\\ slash", "nested": {"items": ["한글"]}},
        ensure_ascii=False,
    ).encode("utf-8")
    path = _write(tmp_path / "valid.json", content)
    assert read_json_object(
        path,
        max_bytes=len(content),
        max_depth=3,
    ) == json.loads(content)


def test_depth_boundary_and_excessive_nesting(tmp_path: Path) -> None:
    """Depth scanning accepts the boundary and rejects the next level."""
    boundary = _write(tmp_path / "boundary.json", b'{"a":{"b":0}}')
    excessive = _write(tmp_path / "excessive.json", b'{"a":{"b":[]}}')
    assert read_json_object(boundary, max_depth=2) == {"a": {"b": 0}}
    with pytest.raises(ValueError, match="maximum allowed depth 2"):
        read_json_object(excessive, max_depth=2)


def test_malformed_closer_remains_a_json_syntax_error(tmp_path: Path) -> None:
    """An unmatched closer cannot corrupt the depth counter."""
    path = _write(tmp_path / "malformed.json", b'}{"value": 1}')
    with pytest.raises(json.JSONDecodeError):
        read_json_object(path)


def test_size_boundary_and_oversized_rejection(tmp_path: Path) -> None:
    """The bounded read admits the exact limit and rejects one extra byte."""
    exact = _write(tmp_path / "exact.json", b'{"a":1}')
    oversized = _write(tmp_path / "oversized.json", b'{"a":1} ')
    assert read_json_object(exact, max_bytes=7) == {"a": 1}
    with pytest.raises(ValueError, match="maximum allowed size 7 bytes"):
        read_json_object(oversized, max_bytes=7)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("max_bytes", 0),
        ("max_bytes", True),
        ("max_bytes", "7"),
        ("max_depth", -1),
        ("max_depth", True),
        ("max_depth", 1.5),
    ),
)
def test_limits_require_positive_non_boolean_integers(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    """Invalid resource-limit configuration fails before decoding."""
    path = _write(tmp_path / "value.json", b"{}")
    with pytest.raises(ValueError, match=f"{keyword} must be a positive integer"):
        read_json_object(path, **{keyword: value})


def test_invalid_utf8_and_non_object_roots_fail_closed(tmp_path: Path) -> None:
    """UTF-8 and object-root contracts remain strict."""
    invalid_utf8 = _write(tmp_path / "invalid.json", b'{"x":"\\xff"}')
    array_root = _write(tmp_path / "array.json", b"[]")
    with pytest.raises(UnicodeDecodeError):
        read_json_object(invalid_utf8)
    with pytest.raises(RuntimeError, match="must be an object"):
        read_json_object(array_root)


def test_decoder_recursion_error_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decoder recursion remains a deterministic validation error."""
    path = _write(tmp_path / "object.json", b"{}")

    def raise_recursion(_: str) -> object:
        raise RecursionError("decoder stack exhausted")

    monkeypatch.setattr(_bounded_json.json, "loads", raise_recursion)
    with pytest.raises(ValueError, match="decoder recursion capacity"):
        read_json_object(path)


@pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
def test_script_readers_delegate_to_shared_contract(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every governed script delegates to the same bounded reader."""
    module = importlib.import_module(module_name)
    requested = tmp_path / "artifact.json"
    marker = {"bounded": True}
    observed: list[Path] = []

    def fake_reader(path: Path) -> dict[str, bool]:
        observed.append(path)
        return marker

    monkeypatch.setattr(module, "read_json_object", fake_reader)
    assert module._read_json(requested) is marker
    assert observed == [requested]
'''

DOCS = '''\
# Bounded JSON input security

Repository automation scripts treat JSON artifacts as untrusted input. Every governed reader delegates to `scripts._bounded_json.read_json_object`, which enforces a bounded 32 MiB read and a non-recursive maximum structural depth of 128 before decoding. The scanner tracks string and escape state, so brackets inside JSON strings do not affect depth.

The standard-library decoder remains authoritative for syntax. Strict UTF-8 decoding and object-root validation are preserved. These controls limit availability risk; they do not make arbitrary JSON trustworthy.

## References

MITRE. (2026, April 30). *CWE-674: Uncontrolled recursion* (CWE Version 4.20). https://cwe.mitre.org/data/definitions/674.html

Python Software Foundation. (2026). *json—JSON encoder and decoder* (Python 3.12.13 documentation). https://docs.python.org/3.12/library/json.html
'''

CHANGELOG = '''\
# Bounded JSON input hardening for automation scripts

## Security

- Consolidated automation JSON reads behind a shared helper with a 32 MiB byte limit, a non-recursive 128-level depth limit, strict UTF-8 decoding, and object-root validation.
- Added deterministic boundary, malformed-input, recursion, and delegation coverage for every affected automation script.
'''

IMPORT_BLOCK = (
    "\ntry:\n"
    "    from scripts._bounded_json import read_json_object\n"
    "except ModuleNotFoundError:\n"
    "    from _bounded_json import read_json_object\n"
)
READ_REPLACEMENT = (
    "def _read_json(path: Path) -> dict[str, Any]:\n"
    "    \"\"\"Read a size- and depth-bounded JSON object from disk.\"\"\"\n"
    "    return read_json_object(path)\n"
)


def patch_reader(path: Path) -> None:
    """Replace one local JSON reader with delegation to the shared contract."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    target = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_read_json"
    )
    lines = text.splitlines(keepends=True)
    lines[target.lineno - 1 : target.end_lineno] = [READ_REPLACEMENT]
    text = "".join(lines)
    text = text.replace("import json\n", "")
    text = text.replace("_MAX_JSON_BYTES = 32 * 1024 * 1024\n", "")
    if "from scripts._bounded_json import read_json_object" not in text:
        tree = ast.parse(text)
        import_end = max(
            node.end_lineno
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and node.end_lineno is not None
        )
        lines = text.splitlines(keepends=True)
        lines.insert(import_end, IMPORT_BLOCK)
        text = "".join(lines)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Write shared security contracts, tests, documentation, and readers."""
    (ROOT / "scripts" / "_bounded_json.py").write_text(HELPER, encoding="utf-8")
    for name in SCRIPT_NAMES:
        patch_reader(ROOT / "scripts" / name)
    (ROOT / "tests" / "test_bounded_json_scripts.py").write_text(
        TESTS,
        encoding="utf-8",
    )
    (ROOT / "docs" / "bounded_json_input_security.md").write_text(
        DOCS,
        encoding="utf-8",
    )
    (ROOT / "docs" / "changelog.d" / "bounded-json-input-hardening.md").write_text(
        CHANGELOG,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
