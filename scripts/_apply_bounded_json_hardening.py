#!/usr/bin/env python3
"""Apply descriptor-safe bounded JSON hardening to repository automation scripts."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "build_benchmark_report.py",
    "build_buyer_packet.py",
    "build_commercial_release.py",
    "build_figma_evidence_sync.py",
    "build_procurement_due_diligence.py",
    "build_release_evidence_index.py",
    "sales_readiness.py",
)

HELPER = r"""\
\"\"\"Descriptor-safe bounded JSON input for repository automation.

The module opens the requested leaf without following symbolic links where the
platform supports that flag, validates the opened descriptor as a stable regular
file, performs one bounded read, validates structural depth without recursion,
then delegates syntax and value construction to :mod:`json`.
\"\"\"

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Final

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_JSON_DEPTH: Final = 128
_READ_CHUNK_BYTES: Final = 64 * 1024
_SAFE_OPEN_ERROR = "JSON input could not be opened as a stable regular file"
_UNSTABLE_PATH_ERROR = "JSON input path changed during the bounded read"


def _positive_limit(value: object, field_name: str) -> int:
    \"\"\"Return ``value`` as a positive, non-Boolean integer.\"\"\"
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _open_flags() -> int:
    \"\"\"Return portable descriptor flags for non-following, nonblocking reads.\"\"\"
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _descriptor_identity(file_status: os.stat_result) -> tuple[int, int]:
    \"\"\"Return the device and inode identity for one stat result.\"\"\"
    return file_status.st_dev, file_status.st_ino


def _validate_path_identity(path: Path, descriptor_status: os.stat_result) -> None:
    \"\"\"Require ``path`` to still name the regular file held by the descriptor.\"\"\"
    try:
        path_status = os.lstat(path)
    except OSError as exc:
        raise ValueError(_UNSTABLE_PATH_ERROR) from None
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError(_SAFE_OPEN_ERROR)
    if _descriptor_identity(path_status) != _descriptor_identity(descriptor_status):
        raise ValueError(_UNSTABLE_PATH_ERROR)


def _read_bounded_descriptor(file_descriptor: int, *, byte_limit: int) -> bytes:
    \"\"\"Read at most ``byte_limit + 1`` bytes from an opened descriptor.\"\"\"
    remaining = byte_limit + 1
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(file_descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > byte_limit:
        raise ValueError(f"JSON input exceeds maximum allowed size {byte_limit} bytes")
    return content


def _read_stable_regular_file(path: Path, *, byte_limit: int) -> bytes:
    \"\"\"Open, identify, and bounded-read one stable regular file.\"\"\"
    try:
        file_descriptor = os.open(path, _open_flags())
    except OSError as exc:
        raise ValueError(_SAFE_OPEN_ERROR) from None
    try:
        descriptor_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise ValueError(_SAFE_OPEN_ERROR)
        _validate_path_identity(path, descriptor_status)
        content = _read_bounded_descriptor(file_descriptor, byte_limit=byte_limit)
        _validate_path_identity(path, descriptor_status)
        return content
    finally:
        os.close(file_descriptor)


def _validate_json_depth(content: bytes, *, max_depth: int) -> None:
    \"\"\"Reject object or array nesting deeper than ``max_depth``.\"\"\"
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
                    f"JSON input exceeds maximum allowed depth {max_depth}"
                )
        elif byte in (0x5D, 0x7D) and depth:
            depth -= 1


def read_json_object(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
) -> dict[str, Any]:
    \"\"\"Read a stable, bounded UTF-8 JSON object from ``path``.

    Args:
        path: Leaf file to open without following a symbolic link where
            supported by the operating system.
        max_bytes: Inclusive maximum number of bytes accepted.
        max_depth: Inclusive maximum object/array nesting depth.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If limits are invalid, the input is not a stable regular
            file, the byte/depth limit is exceeded, or decoder recursion fails.
        ValueError: Also raised when the input is not valid UTF-8 or JSON.
        RuntimeError: If the decoded root is not an object.
    \"\"\"
    byte_limit = _positive_limit(max_bytes, "max_bytes")
    depth_limit = _positive_limit(max_depth, "max_depth")
    content = _read_stable_regular_file(path, byte_limit=byte_limit)
    _validate_json_depth(content, max_depth=depth_limit)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("JSON input is not valid UTF-8") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("JSON input is not valid JSON") from None
    except RecursionError as exc:
        raise ValueError("JSON input exceeds decoder recursion capacity") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("JSON artifact must be an object")
    return payload


__all__ = ["MAX_JSON_BYTES", "MAX_JSON_DEPTH", "read_json_object"]
"""

TESTS = r"""\
\"\"\"Security, invariance, and delegation tests for bounded JSON readers.\"\"\"

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from scripts import _bounded_json
from scripts._bounded_json import read_json_object

_SCRIPT_MODULES = (
    "scripts.build_benchmark_report",
    "scripts.build_buyer_packet",
    "scripts.build_commercial_release",
    "scripts.build_figma_evidence_sync",
    "scripts.build_procurement_due_diligence",
    "scripts.build_release_evidence_index",
    "scripts.sales_readiness",
)


def _write(path: Path, content: bytes) -> Path:
    \"\"\"Write exact bytes and return ``path``.\"\"\"
    path.write_bytes(content)
    return path


def test_exact_size_depth_strings_escapes_and_unicode(tmp_path: Path) -> None:
    \"\"\"Exact limits accept valid objects without counting string delimiters.\"\"\"
    content = json.dumps(
        {
            "text": "[{}] \\\\\\\"quoted\\\\\\\" \\\\\\\\ slash",
            "nested": {"items": ["한글"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    path = _write(tmp_path / "valid.json", content)
    assert read_json_object(
        path,
        max_bytes=len(content),
        max_depth=3,
    ) == json.loads(content)


def test_limit_plus_one_is_rejected(tmp_path: Path) -> None:
    \"\"\"One byte above the configured inclusive byte limit fails closed.\"\"\"
    path = _write(tmp_path / "oversized.json", b'{"a":1} ')
    with pytest.raises(ValueError, match="maximum allowed size 7 bytes"):
        read_json_object(path, max_bytes=7)


def test_depth_boundary_and_excess(tmp_path: Path) -> None:
    \"\"\"The depth boundary is inclusive and the next level is rejected.\"\"\"
    boundary = _write(tmp_path / "boundary.json", b'{"a":{"b":0}}')
    excessive = _write(tmp_path / "excessive.json", b'{"a":{"b":[]}}')
    assert read_json_object(boundary, max_depth=2) == {"a": {"b": 0}}
    with pytest.raises(ValueError, match="maximum allowed depth 2"):
        read_json_object(excessive, max_depth=2)


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
    \"\"\"Invalid limit configuration fails before JSON decoding.\"\"\"
    path = _write(tmp_path / "input.json", b"{}")
    with pytest.raises(ValueError, match=f"{keyword} must be a positive integer"):
        read_json_object(path, **{keyword: value})


def test_invalid_utf8_malformed_json_and_decoder_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    \"\"\"Encoding, syntax, and recursion failures remain deterministic.\"\"\"
    invalid_utf8 = _write(tmp_path / "invalid-utf8.json", b'{"value":"\\xff"}')
    malformed = _write(tmp_path / "malformed.json", b'}{"value":1}')
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_json_object(invalid_utf8)
    with pytest.raises(ValueError, match="not valid JSON"):
        read_json_object(malformed)

    valid = _write(tmp_path / "valid-object.json", b"{}")

    def raise_recursion(_: str) -> object:
        raise RecursionError("decoder stack exhausted")

    monkeypatch.setattr(_bounded_json.json, "loads", raise_recursion)
    with pytest.raises(ValueError, match="decoder recursion capacity"):
        read_json_object(valid)


@pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
def test_every_wrapper_rejects_non_object_json(
    module_name: str,
    tmp_path: Path,
) -> None:
    \"\"\"Each script preserves its object-root contract through delegation.\"\"\"
    module = importlib.import_module(module_name)
    path = _write(tmp_path / f"{module_name.rsplit('.', 1)[-1]}.json", b"[]")
    with pytest.raises(RuntimeError, match="must be an object"):
        module._read_json(path)


@pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
def test_every_wrapper_delegates_to_shared_reader(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    \"\"\"Every governed reader delegates without duplicating parsing logic.\"\"\"
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


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    \"\"\"A symbolic-link leaf is never followed on supporting platforms.\"\"\"
    target = _write(tmp_path / "target.json", b"{}")
    link = tmp_path / "input.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(link)


def test_directory_is_rejected(tmp_path: Path) -> None:
    \"\"\"A directory descriptor is rejected as nonregular input.\"\"\"
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    \"\"\"A FIFO is opened nonblocking and rejected before any content read.\"\"\"
    fifo = tmp_path / "input.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(fifo)


def test_descriptor_path_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    \"\"\"Replacing the path after descriptor open is detected deterministically.\"\"\"
    path = _write(tmp_path / "input.json", b"{}")
    replacement = _write(tmp_path / "replacement.json", b'{"replacement":true}')
    real_read = _bounded_json.os.read
    replaced = False

    def replacing_read(file_descriptor: int, byte_count: int) -> bytes:
        nonlocal replaced
        chunk = real_read(file_descriptor, byte_count)
        if not replaced:
            replaced = True
            path.unlink()
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(_bounded_json.os, "read", replacing_read)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)


def test_missing_path_and_identity_lookup_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    \"\"\"Open and post-open identity failures use content-free errors.\"\"\"
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(tmp_path / "missing.json")

    path = _write(tmp_path / "input.json", b"{}")
    real_lstat = _bounded_json.os.lstat
    calls = 0

    def failing_lstat(value: Path) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileNotFoundError
        return real_lstat(value)

    monkeypatch.setattr(_bounded_json.os, "lstat", failing_lstat)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)


def test_initial_identity_mismatch_and_nonregular_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    \"\"\"Identity mismatch and a nonregular lstat result are rejected.\"\"\"
    path = _write(tmp_path / "input.json", b"{}")
    other = _write(tmp_path / "other.json", b"{}")
    other_status = os.lstat(other)
    monkeypatch.setattr(_bounded_json.os, "lstat", lambda _: other_status)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)

    class NonRegularStatus:
        st_mode = 0
        st_dev = other_status.st_dev
        st_ino = other_status.st_ino

    monkeypatch.setattr(_bounded_json.os, "lstat", lambda _: NonRegularStatus())
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(path)
"""

DOCS = r"""\
# Descriptor-safe bounded JSON input

Repository automation treats JSON artifacts as untrusted local or CI inputs. All
governed readers delegate to `scripts._bounded_json.read_json_object`.

The loader:

1. opens the leaf with `O_CLOEXEC`, `O_NONBLOCK`, and `O_NOFOLLOW` when those
   platform flags are available;
2. validates the opened descriptor with `fstat` as a regular file;
3. compares the descriptor identity with `lstat` before and after the read;
4. reads at most 32 MiB plus one byte through that same descriptor;
5. scans structural nesting non-recursively with an inclusive depth limit of 128,
   ignoring delimiters inside strings and escaped characters;
6. decodes strict UTF-8, delegates syntax/value construction to `json.loads`, and
   requires an object root.

These controls bound availability risk and reject symbolic links, FIFOs,
directories, oversized inputs, excessive nesting, and path replacement. They do
not make arbitrary JSON semantically trustworthy.

## References

MITRE. (2026, April 30). *CWE-400: Uncontrolled resource consumption* (CWE
Version 4.20). https://cwe.mitre.org/data/definitions/400.html

MITRE. (2026, April 30). *CWE-674: Uncontrolled recursion* (CWE Version 4.20).
https://cwe.mitre.org/data/definitions/674.html

Python Software Foundation. (2026). *json—JSON encoder and decoder*. Python
3 documentation. https://docs.python.org/3/library/json.html

Python Software Foundation. (2026). *os—Miscellaneous operating system
interfaces*. Python 3 documentation. https://docs.python.org/3/library/os.html
"""

CHANGELOG = r"""\
# Descriptor-safe bounded JSON input for automation scripts

## Security

- Consolidated governed automation JSON readers behind a descriptor-safe shared
  loader with a 32 MiB inclusive byte bound and a non-recursive 128-level depth
  bound.
- Rejected symbolic links, FIFOs, directories, path replacement, invalid UTF-8,
  malformed JSON, non-object roots, oversized input, and excessive nesting with
  deterministic tests.
"""

IMPORT_BLOCK = (
    "\ntry:\n"
    "    from scripts._bounded_json import read_json_object\n"
    "except ModuleNotFoundError:\n"
    "    from _bounded_json import read_json_object\n"
)
READ_REPLACEMENT = (
    "def _read_json(path: Path) -> dict[str, Any]:\n"
    '    """Read one descriptor-safe bounded JSON object from disk."""\n'
    "    return read_json_object(path)\n"
)


def patch_reader(path: Path) -> None:
    """Replace one local JSON reader with the shared bounded contract."""
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
    """Write the shared loader, wrappers, tests, documentation, and changelog."""
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
