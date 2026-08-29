"""Data-integrity regressions for bounded repository JSON reads."""

from __future__ import annotations

import pytest

from scripts import _bounded_json as bounded_json


def test_read_json_object_rejects_inplace_same_inode_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-size same-inode rewrite during the read must invalidate the snapshot."""
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"value":1}')
    original_read = bounded_json._read_bounded_descriptor

    def read_then_mutate(file_descriptor: int, *, byte_limit: int) -> bytes:
        content = original_read(file_descriptor, byte_limit=byte_limit)
        inode = path.stat().st_ino
        path.write_bytes(b'{"value":2}')
        assert path.stat().st_ino == inode
        return content

    monkeypatch.setattr(
        bounded_json, "_read_bounded_descriptor", read_then_mutate
    )

    with pytest.raises(
        ValueError, match="JSON input path changed during the bounded read"
    ):
        bounded_json.read_json_object(path)


def test_read_json_object_rejects_rewrite_when_metadata_cannot_observe_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content confirmation must catch rewrites hidden by coarse metadata."""
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"value":1}')
    original_read = bounded_json._read_bounded_descriptor
    initial_status = path.stat()
    read_count = 0

    def read_then_mutate(file_descriptor: int, *, byte_limit: int) -> bytes:
        nonlocal read_count
        content = original_read(file_descriptor, byte_limit=byte_limit)
        read_count += 1
        if read_count == 1:
            path.write_bytes(b'{"value":2}')
            assert path.stat().st_ino == initial_status.st_ino
        return content

    monkeypatch.setattr(
        bounded_json, "_read_bounded_descriptor", read_then_mutate
    )
    monkeypatch.setattr(bounded_json.os, "fstat", lambda _: initial_status)

    with pytest.raises(
        ValueError, match="JSON input path changed during the bounded read"
    ):
        bounded_json.read_json_object(path)


def test_parse_json_bounded_rejects_unencodable_utf8_string() -> None:
    """Literal surrogate code points fail with the package UTF-8 diagnostic."""
    content = '{"value":"' + chr(0xD800) + '"}'

    with pytest.raises(ValueError, match="JSON input is not valid UTF-8") as excinfo:
        bounded_json.parse_json_bounded(content)

    assert type(excinfo.value) is ValueError
