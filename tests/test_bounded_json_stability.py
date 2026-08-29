"""Data-integrity regressions for bounded repository JSON reads."""

from __future__ import annotations

import pytest

from scripts import _bounded_json as bounded_json


def test_read_json_object_rejects_inplace_same_inode_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-inode rewrite during the read must invalidate the snapshot."""
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"value":1}')
    original_read = bounded_json._read_bounded_descriptor

    def read_then_mutate(file_descriptor: int, *, byte_limit: int) -> bytes:
        content = original_read(file_descriptor, byte_limit=byte_limit)
        inode = path.stat().st_ino
        path.write_bytes(b'{"replacement":"longer-value"}')
        assert path.stat().st_ino == inode
        return content

    monkeypatch.setattr(
        bounded_json, "_read_bounded_descriptor", read_then_mutate
    )

    with pytest.raises(
        ValueError, match="JSON input path changed during the bounded read"
    ):
        bounded_json.read_json_object(path)
