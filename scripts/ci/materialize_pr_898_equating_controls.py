#!/usr/bin/env python3
"""One-shot, fail-closed patch materializer for PR #898."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import subprocess
import tempfile
import zlib

TARGET = Path("python/fast_mlsirm/equating.py")
EXPECTED_BEFORE = "4dafd0cbaee78742cabe7cfc6cc446e6ade24c22"
PATCH_PAYLOAD = "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"


def git_blob_sha(content: bytes) -> str:
    """Return Git's canonical SHA-1 for one regular-file blob."""
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def main() -> None:
    """Apply only the reviewed hunks and reject source drift or fuzzy failure."""
    before = git_blob_sha(TARGET.read_bytes())
    if before != EXPECTED_BEFORE:
        raise SystemExit(
            f"refusing unexpected {TARGET} blob {before}; expected {EXPECTED_BEFORE}"
        )
    patch = zlib.decompress(base64.b64decode(PATCH_PAYLOAD, validate=True))
    with tempfile.NamedTemporaryFile(suffix=".patch") as handle:
        handle.write(patch)
        handle.flush()
        subprocess.run(
            ["git", "apply", "--check", "--whitespace=error-all", handle.name],
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--whitespace=error-all", handle.name],
            check=True,
        )
    after = git_blob_sha(TARGET.read_bytes())
    if after == before:
        raise SystemExit("patch produced no source change")


if __name__ == "__main__":
    main()
