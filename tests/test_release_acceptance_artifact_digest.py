from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import release_acceptance as subject


def test_acceptance_artifact_sha256_uses_root_relative_paths(tmp_path: Path) -> None:
    """Acceptance sealing records buyer-consumable artifacts by portable path."""
    out_dir = tmp_path / "acceptance"
    artifact = out_dir / "fit_auto" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'{"backend":"rust"}')

    observed = subject._acceptance_artifact_sha256(
        out_dir,
        [{"command": "fit_auto", "files": {"summary": str(artifact)}}],
    )

    assert observed == {
        "fit_auto/fit_summary.json": hashlib.sha256(artifact.read_bytes()).hexdigest()
    }


def test_acceptance_artifact_sha256_rejects_root_escape(tmp_path: Path) -> None:
    """Acceptance summaries must not seal a step artifact outside their evidence root."""
    out_dir = tmp_path / "acceptance"
    out_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside acceptance evidence root"):
        subject._acceptance_artifact_sha256(
            out_dir,
            [{"command": "fit_auto", "files": {"summary": str(outside)}}],
        )
