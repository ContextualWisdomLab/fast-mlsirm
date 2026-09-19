from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location(
        "build_buyer_packet_acceptance_digest", script
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_files_rejects_acceptance_artifact_changed_after_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buyer evidence must replay acceptance-time artifact digests before admission."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    acceptance_dir = tmp_path / "acceptance"
    artifact = acceptance_dir / "artifacts" / "fit" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    original = b'{"backend":"rust","status":"ok"}'
    artifact.write_bytes(original)
    relative = artifact.resolve().relative_to(acceptance_dir.resolve()).as_posix()
    source_commit = "a" * 40
    acceptance_path = acceptance_dir / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_commit": source_commit,
                "artifact_sha256": {
                    relative: hashlib.sha256(original).hexdigest(),
                },
                "steps": [
                    {
                        "command": "fit_auto",
                        "files": {"summary": str(artifact)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sales_path = acceptance_dir / "sales_readiness_manifest.json"
    sales_path.write_text('{"status":"ok"}', encoding="utf-8")
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    artifact.write_text('{"backend":"rust","status":"tampered"}', encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="acceptance artifact SHA256 does not match acceptance_summary.json",
    ):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            expected_source_commit=source_commit,
        )
