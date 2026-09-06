import importlib.util
import json
from pathlib import Path

import pytest


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet_boundary", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_files_rejects_acceptance_artifact_outside_evidence_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buyer packets must not silently adopt artifacts outside acceptance evidence."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    acceptance_dir = tmp_path / "acceptance"
    acceptance_dir.mkdir()
    outside = tmp_path / "outside" / "result.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    acceptance_path = acceptance_dir / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "steps": [
                    {
                        "command": "fit",
                        "files": {"summary": str(outside)},
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

    with pytest.raises(RuntimeError, match="outside acceptance evidence root"):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
        )
