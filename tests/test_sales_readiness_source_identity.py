from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_acquisition_release, build_buyer_packet, sales_readiness


SOURCE_COMMIT = "a" * 40


def _minimal_sales_args(tmp_path: Path, acceptance_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(tmp_path),
        acceptance=str(acceptance_path),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=False,
        require_20b_product=False,
        require_acquisition_readiness=False,
        check_import=False,
        contract_value_krw=None,
        max_acceptance_seconds=None,
    )


def test_sales_readiness_inherits_sealed_acceptance_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standalone sales evidence must carry the source identity sealed by acceptance."""
    acceptance_path = tmp_path / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps({"status": "ok", "source_commit": SOURCE_COMMIT}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sales_readiness, "_validate_required_files", lambda _root: [])
    monkeypatch.setattr(sales_readiness, "_validate_doc_tokens", lambda _root: [])
    monkeypatch.setattr(
        sales_readiness,
        "_validate_acceptance_summary",
        lambda _path, *, require_rust, max_acceptance_seconds: [],
    )
    monkeypatch.setattr(sales_readiness, "_validate_dist", lambda _dist: [])

    manifest = sales_readiness.run_sales_readiness(
        _minimal_sales_args(tmp_path, acceptance_path)
    )

    assert manifest["source_commit"] == SOURCE_COMMIT


def test_acquisition_stage_requires_sales_readiness_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acquisition orchestrator must not exempt sales readiness from provenance."""
    manifest_path = tmp_path / "sales_readiness_manifest.json"
    payload = {"status": "ok"}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        build_acquisition_release,
        "_assert_source_unchanged",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="initial_sales_readiness missing source_commit"):
        build_acquisition_release._verify_generated_stage(
            "initial_sales_readiness",
            payload,
            manifest_path,
            SOURCE_COMMIT,
            tmp_path,
        )


def test_buyer_packet_rejects_sales_readiness_without_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buyer evidence must reject a provenance-less sales-readiness manifest."""
    monkeypatch.setattr(build_buyer_packet, "PRODUCT_DOCS", [])
    monkeypatch.setattr(build_buyer_packet, "PRODUCT_MANIFESTS", [])

    acceptance_dir = tmp_path / "acceptance"
    artifact = acceptance_dir / "artifacts" / "fit" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"backend":"rust"}', encoding="utf-8")
    relative = artifact.resolve().relative_to(acceptance_dir.resolve()).as_posix()
    acceptance_path = acceptance_dir / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_commit": SOURCE_COMMIT,
                "artifact_sha256": {
                    relative: hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
                "steps": [
                    {
                        "command": "fit",
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

    with pytest.raises(RuntimeError, match="sales readiness source commit is missing"):
        build_buyer_packet._collect_files(
            repo_root=tmp_path,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            expected_source_commit=SOURCE_COMMIT,
        )
