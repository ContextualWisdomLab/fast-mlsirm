"""Regression tests for buyer delivery-packet integrity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts import build_release_evidence_index, sales_readiness


def _write_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create separate payload and delivered packet artifacts with a detached digest."""
    payload = tmp_path / "buyer_evidence_payload.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("evidence.txt", "payload")
    payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()

    packet = tmp_path / "fast_mlsirm_buyer_evidence_packet.zip"
    with zipfile.ZipFile(packet, "w") as archive:
        archive.writestr("buyer_evidence_manifest.json", "{}")
        archive.writestr("buyer_evidence_report.html", "<title>Buyer Evidence</title>")
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    digest = tmp_path / "fast_mlsirm_buyer_evidence_packet.sha256"
    digest.write_text(packet_sha + "\n", encoding="ascii")

    report = tmp_path / "buyer_evidence_report.html"
    report.write_text("<!doctype html><title>Buyer Evidence</title>", encoding="utf-8")
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    manifest = {
        "status": "ok",
        "contract_value_krw": None,
        "artifact_count": 2,
        "coverage": {
            "acceptance_summary": True,
            "sales_readiness_manifest": True,
            "wheel": True,
            "sdist": True,
            "product_docs": True,
            "product_manifests": True,
            "acceptance_artifacts": True,
            "html_report": True,
        },
        "payload_zip_file": str(payload),
        "payload_zip_sha256": payload_sha,
        "zip_file": str(payload),
        "zip_sha256": payload_sha,
        "packet_file": str(packet),
        "packet_sha256_file": str(digest),
        "report_file": str(report),
        "report_sha256": report_sha,
    }
    manifest_path = tmp_path / "buyer_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, packet, digest


def _failed_names(manifest_path: Path) -> set[str]:
    checks = sales_readiness._validate_buyer_packet(
        manifest_path,
        required=True,
        contract_value_krw=None,
    )
    return {str(check["name"]) for check in checks if check.get("ok") is not True}


def test_delivery_packet_and_detached_digest_are_required(tmp_path: Path) -> None:
    """The gate must validate the artifact delivered to a buyer, not only its payload ZIP."""
    manifest_path, packet, _digest = _write_manifest(tmp_path)
    assert _failed_names(manifest_path) == set()

    packet.write_bytes(packet.read_bytes() + b"tamper")
    assert "buyer_packet:packet_sha256" in _failed_names(manifest_path)

    manifest_path, _packet, digest = _write_manifest(tmp_path)
    digest.unlink()
    assert "buyer_packet:packet_sha256_file" in _failed_names(manifest_path)


def _write_release_inputs(tmp_path: Path) -> tuple[argparse.Namespace, Path]:
    """Create the minimal valid release-index inputs around one buyer packet."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "fast-mlsirm"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    acceptance = tmp_path / "acceptance_summary.json"
    acceptance.write_text(json.dumps({"status": "ok", "steps": []}), encoding="utf-8")
    sales = tmp_path / "sales_readiness_manifest.json"
    sales.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    benchmark_html = tmp_path / "benchmark_report.html"
    benchmark_html.write_text("<!doctype html><title>Benchmark</title>", encoding="utf-8")
    benchmark = tmp_path / "benchmark_report.json"
    benchmark.write_text(
        json.dumps(
            {
                "status": "ok",
                "budget_ok": True,
                "html_report_file": str(benchmark_html),
                "html_report_sha256": hashlib.sha256(benchmark_html.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    packet_manifest, packet, _digest = _write_manifest(tmp_path)

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "fast_mlsirm-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "fast_mlsirm-0.1.0.tar.gz").write_bytes(b"sdist")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        sales_readiness=str(sales),
        benchmark_report=str(benchmark),
        buyer_packet_manifest=str(packet_manifest),
        dist=str(dist),
        out=str(tmp_path / "release-index"),
        contract_value_krw=None,
    )
    return args, packet


def test_release_index_rejects_tampered_delivery_packet(tmp_path: Path) -> None:
    """Release evidence must cover the final delivery envelope, not only its inner payload."""
    args, packet = _write_release_inputs(tmp_path)
    assert build_release_evidence_index.build_index(args)["status"] == "ok"

    packet.write_bytes(packet.read_bytes() + b"tamper")
    result = build_release_evidence_index.build_index(args)
    assert result["status"] == "failed"
    assert any("delivery packet" in failure.lower() for failure in result["failures"])
