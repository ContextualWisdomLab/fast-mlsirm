#!/usr/bin/env python
"""Build a portable buyer evidence packet for fast-mlsirm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PRODUCT_DOCS = [
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/commercial_readiness.md",
    "docs/enterprise_sales_readiness.md",
    "docs/release_acceptance.md",
    "docs/prd_trd_summary.md",
    "docs/20b_product_readiness.md",
    "docs/buyer_demo_storyboard.md",
    "docs/figma_product_design_packet.md",
    "docs/roi_evidence_model.md",
]

PRODUCT_MANIFESTS = [
    "examples/enterprise_demo/roi_manifest.json",
    "examples/enterprise_demo/benchmark_manifest.json",
    "examples/enterprise_demo/figma_design_packet.json",
    "examples/enterprise_demo/product_completion_manifest.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _source_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _add_file(
    files: dict[str, Path],
    archive_path: str,
    source_path: Path,
    *,
    required: bool = True,
) -> None:
    if not source_path.exists() or not source_path.is_file():
        if required:
            raise RuntimeError(f"required evidence file is missing: {source_path}")
        return
    files[archive_path] = source_path


def _acceptance_artifact_files(acceptance: dict[str, Any]) -> list[Path]:
    artifacts: list[Path] = []
    for step in acceptance.get("steps", []):
        if not isinstance(step, dict):
            continue
        files = step.get("files")
        if isinstance(files, dict):
            artifacts.extend(Path(str(path)) for path in files.values())
    return artifacts


def _collect_files(
    *,
    repo_root: Path,
    acceptance_path: Path,
    sales_readiness_path: Path,
    dist_dir: Path,
) -> dict[str, Path]:
    acceptance = _read_json(acceptance_path)
    files: dict[str, Path] = {}
    _add_file(files, "acceptance/acceptance_summary.json", acceptance_path)
    _add_file(files, "sales/sales_readiness_manifest.json", sales_readiness_path)

    for path in sorted(dist_dir.glob("*.whl")):
        _add_file(files, f"dist/{path.name}", path)
    for path in sorted(dist_dir.glob("*.tar.gz")):
        _add_file(files, f"dist/{path.name}", path)

    for relative in PRODUCT_DOCS:
        _add_file(files, relative, repo_root / relative)
    for relative in PRODUCT_MANIFESTS:
        _add_file(files, relative, repo_root / relative)

    acceptance_dir = acceptance_path.parent.resolve()
    for path in _acceptance_artifact_files(acceptance):
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise RuntimeError(f"acceptance artifact is missing: {resolved}")
        try:
            relative = resolved.relative_to(acceptance_dir)
            archive_path = f"acceptance/artifacts/{relative.as_posix()}"
        except ValueError:
            archive_path = f"acceptance/artifacts/{resolved.name}"
        files.setdefault(archive_path, resolved)
    return files


def _coverage(files: dict[str, Path]) -> dict[str, bool]:
    return {
        "acceptance_summary": "acceptance/acceptance_summary.json" in files,
        "sales_readiness_manifest": "sales/sales_readiness_manifest.json" in files,
        "wheel": any(path.startswith("dist/") and path.endswith(".whl") for path in files),
        "sdist": any(path.startswith("dist/") and path.endswith(".tar.gz") for path in files),
        "product_docs": all(relative in files for relative in PRODUCT_DOCS),
        "product_manifests": all(relative in files for relative in PRODUCT_MANIFESTS),
        "acceptance_artifacts": any(path.startswith("acceptance/artifacts/") for path in files),
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "fast_mlsirm_buyer_evidence_packet.zip"
    manifest_path = out_dir / "buyer_evidence_manifest.json"

    files = _collect_files(
        repo_root=repo_root,
        acceptance_path=Path(args.acceptance).resolve(),
        sales_readiness_path=Path(args.sales_readiness).resolve(),
        dist_dir=Path(args.dist).resolve(),
    )
    coverage = _coverage(files)
    if not all(coverage.values()):
        missing = [name for name, ok in coverage.items() if not ok]
        raise RuntimeError(f"buyer evidence packet is missing required coverage: {missing}")

    file_entries = [
        {
            "archive_path": archive_path,
            "source_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for archive_path, path in sorted(files.items())
    ]
    manifest: dict[str, Any] = {
        "status": "ok",
        "command": "build_buyer_packet",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "artifact_count": len(file_entries),
        "coverage": coverage,
        "files": file_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as packet:
        for archive_path, path in sorted(files.items()):
            packet.write(path, archive_path)
        packet.write(manifest_path, "buyer_evidence_manifest.json")

    manifest["zip_file"] = str(zip_path)
    manifest["zip_sha256"] = _sha256(zip_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a portable fast-mlsirm buyer evidence packet.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing docs and demo manifests.")
    parser.add_argument("--acceptance", required=True, help="Path to acceptance_summary.json.")
    parser.add_argument("--sales-readiness", required=True, help="Path to sales_readiness_manifest.json.")
    parser.add_argument("--dist", required=True, help="Directory containing the built wheel and sdist.")
    parser.add_argument("--out", default="buyer-evidence-packet", help="Output directory for packet files.")
    parser.add_argument("--contract-value-krw", type=int, default=2_000_000_000, help="Target contract value.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_packet(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(Path(args.out).resolve()),
                "manifest": str(Path(args.out).resolve() / "buyer_evidence_manifest.json"),
                "zip": manifest["zip_file"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
