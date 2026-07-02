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
from html import escape
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
        "html_report": "buyer_evidence_report.html" in files,
    }


def _content_security_policy() -> str:
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "go" if value else "missing"
    if value is None:
        return "None"
    return str(value)


def _render_report_html(manifest: dict[str, Any]) -> str:
    coverage = manifest.get("coverage", {})
    coverage_rows = []
    if isinstance(coverage, dict):
        for name, ok in sorted(coverage.items()):
            coverage_rows.append(
                "\n".join(
                    [
                        "<tr>",
                        f"<th scope=\"row\">{escape(name.replace('_', ' ').title())}</th>",
                        f"<td>{escape(_format_value(ok))}</td>",
                        "</tr>",
                    ]
                )
            )
    file_rows = []
    files = manifest.get("files", [])
    if isinstance(files, list):
        for item in files[:25]:
            if not isinstance(item, dict):
                continue
            file_rows.append(
                "\n".join(
                    [
                        "<tr>",
                        f"<th scope=\"row\">{escape(str(item.get('archive_path', '')))}</th>",
                        f"<td>{escape(str(item.get('size_bytes', '')))}</td>",
                        f"<td><code>{escape(str(item.get('sha256', '')))}</code></td>",
                        "</tr>",
                    ]
                )
            )
    contract_value = manifest.get("contract_value_krw")
    if isinstance(contract_value, int):
        contract_value_display = f"KRW {contract_value:,}"
    elif contract_value in (None, ""):
        contract_value_display = ""
    else:
        contract_value_display = f"KRW {contract_value}"
    cards = [
        ("Contract Value", contract_value_display),
        ("Artifact Count", manifest.get("artifact_count", "")),
        ("Source Commit", manifest.get("source_commit", "")),
        ("Packet ZIP SHA256", manifest.get("zip_sha256", "calculated after archive write")),
    ]
    card_markup = [
        "\n".join(
            [
                '<article class="metric-card">',
                f"<span>{escape(label)}</span>",
                f"<strong>{escape(_format_value(value))}</strong>",
                "</article>",
            ]
        )
        for label, value in cards
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm Buyer Evidence Review</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            "<p>fast-mlsirm procurement packet</p>",
            "<h1>Buyer Evidence Review</h1>",
            f"<span>Generated: {escape(str(manifest.get('generated_at', '')))}</span>",
            "</section>",
            '<section class="report-section">',
            "<h2>Decision Summary</h2>",
            '<div class="metrics-grid">',
            *card_markup,
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Required Evidence Coverage</h2>",
            '<div class="table-wrap" role="region" aria-label="Required evidence coverage table" tabindex="0">',
            "<table>",
            "<caption>Required evidence coverage table</caption>",
            "<thead><tr><th scope=\"col\">Evidence</th><th scope=\"col\">Status</th></tr></thead>",
            "<tbody>",
            *coverage_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Artifact Digest Sample</h2>",
            '<div class="table-wrap" role="region" aria-label="Artifact digest table" tabindex="0">',
            "<table>",
            "<caption>Artifact digest table</caption>",
            "<thead><tr><th scope=\"col\">Archive Path</th><th scope=\"col\">Bytes</th><th scope=\"col\">SHA256</th></tr></thead>",
            "<tbody>",
            *file_rows,
            "</tbody>",
            "</table>",
            "</div>",
            '<p class="note">This report summarizes procurement evidence only. It is not a valuation guarantee or regulated-use approval.</p>',
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _report_css() -> str:
    return """
:root {
  color: #172026;
  background: #f5f7f8;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
}
main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
.hero {
  background: #12343b;
  color: #ffffff;
  border-radius: 8px;
  padding: 28px;
}
.hero p,
.hero h1 {
  margin: 0;
}
.hero p {
  color: #b7d7d0;
  font-size: 0.86rem;
  font-weight: 700;
  text-transform: uppercase;
}
.hero h1 {
  margin-top: 8px;
  font-size: 2rem;
}
.hero span {
  display: inline-block;
  margin-top: 14px;
  color: #dce8e5;
}
.report-section {
  margin-top: 22px;
  background: #ffffff;
  border: 1px solid #d8e1e3;
  border-radius: 8px;
  padding: 22px;
}
.report-section h2 {
  margin: 0 0 16px;
  font-size: 1.16rem;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.metric-card {
  border: 1px solid #d8e1e3;
  border-radius: 8px;
  padding: 14px;
}
.metric-card span {
  display: block;
  color: #5e6f76;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
}
.metric-card strong {
  display: block;
  margin-top: 8px;
  overflow-wrap: anywhere;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid #d8e1e3;
  border-radius: 8px;
}
.table-wrap:focus {
  outline: 3px solid #0f766e;
  outline-offset: 3px;
}
table {
  width: 100%;
  min-width: 620px;
  border-collapse: collapse;
}
caption {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid #e8edef;
  text-align: left;
  vertical-align: top;
}
tbody tr:last-child th,
tbody tr:last-child td {
  border-bottom: 0;
}
code {
  overflow-wrap: anywhere;
}
.note {
  color: #5e6f76;
  margin-bottom: 0;
}
"""


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "fast_mlsirm_buyer_evidence_packet.zip"
    manifest_path = out_dir / "buyer_evidence_manifest.json"
    report_path = out_dir / "buyer_evidence_report.html"

    files = _collect_files(
        repo_root=repo_root,
        acceptance_path=Path(args.acceptance).resolve(),
        sales_readiness_path=Path(args.sales_readiness).resolve(),
        dist_dir=Path(args.dist).resolve(),
    )
    file_entries = [
        {
            "archive_path": archive_path,
            "source_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for archive_path, path in sorted(files.items())
    ]
    coverage = _coverage(files)
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
    report_path.write_text(_render_report_html(manifest), encoding="utf-8")
    files["buyer_evidence_report.html"] = report_path
    file_entries = [
        {
            "archive_path": archive_path,
            "source_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for archive_path, path in sorted(files.items())
    ]
    coverage = _coverage(files)
    if not all(coverage.values()):
        missing = [name for name, ok in coverage.items() if not ok]
        raise RuntimeError(f"buyer evidence packet is missing required coverage: {missing}")
    manifest["coverage"] = coverage
    manifest["artifact_count"] = len(file_entries)
    manifest["files"] = file_entries
    manifest["report_file"] = str(report_path)
    manifest["report_sha256"] = _sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as packet:
        for archive_path, path in sorted(files.items()):
            packet.write(path, archive_path)
        packet.write(manifest_path, "buyer_evidence_manifest.json")

    manifest["zip_file"] = str(zip_path)
    manifest["zip_sha256"] = _sha256(zip_path)
    report_path.write_text(_render_report_html(manifest), encoding="utf-8")
    manifest["report_sha256"] = _sha256(report_path)
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
                "report": str(Path(args.out).resolve() / "buyer_evidence_report.html"),
                "zip": manifest["zip_file"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
