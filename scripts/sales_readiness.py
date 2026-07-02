#!/usr/bin/env python
"""Enterprise sales-readiness verifier for fast-mlsirm.

This verifier is intentionally evidence-oriented. It does not claim that a
sale is guaranteed; it checks that a release candidate carries the artifacts,
scope statements, backend proof, and acceptance results expected before a
high-value enterprise procurement review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any


REQUIRED_POLICY_FILES = [
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
]

REQUIRED_20B_PRODUCT_FILES = [
    "docs/20b_product_readiness.md",
    "docs/buyer_demo_storyboard.md",
    "docs/figma_product_design_packet.md",
    "docs/roi_evidence_model.md",
    "examples/enterprise_demo/README.md",
    "examples/enterprise_demo/roi_manifest.json",
    "examples/enterprise_demo/benchmark_manifest.json",
    "examples/enterprise_demo/figma_design_packet.json",
    "examples/enterprise_demo/product_completion_manifest.json",
    "docs/superpowers/specs/2026-07-02-20b-product-readiness-design.md",
    "docs/superpowers/plans/2026-07-02-20b-product-readiness.md",
    "docs/superpowers/specs/2026-07-02-20b-benchmark-evidence-design.md",
    "docs/superpowers/plans/2026-07-02-20b-benchmark-evidence.md",
    "docs/superpowers/specs/2026-07-02-20b-release-evidence-index-design.md",
    "docs/superpowers/plans/2026-07-02-20b-release-evidence-index.md",
    "docs/superpowers/specs/2026-07-03-20b-commercial-release-builder-design.md",
    "docs/superpowers/plans/2026-07-03-20b-commercial-release-builder.md",
    "docs/superpowers/specs/2026-07-03-20b-procurement-due-diligence-design.md",
    "docs/superpowers/plans/2026-07-03-20b-procurement-due-diligence.md",
]

REQUIRED_DOC_TOKENS = {
    "README.md": [
        "Commercial Readiness",
        "Enterprise Sales Readiness",
        "scripts/release_acceptance.py",
        "scripts/sales_readiness.py",
        "scripts/build_release_evidence_index.py",
        "scripts/build_commercial_release.py",
        "scripts/build_procurement_due_diligence.py",
    ],
    "docs/commercial_readiness.md": [
        "Seller Acceptance Checklist",
        "Enterprise Sales Gate",
        "Security",
        "Support",
        "Release Gate",
    ],
    "docs/enterprise_sales_readiness.md": [
        "KRW 2,000,000,000",
        "Procurement Evidence",
        "Customer Acceptance Evidence",
        "Go/No-Go",
        "Out of Scope",
    ],
    "docs/release_acceptance.md": [
        "acceptance_summary.json",
        "sales_readiness_manifest.json",
        "release_evidence_index.json",
        "commercial_release_manifest.json",
        "procurement_due_diligence_manifest.json",
        "--require-rust",
    ],
}

REQUIRED_20B_DOC_TOKENS = {
    "docs/20b_product_readiness.md": [
        "KRW 2,000,000,000",
        "Buyer-Facing Product Standard",
        "Product Design Scope",
        "Data Analytics Scope",
        "Figma Code Connect",
        "benchmark_report.html",
        "release_evidence_index.html",
        "commercial_release_report.html",
        "procurement_due_diligence_report.html",
        "Go/No-Go",
    ],
    "docs/buyer_demo_storyboard.md": [
        "Package Evidence",
        "Synthetic Data",
        "Fit Workflow",
        "Diagnostics Workflow",
        "Procurement Packet",
    ],
    "docs/figma_product_design_packet.md": [
        "Figma Code Connect is explicitly out of scope",
        "Package Evidence",
        "Procurement Packet",
        "figma_design_packet.json",
    ],
    "docs/roi_evidence_model.md": [
        "Driver Metrics",
        "Required Evidence",
        "roi_manifest.json",
        "benchmark_report.json",
        "Caveats",
    ],
}

REQUIRED_20B_JSON_FIELDS = {
    "examples/enterprise_demo/roi_manifest.json": [
        "contract_value_krw",
        "position",
        "drivers",
        "required_evidence",
        "go_no_go",
    ],
    "examples/enterprise_demo/benchmark_manifest.json": [
        "benchmark_scope",
        "runtime_budget_seconds",
        "scenarios",
        "required_artifacts",
        "caveats",
    ],
    "examples/enterprise_demo/figma_design_packet.json": [
        "code_connect",
        "mode",
        "source",
        "frames",
        "handoff",
    ],
    "examples/enterprise_demo/product_completion_manifest.json": [
        "contract_value_krw",
        "scorecard_version",
        "checks",
        "go_no_go",
    ],
}

REQUIRED_COMPLETION_CHECKS = {
    "release_acceptance",
    "html_report_csp",
    "cli_stack_trace_guard",
    "report_table_accessibility",
    "figma_buyer_review",
    "buyer_evidence_packet",
    "buyer_evidence_html_report",
    "automated_benchmark_report",
    "release_evidence_index",
    "commercial_release_builder",
    "procurement_due_diligence",
}

REQUIRED_ACCEPTANCE_COMMANDS = {
    "simulate",
    "fit",
    "diagnose-fit",
    "diagnose-dimensions",
    "render-report",
}

REQUIRED_BUYER_PACKET_COVERAGE = {
    "acceptance_summary",
    "sales_readiness_manifest",
    "wheel",
    "sdist",
    "product_docs",
    "product_manifests",
    "acceptance_artifacts",
    "html_report",
}

REQUIRED_RELEASE_INDEX_COVERAGE = {
    "acceptance_summary",
    "sales_readiness_manifest",
    "benchmark_report",
    "benchmark_html_report",
    "buyer_packet_manifest",
    "buyer_packet_zip",
    "buyer_packet_html_report",
    "wheel",
    "sdist",
}

REQUIRED_PROCUREMENT_DUE_DILIGENCE_CATEGORIES = {
    "package",
    "policy",
    "commercial_release",
    "github",
}


def _check(name: str, ok: bool, detail: str, **metadata: object) -> dict[str, object]:
    payload: dict[str, object] = {"name": name, "ok": ok, "detail": detail}
    payload.update(metadata)
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        return _parse_project_version(_read_text(pyproject))
    with pyproject.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _parse_project_version(pyproject_text: str) -> str:
    in_project = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project:
            key, separator, value = line.partition("=")
            if separator and key.strip() == "version":
                return value.strip().strip("\"'")
    raise RuntimeError("pyproject.toml does not define [project].version")


def _validate_required_files(repo_root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative in REQUIRED_POLICY_FILES:
        path = repo_root / relative
        ok = path.exists() and path.is_file() and path.stat().st_size > 0
        checks.append(_check(f"file:{relative}", ok, "required product evidence file exists"))
    return checks


def _validate_doc_tokens(repo_root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative, tokens in REQUIRED_DOC_TOKENS.items():
        path = repo_root / relative
        if not path.exists():
            checks.append(_check(f"doc_tokens:{relative}", False, "document missing"))
            continue
        text = _read_text(path)
        missing = [token for token in tokens if token not in text]
        checks.append(
            _check(
                f"doc_tokens:{relative}",
                not missing,
                "required enterprise sales-readiness language is present",
                missing=missing,
            )
        )
    return checks


def _validate_20b_product_evidence(repo_root: Path, *, contract_value_krw: int) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative in REQUIRED_20B_PRODUCT_FILES:
        path = repo_root / relative
        ok = path.exists() and path.is_file() and path.stat().st_size > 0
        checks.append(_check(f"20b:file:{relative}", ok, "required 20B product evidence file exists"))

    for relative, tokens in REQUIRED_20B_DOC_TOKENS.items():
        path = repo_root / relative
        if not path.exists():
            checks.append(_check(f"20b:doc_tokens:{relative}", False, "document missing"))
            continue
        text = _read_text(path)
        missing = [token for token in tokens if token not in text]
        checks.append(
            _check(
                f"20b:doc_tokens:{relative}",
                not missing,
                "required 20B product-readiness language is present",
                missing=missing,
            )
        )

    for relative, fields in REQUIRED_20B_JSON_FIELDS.items():
        path = repo_root / relative
        if not path.exists():
            checks.append(_check(f"20b:json:{relative}", False, "manifest missing"))
            continue
        try:
            payload = _read_json(path)
        except Exception as exc:
            checks.append(_check(f"20b:json:{relative}", False, f"manifest is not valid JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            checks.append(
                _check(
                    f"20b:json_shape:{relative}",
                    False,
                    "manifest must be a JSON object",
                    actual_type=type(payload).__name__,
                )
            )
            continue
        missing = [field for field in fields if field not in payload]
        non_empty_failures = [
            field
            for field in fields
            if field in payload
            and (payload[field] is None or (isinstance(payload[field], (list, dict, str)) and not payload[field]))
        ]
        checks.append(
            _check(
                f"20b:json_fields:{relative}",
                not missing and not non_empty_failures,
                "manifest includes required non-empty fields",
                missing=missing,
                empty=non_empty_failures,
            )
        )

        if relative.endswith("roi_manifest.json"):
            checks.append(
                _check(
                    "20b:roi_contract_value",
                    payload.get("contract_value_krw") == contract_value_krw,
                    "ROI manifest contract value matches readiness gate",
                    expected=contract_value_krw,
                    actual=payload.get("contract_value_krw"),
                )
            )
        if relative.endswith("figma_design_packet.json"):
            checks.append(
                _check(
                    "20b:figma_code_connect_disabled",
                    payload.get("code_connect") is False,
                    "Figma design packet keeps Code Connect disabled",
                    actual=payload.get("code_connect"),
                )
            )
            if "figma_artifact_url" in payload:
                figma_url = payload["figma_artifact_url"]
                checks.append(
                    _check(
                        "20b:figma_artifact_url",
                        isinstance(figma_url, str) and figma_url.startswith("https://www.figma.com/design/"),
                        (
                            "optional Figma artifact URL must start with https://www.figma.com/design/; "
                            "FigJam /board/ URLs do not satisfy the design-file evidence contract"
                        ),
                        actual=figma_url,
                    )
                )
        if relative.endswith("product_completion_manifest.json"):
            completion_checks = payload.get("checks", [])
            completion_ids = {
                check.get("id")
                for check in completion_checks
                if isinstance(check, dict) and isinstance(check.get("id"), str)
            }
            non_go = [
                check.get("id")
                for check in completion_checks
                if isinstance(check, dict) and check.get("status") != "go"
            ]
            checks.append(
                _check(
                    "20b:completion_contract_value",
                    payload.get("contract_value_krw") == contract_value_krw,
                    "completion manifest contract value matches readiness gate",
                    expected=contract_value_krw,
                    actual=payload.get("contract_value_krw"),
                )
            )
            checks.append(
                _check(
                    "20b:completion_scorecard",
                    isinstance(completion_checks, list)
                    and REQUIRED_COMPLETION_CHECKS.issubset(completion_ids)
                    and not non_go,
                    "completion manifest includes required go-status hardening checks",
                    missing=sorted(REQUIRED_COMPLETION_CHECKS - completion_ids),
                    non_go=non_go,
                )
            )
    return checks


def _validate_acceptance_summary(
    acceptance_path: Path,
    *,
    require_rust: bool,
    max_acceptance_seconds: float | None,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if not acceptance_path.exists():
        return [_check("acceptance:summary", False, f"missing acceptance summary: {acceptance_path}")]

    summary = _read_json(acceptance_path)
    checks.append(_check("acceptance:status", summary.get("status") == "ok", "acceptance summary status is ok"))

    steps = summary.get("steps", [])
    commands = {step.get("command") for step in steps if isinstance(step, dict)}
    missing_commands = sorted(REQUIRED_ACCEPTANCE_COMMANDS - commands)
    checks.append(
        _check(
            "acceptance:commands",
            not missing_commands,
            "acceptance summary includes required workflow commands",
            missing=missing_commands,
        )
    )

    fit_backends = [step.get("backend") for step in steps if isinstance(step, dict) and step.get("command") == "fit"]
    checks.append(
        _check(
            "acceptance:fit_backend_record",
            all(backend in {"numpy", "rust"} for backend in fit_backends) and bool(fit_backends),
            "fit steps record resolved backend",
            backends=fit_backends,
        )
    )

    if require_rust:
        explicit_rust = [
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("command") == "fit"
            and step.get("backend") == "rust"
            and "fit_rust" in str(step.get("out", ""))
        ]
        checks.append(
            _check(
                "acceptance:explicit_rust_fit",
                bool(explicit_rust),
                "acceptance summary includes explicit rust fit artifact path",
            )
        )

    total_duration = summary.get("total_duration_seconds")
    checks.append(
        _check(
            "acceptance:timing",
            isinstance(total_duration, (int, float)) and total_duration >= 0,
            "acceptance summary records total runtime",
            total_duration_seconds=total_duration,
        )
    )

    if max_acceptance_seconds is not None and isinstance(total_duration, (int, float)):
        checks.append(
            _check(
                "acceptance:runtime_budget",
                total_duration <= max_acceptance_seconds,
                "acceptance runtime is within configured budget",
                budget_seconds=max_acceptance_seconds,
                total_duration_seconds=total_duration,
            )
        )

    artifact_paths: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        files = step.get("files")
        if isinstance(files, dict):
            artifact_paths.extend(str(path) for path in files.values())

    missing_artifacts = [path for path in artifact_paths if not Path(path).exists()]
    checks.append(
        _check(
            "acceptance:artifacts",
            not missing_artifacts and bool(artifact_paths),
            "acceptance artifacts referenced by summary exist",
            missing=missing_artifacts,
        )
    )
    return checks


def _validate_dist(dist_dir: Path | None) -> list[dict[str, object]]:
    if dist_dir is None:
        return [_check("dist:skipped", True, "distribution artifact check not requested")]
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    return [
        _check("dist:wheel", bool(wheels), "wheel artifact exists", files=[str(path) for path in wheels]),
        _check("dist:sdist", bool(sdists), "source distribution artifact exists", files=[str(path) for path in sdists]),
    ]


def _validate_buyer_packet(
    manifest_path: Path | None,
    *,
    required: bool,
    contract_value_krw: int,
) -> list[dict[str, object]]:
    if manifest_path is None:
        return [
            _check(
                "buyer_packet:skipped",
                not required,
                "buyer evidence packet check not requested",
            )
        ]
    if not manifest_path.exists():
        return [_check("buyer_packet:manifest", False, f"missing buyer packet manifest: {manifest_path}")]
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:
        return [_check("buyer_packet:manifest", False, f"buyer packet manifest is not valid JSON: {exc}")]
    if not isinstance(payload, dict):
        return [_check("buyer_packet:manifest_shape", False, "buyer packet manifest must be a JSON object")]

    coverage = payload.get("coverage", {})
    coverage_missing = [
        name
        for name in sorted(REQUIRED_BUYER_PACKET_COVERAGE)
        if not isinstance(coverage, dict) or coverage.get(name) is not True
    ]
    zip_file = payload.get("zip_file")
    zip_path = Path(str(zip_file)) if isinstance(zip_file, str) and zip_file else None
    if zip_path is not None and not zip_path.is_absolute():
        zip_path = manifest_path.parent / zip_path
    zip_exists = zip_path is not None and zip_path.exists() and zip_path.is_file()
    expected_zip_sha = payload.get("zip_sha256")
    actual_zip_sha = _sha256(zip_path) if zip_exists else None
    report_file = payload.get("report_file")
    report_path = Path(str(report_file)) if isinstance(report_file, str) and report_file else None
    if report_path is not None and not report_path.is_absolute():
        report_path = manifest_path.parent / report_path
    report_exists = report_path is not None and report_path.exists() and report_path.is_file()
    expected_report_sha = payload.get("report_sha256")
    actual_report_sha = _sha256(report_path) if report_exists else None
    return [
        _check(
            "buyer_packet:status",
            payload.get("status") == "ok",
            "buyer packet manifest status is ok",
            actual=payload.get("status"),
        ),
        _check(
            "buyer_packet:contract_value",
            payload.get("contract_value_krw") == contract_value_krw,
            "buyer packet contract value matches readiness gate",
            expected=contract_value_krw,
            actual=payload.get("contract_value_krw"),
        ),
        _check(
            "buyer_packet:artifact_count",
            isinstance(payload.get("artifact_count"), int) and payload["artifact_count"] > 0,
            "buyer packet records included artifact count",
            actual=payload.get("artifact_count"),
        ),
        _check(
            "buyer_packet:coverage",
            not coverage_missing,
            "buyer packet includes required evidence coverage",
            missing=coverage_missing,
        ),
        _check(
            "buyer_packet:zip_file",
            zip_exists,
            "buyer packet zip file exists",
            actual=str(zip_path) if zip_path is not None else None,
        ),
        _check(
            "buyer_packet:zip_sha256",
            zip_exists and isinstance(expected_zip_sha, str) and expected_zip_sha == actual_zip_sha,
            "buyer packet zip SHA256 matches manifest",
            expected=expected_zip_sha,
            actual=actual_zip_sha,
        ),
        _check(
            "buyer_packet:html_report",
            report_exists,
            "buyer evidence HTML review report exists",
            actual=str(report_path) if report_path is not None else None,
        ),
        _check(
            "buyer_packet:html_report_sha256",
            report_exists and isinstance(expected_report_sha, str) and expected_report_sha == actual_report_sha,
            "buyer evidence HTML report SHA256 matches manifest",
            expected=expected_report_sha,
            actual=actual_report_sha,
        ),
    ]


def _validate_benchmark_report(
    manifest_path: Path | None,
    *,
    required: bool,
) -> list[dict[str, object]]:
    if manifest_path is None:
        return [
            _check(
                "benchmark_report:skipped",
                not required,
                "benchmark report check not requested",
            )
        ]
    if not manifest_path.exists():
        return [_check("benchmark_report:manifest", False, f"missing benchmark report: {manifest_path}")]
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:
        return [_check("benchmark_report:manifest", False, f"benchmark report is not valid JSON: {exc}")]
    if not isinstance(payload, dict):
        return [_check("benchmark_report:manifest_shape", False, "benchmark report must be a JSON object")]

    scenario = payload.get("scenario_coverage", {})
    missing_backends = scenario.get("missing_backends") if isinstance(scenario, dict) else None
    artifacts = payload.get("artifact_coverage", {})
    missing_artifacts = artifacts.get("missing") if isinstance(artifacts, dict) else None
    missing_paths = artifacts.get("missing_paths") if isinstance(artifacts, dict) else None
    html_file = payload.get("html_report_file")
    html_path = Path(str(html_file)) if isinstance(html_file, str) and html_file else None
    if html_path is not None and not html_path.is_absolute():
        html_path = manifest_path.parent / html_path
    html_exists = html_path is not None and html_path.exists() and html_path.is_file()
    expected_html_sha = payload.get("html_report_sha256")
    actual_html_sha = _sha256(html_path) if html_exists else None
    return [
        _check(
            "benchmark_report:status",
            payload.get("status") == "ok",
            "benchmark report status is ok",
            actual=payload.get("status"),
        ),
        _check(
            "benchmark_report:runtime_budget",
            payload.get("budget_ok") is True,
            "benchmark report runtime is within configured budget",
            budget_seconds=payload.get("runtime_budget_seconds"),
            total_duration_seconds=payload.get("total_duration_seconds"),
        ),
        _check(
            "benchmark_report:scenario_coverage",
            missing_backends == [],
            "benchmark report covers required benchmark scenario backends",
            missing=missing_backends,
        ),
        _check(
            "benchmark_report:artifact_coverage",
            missing_artifacts == [] and missing_paths == [],
            "benchmark report covers required benchmark artifacts",
            missing=missing_artifacts,
            missing_paths=missing_paths,
        ),
        _check(
            "benchmark_report:html_report",
            html_exists,
            "benchmark HTML report exists",
            actual=str(html_path) if html_path is not None else None,
        ),
        _check(
            "benchmark_report:html_report_sha256",
            html_exists and isinstance(expected_html_sha, str) and expected_html_sha == actual_html_sha,
            "benchmark HTML report SHA256 matches manifest",
            expected=expected_html_sha,
            actual=actual_html_sha,
        ),
    ]


def _validate_release_evidence_index(
    manifest_path: Path | None,
    *,
    required: bool,
    contract_value_krw: int,
) -> list[dict[str, object]]:
    if manifest_path is None:
        return [
            _check(
                "release_evidence_index:skipped",
                not required,
                "release evidence index check not requested",
            )
        ]
    if not manifest_path.exists():
        return [_check("release_evidence_index:manifest", False, f"missing release evidence index: {manifest_path}")]
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:
        return [_check("release_evidence_index:manifest", False, f"release evidence index is not valid JSON: {exc}")]
    if not isinstance(payload, dict):
        return [_check("release_evidence_index:manifest_shape", False, "release evidence index must be a JSON object")]

    coverage = payload.get("coverage", {})
    coverage_missing = [
        name
        for name in sorted(REQUIRED_RELEASE_INDEX_COVERAGE)
        if not isinstance(coverage, dict) or coverage.get(name) is not True
    ]
    dist = payload.get("dist", {})
    artifacts = dist.get("artifacts") if isinstance(dist, dict) else None
    dist_artifacts_ok = (
        isinstance(artifacts, list)
        and any(isinstance(entry, dict) and entry.get("kind") == "wheel" for entry in artifacts)
        and any(isinstance(entry, dict) and entry.get("kind") == "sdist" for entry in artifacts)
        and all(
            isinstance(entry, dict)
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
            and isinstance(entry.get("size_bytes"), int)
            and entry["size_bytes"] > 0
            for entry in artifacts
        )
    )
    html_file = payload.get("html_report_file")
    html_path = Path(str(html_file)) if isinstance(html_file, str) and html_file else None
    if html_path is not None and not html_path.is_absolute():
        html_path = manifest_path.parent / html_path
    html_exists = html_path is not None and html_path.exists() and html_path.is_file()
    expected_html_sha = payload.get("html_report_sha256")
    actual_html_sha = _sha256(html_path) if html_exists else None
    failures = payload.get("failures")
    return [
        _check(
            "release_evidence_index:status",
            payload.get("status") == "ok",
            "release evidence index status is ok",
            actual=payload.get("status"),
        ),
        _check(
            "release_evidence_index:contract_value",
            payload.get("contract_value_krw") == contract_value_krw,
            "release evidence index contract value matches readiness gate",
            expected=contract_value_krw,
            actual=payload.get("contract_value_krw"),
        ),
        _check(
            "release_evidence_index:coverage",
            not coverage_missing,
            "release evidence index includes required release evidence coverage",
            missing=coverage_missing,
        ),
        _check(
            "release_evidence_index:dist_artifacts",
            dist_artifacts_ok,
            "release evidence index records wheel and sdist digests",
            artifact_count=len(artifacts) if isinstance(artifacts, list) else None,
        ),
        _check(
            "release_evidence_index:failures",
            failures == [],
            "release evidence index has no recorded failures",
            failures=failures,
        ),
        _check(
            "release_evidence_index:html_report",
            html_exists,
            "release evidence HTML report exists",
            actual=str(html_path) if html_path is not None else None,
        ),
        _check(
            "release_evidence_index:html_report_sha256",
            html_exists and isinstance(expected_html_sha, str) and expected_html_sha == actual_html_sha,
            "release evidence HTML report SHA256 matches manifest",
            expected=expected_html_sha,
            actual=actual_html_sha,
        ),
    ]


def _validate_procurement_due_diligence(
    manifest_path: Path | None,
    *,
    required: bool,
    contract_value_krw: int,
) -> list[dict[str, object]]:
    if manifest_path is None:
        return [
            _check(
                "procurement_due_diligence:skipped",
                not required,
                "procurement due-diligence check not requested",
            )
        ]
    if not manifest_path.exists():
        return [
            _check(
                "procurement_due_diligence:manifest",
                False,
                f"missing procurement due-diligence manifest: {manifest_path}",
            )
        ]
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:
        return [
            _check(
                "procurement_due_diligence:manifest",
                False,
                f"procurement due-diligence manifest is not valid JSON: {exc}",
            )
        ]
    if not isinstance(payload, dict):
        return [
            _check(
                "procurement_due_diligence:manifest_shape",
                False,
                "procurement due-diligence manifest must be a JSON object",
            )
        ]

    checks = payload.get("checks", [])
    ok_categories = {
        check.get("category")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is True and isinstance(check.get("category"), str)
    }
    failed_checks = payload.get("failed_checks")
    html_file = payload.get("html_report_file")
    html_path = Path(str(html_file)) if isinstance(html_file, str) and html_file else None
    if html_path is not None and not html_path.is_absolute():
        html_path = manifest_path.parent / html_path
    html_exists = html_path is not None and html_path.exists() and html_path.is_file()
    expected_html_sha = payload.get("html_report_sha256")
    actual_html_sha = _sha256(html_path) if html_exists else None
    return [
        _check(
            "procurement_due_diligence:status",
            payload.get("status") == "ok",
            "procurement due-diligence manifest status is ok",
            actual=payload.get("status"),
        ),
        _check(
            "procurement_due_diligence:contract_value",
            payload.get("contract_value_krw") == contract_value_krw,
            "procurement due-diligence contract value matches readiness gate",
            expected=contract_value_krw,
            actual=payload.get("contract_value_krw"),
        ),
        _check(
            "procurement_due_diligence:failed_checks",
            failed_checks == [],
            "procurement due-diligence manifest has no recorded failures",
            failed_checks=failed_checks,
        ),
        _check(
            "procurement_due_diligence:category_coverage",
            REQUIRED_PROCUREMENT_DUE_DILIGENCE_CATEGORIES.issubset(ok_categories),
            "procurement due-diligence covers package, policy, commercial release, and GitHub evidence",
            missing=sorted(REQUIRED_PROCUREMENT_DUE_DILIGENCE_CATEGORIES - ok_categories),
        ),
        _check(
            "procurement_due_diligence:html_report",
            html_exists,
            "procurement due-diligence HTML report exists",
            actual=str(html_path) if html_path is not None else None,
        ),
        _check(
            "procurement_due_diligence:html_report_sha256",
            html_exists and isinstance(expected_html_sha, str) and expected_html_sha == actual_html_sha,
            "procurement due-diligence HTML report SHA256 matches manifest",
            expected=expected_html_sha,
            actual=actual_html_sha,
        ),
    ]


def _validate_imports(repo_root: Path, *, require_rust: bool) -> list[dict[str, object]]:
    project_version = _project_version(repo_root)
    checks: list[dict[str, object]] = []
    try:
        package = importlib.import_module("fast_mlsirm")
    except Exception as exc:  # pragma: no cover - exception detail is surfaced in manifest.
        return [_check("import:fast_mlsirm", False, f"failed to import fast_mlsirm: {exc}")]

    imported_version = getattr(package, "__version__", "")
    checks.append(
        _check(
            "import:version",
            imported_version == project_version,
            "installed package version matches pyproject",
            pyproject_version=project_version,
            imported_version=imported_version,
        )
    )

    if require_rust:
        try:
            core = importlib.import_module("fast_mlsirm._core")
        except Exception as exc:  # pragma: no cover - exception detail is surfaced in manifest.
            checks.append(_check("import:rust_core", False, f"failed to import fast_mlsirm._core: {exc}"))
        else:
            checks.append(
                _check(
                    "import:rust_core",
                    hasattr(core, "neg_loglik_and_grad"),
                    "Rust/PyO3 objective symbol is available",
                )
            )
    return checks


def run_sales_readiness(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(args.repo_root).resolve()
    buyer_packet_manifest = getattr(args, "buyer_packet_manifest", None)
    require_buyer_packet = getattr(args, "require_buyer_packet", False)
    benchmark_report = getattr(args, "benchmark_report", None)
    require_benchmark_report = getattr(args, "require_benchmark_report", False)
    release_evidence_index = getattr(args, "release_evidence_index", None)
    require_release_evidence_index = getattr(args, "require_release_evidence_index", False)
    procurement_due_diligence = getattr(args, "procurement_due_diligence", None)
    require_procurement_due_diligence = getattr(args, "require_procurement_due_diligence", False)
    checks: list[dict[str, object]] = []
    checks.extend(_validate_required_files(repo_root))
    checks.extend(_validate_doc_tokens(repo_root))
    if args.require_20b_product:
        checks.extend(_validate_20b_product_evidence(repo_root, contract_value_krw=args.contract_value_krw))
    checks.extend(
        _validate_acceptance_summary(
            Path(args.acceptance).resolve(),
            require_rust=args.require_rust,
            max_acceptance_seconds=args.max_acceptance_seconds,
        )
    )
    checks.extend(_validate_dist(Path(args.dist).resolve() if args.dist else None))
    if buyer_packet_manifest or require_buyer_packet:
        checks.extend(
            _validate_buyer_packet(
                Path(buyer_packet_manifest).resolve() if buyer_packet_manifest else None,
                required=require_buyer_packet,
                contract_value_krw=args.contract_value_krw,
            )
        )
    if benchmark_report or require_benchmark_report:
        checks.extend(
            _validate_benchmark_report(
                Path(benchmark_report).resolve() if benchmark_report else None,
                required=require_benchmark_report,
            )
        )
    if release_evidence_index or require_release_evidence_index:
        checks.extend(
            _validate_release_evidence_index(
                Path(release_evidence_index).resolve() if release_evidence_index else None,
                required=require_release_evidence_index,
                contract_value_krw=args.contract_value_krw,
            )
        )
    if procurement_due_diligence or require_procurement_due_diligence:
        checks.extend(
            _validate_procurement_due_diligence(
                Path(procurement_due_diligence).resolve() if procurement_due_diligence else None,
                required=require_procurement_due_diligence,
                contract_value_krw=args.contract_value_krw,
            )
        )
    if args.check_import:
        checks.extend(_validate_imports(repo_root, require_rust=args.require_rust))

    failed = [check for check in checks if not check["ok"]]
    manifest = {
        "command": "sales_readiness",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
        "require_20b_product": args.require_20b_product,
        "require_buyer_packet": require_buyer_packet,
        "require_benchmark_report": require_benchmark_report,
        "require_release_evidence_index": require_release_evidence_index,
        "require_procurement_due_diligence": require_procurement_due_diligence,
        "repo_root": str(repo_root),
        "acceptance": str(Path(args.acceptance).resolve()),
        "checks": checks,
        "failed_checks": failed,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify enterprise sales-readiness evidence.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing product evidence files.")
    parser.add_argument("--acceptance", required=True, help="Path to release acceptance_summary.json.")
    parser.add_argument("--out", default="sales_readiness_manifest.json", help="Manifest output path.")
    parser.add_argument("--dist", help="Optional dist directory containing built wheel and sdist artifacts.")
    parser.add_argument("--require-rust", action="store_true", help="Require explicit Rust backend evidence.")
    parser.add_argument(
        "--require-20b-product",
        action="store_true",
        help="Require Product Design, Figma, ROI, benchmark, and demo evidence for KRW 2B review.",
    )
    parser.add_argument("--check-import", action="store_true", help="Import installed package and optional Rust core.")
    parser.add_argument("--buyer-packet-manifest", help="Optional buyer_evidence_manifest.json to validate.")
    parser.add_argument(
        "--require-buyer-packet",
        action="store_true",
        help="Fail unless --buyer-packet-manifest points to a complete buyer evidence packet.",
    )
    parser.add_argument("--benchmark-report", help="Optional benchmark_report.json to validate.")
    parser.add_argument(
        "--require-benchmark-report",
        action="store_true",
        help="Fail unless --benchmark-report points to a complete automated benchmark report.",
    )
    parser.add_argument("--release-evidence-index", help="Optional release_evidence_index.json to validate.")
    parser.add_argument(
        "--require-release-evidence-index",
        action="store_true",
        help="Fail unless --release-evidence-index points to a complete release evidence index.",
    )
    parser.add_argument("--procurement-due-diligence", help="Optional procurement_due_diligence_manifest.json to validate.")
    parser.add_argument(
        "--require-procurement-due-diligence",
        action="store_true",
        help="Fail unless --procurement-due-diligence points to complete procurement due-diligence evidence.",
    )
    parser.add_argument("--contract-value-krw", type=int, default=2_000_000_000, help="Target contract value for this gate.")
    parser.add_argument(
        "--max-acceptance-seconds",
        type=float,
        default=None,
        help="Optional maximum release-acceptance runtime budget.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_sales_readiness(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": manifest["status"], "out": str(Path(args.out).resolve())}, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
