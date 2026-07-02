#!/usr/bin/env python
"""Enterprise sales-readiness verifier for fast-mlsirm.

This verifier is intentionally evidence-oriented. It does not claim that a
sale is guaranteed; it checks that a release candidate carries the artifacts,
scope statements, backend proof, and acceptance results expected before a
high-value enterprise procurement review.
"""

from __future__ import annotations

import argparse
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

REQUIRED_DOC_TOKENS = {
    "README.md": [
        "Commercial Readiness",
        "Enterprise Sales Readiness",
        "scripts/release_acceptance.py",
        "scripts/sales_readiness.py",
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
        "--require-rust",
    ],
}

REQUIRED_ACCEPTANCE_COMMANDS = {
    "simulate",
    "fit",
    "diagnose-fit",
    "diagnose-dimensions",
    "render-report",
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
    checks: list[dict[str, object]] = []
    checks.extend(_validate_required_files(repo_root))
    checks.extend(_validate_doc_tokens(repo_root))
    checks.extend(
        _validate_acceptance_summary(
            Path(args.acceptance).resolve(),
            require_rust=args.require_rust,
            max_acceptance_seconds=args.max_acceptance_seconds,
        )
    )
    checks.extend(_validate_dist(Path(args.dist).resolve() if args.dist else None))
    if args.check_import:
        checks.extend(_validate_imports(repo_root, require_rust=args.require_rust))

    failed = [check for check in checks if not check["ok"]]
    manifest = {
        "command": "sales_readiness",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
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
    parser.add_argument("--check-import", action="store_true", help="Import installed package and optional Rust core.")
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
