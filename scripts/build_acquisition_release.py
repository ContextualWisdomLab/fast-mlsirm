#!/usr/bin/env python
"""Build a price-neutral acquisition-readiness evidence bundle for fast-mlsirm."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts import (
        build_benchmark_report,
        build_buyer_packet,
        build_figma_evidence_sync,
        build_pr_queue_governance,
        build_procurement_due_diligence,
        build_release_evidence_index,
        sales_readiness,
    )
    from scripts._bounded_subprocess import run_bounded_capture
    from scripts.release_acceptance import (
        RELEASE_ACCEPTANCE_TIMEOUT_SECONDS as _INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "scripts",
        "scripts._bounded_subprocess",
        "scripts.build_benchmark_report",
        "scripts.build_buyer_packet",
        "scripts.build_figma_evidence_sync",
        "scripts.build_pr_queue_governance",
        "scripts.build_procurement_due_diligence",
        "scripts.build_release_evidence_index",
        "scripts.release_acceptance",
        "scripts.sales_readiness",
    }:
        raise
    import build_benchmark_report
    import build_buyer_packet
    import build_figma_evidence_sync
    import build_pr_queue_governance
    import build_procurement_due_diligence
    import build_release_evidence_index
    import sales_readiness
    from _bounded_subprocess import run_bounded_capture
    from release_acceptance import (
        RELEASE_ACCEPTANCE_TIMEOUT_SECONDS as _INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS,
    )


_STAGE_TIMEOUT_SECONDS = 900.0
_RELEASE_ACCEPTANCE_ORCHESTRATION_MARGIN_SECONDS = 60.0
_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS = (
    sum(_INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS.values())
    + _RELEASE_ACCEPTANCE_ORCHESTRATION_MARGIN_SECONDS
)
_MAX_MANIFEST_BYTES = 10 * 1024 * 1024


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    """Describe one required acquisition artifact without inventing evidence."""
    exists = path.exists() and path.is_file()
    return {
        "path": str(path),
        "name": path.name,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
    }


def _source_commit(repo_root: Path) -> str:
    """Return the exact checked-out Git commit or fail closed."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    commit = completed.stdout.strip()
    if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
        raise RuntimeError("source commit lookup returned an invalid object id")
    return commit


def _admit_output_root(repo_root: Path, output_root: Path) -> Path:
    """Reject output roots whose untracked-file exception could cover source."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root == repo_root or output_root in repo_root.parents:
        raise RuntimeError("output directory must not be the repository root or an ancestor")
    return output_root


def _require_clean_source(
    repo_root: Path,
    *,
    allowed_untracked_root: Path | None = None,
) -> None:
    """Require a clean Git source tree, except generated files under one output root."""
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    allowed = allowed_untracked_root.resolve() if allowed_untracked_root is not None else None
    violations: list[str] = []
    for record in completed.stdout.split("\0"):
        if not record:
            continue
        if len(record) < 4:
            violations.append(record)
            continue
        status = record[:2]
        relative_path = record[3:]
        if status != "??":
            violations.append(record)
            continue
        candidate = (repo_root / relative_path).resolve()
        if allowed is not None and (candidate == allowed or allowed in candidate.parents):
            continue
        violations.append(record)
    if violations:
        preview = ", ".join(violations[:5])
        raise RuntimeError(f"source working tree is not clean: {preview}")


def _assert_source_unchanged(
    repo_root: Path,
    expected: str,
    *,
    allowed_untracked_root: Path | None = None,
) -> None:
    """Fail closed if source HEAD or working-tree identity changed after sealing."""
    actual = _source_commit(repo_root)
    if actual != expected:
        raise RuntimeError(
            f"source HEAD changed during acquisition run: expected {expected}, observed {actual}"
        )
    _require_clean_source(repo_root, allowed_untracked_root=allowed_untracked_root)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic human-readable JSON evidence document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one bounded JSON manifest object for provenance validation."""
    if not path.is_file():
        raise RuntimeError(f"required stage manifest is missing: {path}")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise RuntimeError(f"stage manifest exceeds {_MAX_MANIFEST_BYTES} bytes: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"stage manifest is unreadable or malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"stage manifest must be a JSON object: {path}")
    return payload


def _require_source_identity(
    name: str,
    payload: dict[str, Any],
    expected: str,
    *,
    required: bool = True,
) -> None:
    """Require a stage's available source identity to match the sealed commit."""
    actual = payload.get("source_commit")
    if actual is None:
        if required:
            raise RuntimeError(f"{name} missing source_commit")
        return
    if not isinstance(actual, str) or len(actual) not in {40, 64} or any(
        ch not in "0123456789abcdef" for ch in actual
    ):
        raise RuntimeError(f"{name} has malformed source_commit")
    if actual != expected:
        raise RuntimeError(
            f"{name} source_commit mismatch: expected {expected}, observed {actual}"
        )


def _require_manifest_source_identity(
    name: str,
    path: Path,
    expected: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Read a generated manifest and verify its source identity when required."""
    payload = _read_manifest(path)
    _require_source_identity(name, payload, expected, required=required)
    return payload


def _is_release_acceptance_command(command: list[str]) -> bool:
    """Return whether a stage command invokes the bounded release acceptance suite."""
    return len(command) >= 2 and Path(command[1]).name == "release_acceptance.py"


def _run(command: list[str], *, cwd: Path) -> None:
    """Run one bounded subprocess stage and surface a stable failure."""
    timeout_seconds = (
        _RELEASE_ACCEPTANCE_TIMEOUT_SECONDS
        if _is_release_acceptance_command(command)
        else _STAGE_TIMEOUT_SECONDS
    )
    completed = run_bounded_capture(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=10 * 1024 * 1024,
        max_stderr_bytes=10 * 1024 * 1024,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"stage failed ({completed.returncode}): {' '.join(command)}"
            + (f"\n{stderr[-2000:]}" if stderr else "")
        )


def _select_distribution_directory(args: argparse.Namespace, out_dir: Path) -> Path:
    """Return isolated build output or the explicit caller-supplied reuse directory."""
    configured = Path(args.dist).resolve()
    if args.skip_build:
        return configured

    distribution = out_dir / "distribution"
    if distribution.exists():
        if distribution.is_symlink() or not distribution.is_dir():
            raise RuntimeError(
                "run-specific distribution path must be a real directory when it exists"
            )
        shutil.rmtree(distribution)
    distribution.mkdir(parents=True, exist_ok=False)
    return distribution


def _select_candidate_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    """Select exactly one real wheel and one real source distribution."""
    wheels = tuple(
        path
        for path in sorted(dist_dir.glob("*.whl"))
        if path.is_file() and not path.is_symlink()
    )
    sdists = tuple(
        path
        for path in sorted(dist_dir.glob("*.tar.gz"))
        if path.is_file() and not path.is_symlink()
    )
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            "candidate release must contain exactly one wheel and one source distribution"
        )
    return wheels[0], sdists[0]


def _prepare_candidate_environment(*, wheel: Path, out_dir: Path, python: str) -> Path:
    """Install the selected wheel into the interpreter used by release acceptance.

    The venv inherits only the caller interpreter's dependency environment; the
    fast-mlsirm distribution itself is forcibly replaced by the exact selected
    wheel. This keeps acceptance offline-capable while preventing repository
    source or another installed fast-mlsirm build from becoming the candidate.
    """
    wheel = wheel.resolve()
    if wheel.is_symlink() or not wheel.is_file():
        raise RuntimeError("candidate wheel must be a regular file")
    environment = out_dir / "candidate-runtime-env"
    if environment.exists():
        if environment.is_symlink() or not environment.is_dir():
            raise RuntimeError("candidate runtime environment must be a real directory")
        shutil.rmtree(environment)
    _run(
        [python, "-m", "venv", "--system-site-packages", str(environment)],
        cwd=out_dir,
    )
    environment_python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    _run(
        [
            str(environment_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=out_dir,
    )
    return environment_python


def _verify_candidate_import(
    *,
    wheel: Path,
    out_dir: Path,
    source_commit: str,
    require_rust: bool,
    python: str,
    environment_python: Path | None = None,
) -> dict[str, Any]:
    """Probe the exact candidate runtime and persist wheel-bound import evidence."""
    wheel = wheel.resolve()
    if environment_python is None:
        environment_python = _prepare_candidate_environment(
            wheel=wheel, out_dir=out_dir, python=python
        )
    environment_python = environment_python.resolve()
    environment = environment_python.parent.parent
    probe = (
        "import importlib, importlib.metadata, json\n"
        "package = importlib.import_module('fast_mlsirm')\n"
        "package_version = str(getattr(package, '__version__', ''))\n"
        "distribution_version = importlib.metadata.version('fast-mlsirm')\n"
        "try:\n"
        "    core = importlib.import_module('fast_mlsirm._core')\n"
        "    rust_core = bool(hasattr(core, 'neg_loglik_and_grad'))\n"
        "except (ImportError, OSError):\n"
        "    rust_core = False\n"
        "print(json.dumps({\n"
        "    'package_version': package_version,\n"
        "    'distribution_version': distribution_version,\n"
        "    'package_file': str(package.__file__ or ''),\n"
        "    'rust_core': rust_core,\n"
        "}))\n"
    )
    completed = run_bounded_capture(
        [str(environment_python), "-c", probe],
        cwd=out_dir,
        timeout_seconds=60.0,
        max_stdout_bytes=1024 * 1024,
        max_stderr_bytes=1024 * 1024,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "candidate wheel import probe failed"
            + (f": {stderr[-2000:]}" if stderr else "")
        )
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("candidate wheel import probe returned malformed JSON") from exc
    if not isinstance(observed, dict):
        raise RuntimeError("candidate wheel import probe must return a JSON object")

    package_version = observed.get("package_version")
    distribution_version = observed.get("distribution_version")
    package_file_raw = observed.get("package_file")
    rust_core = observed.get("rust_core") is True
    if not isinstance(package_version, str) or not package_version:
        raise RuntimeError("candidate wheel import probe returned no package version")
    if not isinstance(distribution_version, str) or not distribution_version:
        raise RuntimeError("candidate wheel import probe returned no distribution version")
    if package_version != distribution_version:
        raise RuntimeError(
            "candidate package/distribution version mismatch: "
            f"{package_version!r} != {distribution_version!r}"
        )
    if not isinstance(package_file_raw, str) or not package_file_raw:
        raise RuntimeError("candidate wheel import probe returned no package file")
    package_file = Path(package_file_raw).resolve()
    environment_root = environment.resolve()
    if environment_root != package_file and environment_root not in package_file.parents:
        raise RuntimeError("candidate package import did not come from the candidate environment")
    if require_rust and not rust_core:
        raise RuntimeError("candidate wheel is missing the required Rust core")

    evidence: dict[str, Any] = {
        "command": "verify_candidate_import",
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source_commit,
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "runtime_python": str(environment_python),
        "package_version": package_version,
        "distribution_version": distribution_version,
        "package_file": str(package_file),
        "rust_core": rust_core,
    }
    _write_json(out_dir / "candidate_import_evidence.json", evidence)
    return evidence


def _set_contract_value(namespace: argparse.Namespace, value: int | None) -> argparse.Namespace:
    """Override legacy helper CLI defaults with the explicit transaction scenario."""
    if hasattr(namespace, "contract_value_krw"):
        namespace.contract_value_krw = value
    return namespace


def _sales_args(
    *,
    repo_root: Path,
    acceptance: Path,
    dist: Path,
    out: Path,
    benchmark: Path | None = None,
    buyer_packet: Path | None = None,
    release_index: Path | None = None,
    procurement: Path | None = None,
    pr_queue: Path | None = None,
    figma: Path | None = None,
    require_rust: bool,
    check_import: bool,
    contract_value_krw: int | None,
    acquisition: bool,
) -> argparse.Namespace:
    """Build one sales-readiness namespace without the deprecated 20B profile."""
    argv = [
        "--repo-root",
        str(repo_root),
        "--acceptance",
        str(acceptance),
        "--dist",
        str(dist),
        "--out",
        str(out),
    ]
    if benchmark is not None:
        argv.extend(["--benchmark-report", str(benchmark), "--require-benchmark-report"])
    if buyer_packet is not None:
        argv.extend(["--buyer-packet-manifest", str(buyer_packet), "--require-buyer-packet"])
    if release_index is not None:
        argv.extend(
            [
                "--release-evidence-index",
                str(release_index),
                "--require-release-evidence-index",
            ]
        )
    if procurement is not None:
        argv.extend(
            [
                "--procurement-due-diligence",
                str(procurement),
                "--require-procurement-due-diligence",
            ]
        )
    if pr_queue is not None:
        argv.extend(
            [
                "--pr-queue-governance",
                str(pr_queue),
                "--require-pr-queue-governance",
            ]
        )
    if figma is not None:
        argv.extend(
            [
                "--figma-evidence-sync",
                str(figma),
                "--require-figma-evidence-sync",
            ]
        )
    if require_rust:
        argv.append("--require-rust")
    if check_import:
        argv.append("--check-import")
    if acquisition:
        argv.append("--require-acquisition-readiness")
    if contract_value_krw is not None:
        argv.extend(["--contract-value-krw", str(contract_value_krw)])
    namespace = sales_readiness.build_parser().parse_args(argv)
    if namespace.require_20b_product:
        raise RuntimeError("generic acquisition builder must not enable legacy 20B compatibility")
    return namespace


def _require_ok(name: str, payload: dict[str, Any]) -> None:
    """Fail closed when a direct evidence builder reports a failed status."""
    if payload.get("status") != "ok":
        raise RuntimeError(f"{name} reported status={payload.get('status')!r}")


def _verify_generated_stage(
    name: str,
    payload: dict[str, Any],
    path: Path,
    source_commit: str,
    repo_root: Path,
    *,
    allowed_untracked_root: Path | None = None,
) -> None:
    """Validate a generated stage payload, persisted manifest, and live source."""
    _require_ok(name, payload)
    require_source = "sales_readiness" not in name
    _require_source_identity(name, payload, source_commit, required=require_source)
    _require_manifest_source_identity(
        name, path, source_commit, required=require_source
    )
    _assert_source_unchanged(
        repo_root,
        source_commit,
        allowed_untracked_root=allowed_untracked_root,
    )


def _required_packet_artifact(payload: dict[str, Any], field: str) -> Path:
    """Resolve one delivery artifact from the generated buyer-packet contract."""
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"buyer packet missing {field}")
    path = Path(value).resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"buyer packet {field} must be a regular file")
    return path


def build_acquisition_release(args: argparse.Namespace) -> dict[str, Any]:
    """Build one exact-source, price-neutral acquisition-readiness evidence bundle."""
    repo_root = Path(args.repo_root).resolve()
    out_dir = _admit_output_root(repo_root, Path(args.out))
    _require_clean_source(repo_root, allowed_untracked_root=out_dir)
    source_commit = _source_commit(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = _select_distribution_directory(args, out_dir)

    if not args.skip_build:
        _run(
            [args.python, "-m", "build", "--outdir", str(dist_dir)],
            cwd=repo_root,
        )
    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )
    wheel, sdist = _select_candidate_artifacts(dist_dir)

    candidate_python = _prepare_candidate_environment(
        wheel=wheel, out_dir=out_dir, python=args.python
    )
    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )

    candidate_import_path = out_dir / "candidate_import_evidence.json"
    candidate_import: dict[str, Any] | None = None
    if args.check_import:
        candidate_import = _verify_candidate_import(
            wheel=wheel,
            out_dir=out_dir,
            source_commit=source_commit,
            require_rust=args.require_rust,
            python=args.python,
            environment_python=candidate_python,
        )
        _require_ok("candidate_import", candidate_import)
        _require_source_identity("candidate_import", candidate_import, source_commit)
        _require_manifest_source_identity(
            "candidate_import", candidate_import_path, source_commit
        )
        _assert_source_unchanged(
            repo_root, source_commit, allowed_untracked_root=out_dir
        )

    acceptance_dir = out_dir / "release-acceptance"
    acceptance_command = [
        str(candidate_python),
        str(repo_root / "scripts" / "release_acceptance.py"),
        "--out",
        str(acceptance_dir),
        "--distribution-root",
        str(dist_dir),
    ]
    if args.require_rust:
        acceptance_command.append("--require-rust")
    _run(acceptance_command, cwd=repo_root)
    acceptance_path = acceptance_dir / "acceptance_summary.json"
    _require_manifest_source_identity(
        "release_acceptance", acceptance_path, source_commit, required=False
    )
    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )

    benchmark_dir = acceptance_dir / "benchmark"
    benchmark_args = build_benchmark_report.build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--acceptance",
            str(acceptance_path),
            "--out",
            str(benchmark_dir),
        ]
    )
    benchmark = build_benchmark_report.build_report(benchmark_args)
    benchmark_path = benchmark_dir / "benchmark_report.json"
    _verify_generated_stage(
        "benchmark_report",
        benchmark,
        benchmark_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    initial_sales_path = acceptance_dir / "sales_readiness_manifest.json"
    initial_sales = sales_readiness.run_sales_readiness(
        _sales_args(
            repo_root=repo_root,
            acceptance=acceptance_path,
            dist=dist_dir,
            out=initial_sales_path,
            benchmark=benchmark_path,
            require_rust=args.require_rust,
            check_import=False,
            contract_value_krw=args.contract_value_krw,
            acquisition=False,
        )
    )
    _verify_generated_stage(
        "initial_sales_readiness",
        initial_sales,
        initial_sales_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    packet_dir = out_dir / "buyer-evidence-packet"
    packet_args = build_buyer_packet.build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--acceptance",
            str(acceptance_path),
            "--sales-readiness",
            str(initial_sales_path),
            "--dist",
            str(dist_dir),
            "--benchmark-report",
            str(benchmark_path),
            "--out",
            str(packet_dir),
        ]
    )
    packet_args = _set_contract_value(packet_args, args.contract_value_krw)
    packet = build_buyer_packet.build_packet(packet_args)
    packet_path = packet_dir / "buyer_evidence_manifest.json"
    _verify_generated_stage(
        "buyer_packet",
        packet,
        packet_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )
    buyer_delivery = _required_packet_artifact(packet, "packet_file")
    buyer_delivery_digest = _required_packet_artifact(packet, "packet_sha256_file")

    index_dir = out_dir / "release-evidence-index"
    index_args = build_release_evidence_index.build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--acceptance",
            str(acceptance_path),
            "--sales-readiness",
            str(initial_sales_path),
            "--dist",
            str(dist_dir),
            "--benchmark-report",
            str(benchmark_path),
            "--buyer-packet-manifest",
            str(packet_path),
            "--out",
            str(index_dir),
        ]
    )
    index_args = _set_contract_value(index_args, args.contract_value_krw)
    release_index = build_release_evidence_index.build_index(index_args)
    index_path = index_dir / "release_evidence_index.json"
    _verify_generated_stage(
        "release_evidence_index",
        release_index,
        index_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    preproc_sales_path = acceptance_dir / "preproc_sales_readiness_manifest.json"
    preproc_sales = sales_readiness.run_sales_readiness(
        _sales_args(
            repo_root=repo_root,
            acceptance=acceptance_path,
            dist=dist_dir,
            out=preproc_sales_path,
            benchmark=benchmark_path,
            buyer_packet=packet_path,
            release_index=index_path,
            require_rust=args.require_rust,
            check_import=False,
            contract_value_krw=args.contract_value_krw,
            acquisition=False,
        )
    )
    _verify_generated_stage(
        "preproc_sales_readiness",
        preproc_sales,
        preproc_sales_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    candidate_manifest_path = out_dir / "acquisition_candidate_manifest.json"
    candidate_artifacts: dict[str, dict[str, Any]] = {
        "wheel": _artifact(wheel),
        "sdist": _artifact(sdist),
        "final_sales_readiness": _artifact(preproc_sales_path),
        "buyer_packet_delivery": _artifact(buyer_delivery),
        "buyer_packet_delivery_digest": _artifact(buyer_delivery_digest),
    }
    if args.check_import:
        candidate_artifacts["candidate_import_evidence"] = _artifact(candidate_import_path)
    candidate_manifest = {
        "command": "build_acquisition_release_candidate",
        "status": "ok",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source_commit,
        "acceptance_runtime": {
            "python": str(candidate_python),
            "wheel": str(wheel),
            "wheel_sha256": _sha256(wheel),
        },
        "artifacts": candidate_artifacts,
    }
    if not all(item["exists"] for item in candidate_artifacts.values()):
        raise RuntimeError("candidate release is missing a required distribution/readiness artifact")
    _write_json(candidate_manifest_path, candidate_manifest)
    _require_manifest_source_identity(
        "acquisition_candidate", candidate_manifest_path, source_commit
    )
    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )

    procurement_dir = out_dir / "procurement-due-diligence"
    procurement_argv = [
        "--repo-root",
        str(repo_root),
        "--dist",
        str(dist_dir),
        "--commercial-release-manifest",
        str(candidate_manifest_path),
        "--out",
        str(procurement_dir),
        "--repo",
        args.repo,
    ]
    if args.offline_github:
        procurement_argv.append("--offline-github")
    procurement_args = build_procurement_due_diligence.build_parser().parse_args(
        procurement_argv
    )
    procurement_args = _set_contract_value(procurement_args, args.contract_value_krw)
    procurement = build_procurement_due_diligence.build_procurement_due_diligence(
        procurement_args
    )
    procurement_path = procurement_dir / "procurement_due_diligence_manifest.json"
    _verify_generated_stage(
        "procurement_due_diligence",
        procurement,
        procurement_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    pr_queue_dir = out_dir / "pr-queue-governance"
    pr_queue_argv = [
        "--repo-root",
        str(repo_root),
        "--out",
        str(pr_queue_dir),
        "--repo",
        args.repo,
        "--max-stale-days",
        str(args.pr_queue_max_stale_days),
    ]
    if args.pr_queue_offline_snapshot:
        pr_queue_argv.extend(["--offline-snapshot", args.pr_queue_offline_snapshot])
    pr_queue_args = build_pr_queue_governance.build_parser().parse_args(pr_queue_argv)
    pr_queue_args = _set_contract_value(pr_queue_args, args.contract_value_krw)
    pr_queue = build_pr_queue_governance.build_pr_queue_governance(pr_queue_args)
    pr_queue_path = pr_queue_dir / "pr_queue_governance_manifest.json"
    _verify_generated_stage(
        "pr_queue_governance",
        pr_queue,
        pr_queue_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    figma_dir = out_dir / "figma-evidence-sync"
    figma_argv = [
        "--repo-root",
        str(repo_root),
        "--packet",
        str(repo_root / "examples" / "enterprise_demo" / "figma_design_packet.json"),
        "--out",
        str(figma_dir),
        "--figma-url",
        args.figma_url,
    ]
    if args.figma_metadata_snapshot:
        figma_argv.extend(["--metadata-snapshot", args.figma_metadata_snapshot])
    figma_args = build_figma_evidence_sync.build_parser().parse_args(figma_argv)
    figma_args = _set_contract_value(figma_args, args.contract_value_krw)
    figma = build_figma_evidence_sync.build_figma_evidence_sync(figma_args)
    figma_path = figma_dir / "figma_evidence_sync_manifest.json"
    _verify_generated_stage(
        "figma_evidence_sync",
        figma,
        figma_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    final_sales_path = acceptance_dir / "final_acquisition_readiness_manifest.json"
    final_sales = sales_readiness.run_sales_readiness(
        _sales_args(
            repo_root=repo_root,
            acceptance=acceptance_path,
            dist=dist_dir,
            out=final_sales_path,
            benchmark=benchmark_path,
            buyer_packet=packet_path,
            release_index=index_path,
            procurement=procurement_path,
            pr_queue=pr_queue_path,
            figma=figma_path,
            require_rust=args.require_rust,
            check_import=False,
            contract_value_krw=args.contract_value_krw,
            acquisition=True,
        )
    )
    _verify_generated_stage(
        "final_acquisition_readiness",
        final_sales,
        final_sales_path,
        source_commit,
        repo_root,
        allowed_untracked_root=out_dir,
    )

    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )
    manifest_path = out_dir / "acquisition_release_manifest.json"
    artifacts: dict[str, dict[str, Any]] = {
        "acceptance_summary": _artifact(acceptance_path),
        "benchmark_report": _artifact(benchmark_path),
        "buyer_packet": _artifact(packet_path),
        "buyer_packet_delivery": _artifact(buyer_delivery),
        "buyer_packet_delivery_digest": _artifact(buyer_delivery_digest),
        "release_evidence_index": _artifact(index_path),
        "procurement_due_diligence": _artifact(procurement_path),
        "pr_queue_governance": _artifact(pr_queue_path),
        "figma_evidence_sync": _artifact(figma_path),
        "final_acquisition_readiness": _artifact(final_sales_path),
        "wheel": _artifact(wheel),
        "sdist": _artifact(sdist),
    }
    if args.check_import:
        artifacts["candidate_import_evidence"] = _artifact(candidate_import_path)
    manifest: dict[str, Any] = {
        "command": "build_acquisition_release",
        "status": "ok",
        "contract_value_krw": args.contract_value_krw,
        "transaction_scenario": (
            {"contract_value_krw": args.contract_value_krw}
            if args.contract_value_krw is not None
            else None
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source_commit,
        "repo_root": str(repo_root),
        "out": str(out_dir),
        "acceptance_runtime": {
            "python": str(candidate_python),
            "wheel": str(wheel),
            "wheel_sha256": _sha256(wheel),
        },
        "artifacts": artifacts,
    }
    if not all(item["exists"] for item in artifacts.values()):
        raise RuntimeError("final acquisition bundle is missing required evidence")
    _write_json(manifest_path, manifest)
    _require_manifest_source_identity("acquisition_release", manifest_path, source_commit)
    _assert_source_unchanged(
        repo_root, source_commit, allowed_untracked_root=out_dir
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the price-neutral acquisition-release CLI."""
    parser = argparse.ArgumentParser(
        description="Build the complete fast-mlsirm acquisition-readiness evidence bundle."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--out", default="acquisition-release", help="Evidence output directory.")
    parser.add_argument("--dist", default="dist", help="Distribution artifact directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess stages.")
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=None,
        help="Optional transaction-scenario value in KRW; omitted for the generic readiness gate.",
    )
    parser.add_argument("--require-rust", action="store_true", help="Require explicit Rust backend evidence.")
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="Probe imports from the same exact-wheel runtime used by release acceptance.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Use existing distribution artifacts.")
    parser.add_argument("--offline-github", action="store_true", help="Use the procurement tool's offline GitHub mode.")
    parser.add_argument("--pr-queue-offline-snapshot", help="Optional PR queue snapshot JSON.")
    parser.add_argument("--pr-queue-max-stale-days", type=int, default=14, help="Open-PR staleness threshold.")
    parser.add_argument("--figma-metadata-snapshot", help="Optional exported live Figma metadata snapshot JSON.")
    parser.add_argument(
        "--figma-url",
        default="https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem",
        help="Fallback Figma design URL.",
    )
    parser.add_argument(
        "--repo",
        default="ContextualWisdomLab/fast-mlsirm",
        help="GitHub repository used by due-diligence evidence stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run acquisition evidence generation and emit a compact JSON summary."""
    args = build_parser().parse_args(argv)
    try:
        manifest = build_acquisition_release(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(Path(args.out).resolve() / "acquisition_release_manifest.json"),
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
