#!/usr/bin/env python
"""Buyer-packet entry point with acceptance-artifact integrity enforcement."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

try:
    from scripts import _build_buyer_packet_impl as _impl
except ModuleNotFoundError:  # executed directly from scripts/
    import _build_buyer_packet_impl as _impl


for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

PRODUCT_DOCS = _impl.PRODUCT_DOCS
PRODUCT_MANIFESTS = _impl.PRODUCT_MANIFESTS
_original_collect_files = _impl._collect_files
_original_build_packet = _impl.build_packet


def _validate_acceptance_artifact_digests(
    acceptance_path: Path, expected_source_commit: str | None
) -> None:
    """Replay acceptance-time digests before artifacts become buyer evidence."""
    if expected_source_commit is None:
        return
    acceptance = _impl._read_json(acceptance_path)
    digest_map = acceptance.get("artifact_sha256")
    if not isinstance(digest_map, dict):
        raise RuntimeError("acceptance artifact SHA256 manifest is missing")

    evidence_root = acceptance_path.parent.resolve()
    for path in _impl._acceptance_artifact_files(acceptance):
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise RuntimeError(f"acceptance artifact is missing: {resolved}")
        try:
            relative = resolved.relative_to(evidence_root)
        except ValueError:
            raise RuntimeError(
                f"acceptance artifact is outside acceptance evidence root: {resolved}"
            ) from None
        relative_name = relative.as_posix()
        expected_digest = digest_map.get(relative_name)
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise RuntimeError(
                f"acceptance artifact SHA256 is missing or invalid: {relative_name}"
            )
        if _impl._sha256(resolved) != expected_digest:
            raise RuntimeError(
                "acceptance artifact SHA256 does not match acceptance_summary.json"
            )


def _validate_sales_readiness_source(
    sales_readiness_path: Path, expected_source_commit: str | None
) -> None:
    """Require source identity before sales evidence enters a source-bound packet."""
    if expected_source_commit is None:
        return
    sales_readiness = _impl._read_json(sales_readiness_path)
    if sales_readiness.get("source_commit") is None:
        raise RuntimeError("sales readiness source commit is missing")


def _validate_optional_source_identity(
    evidence_path: Path | None,
    expected_source_commit: str | None,
    *,
    evidence_name: str,
) -> None:
    """Require source identity for supplied optional evidence in a source-bound packet."""
    if evidence_path is None or expected_source_commit is None:
        return
    evidence = _impl._read_json(evidence_path)
    if evidence.get("source_commit") is None:
        raise RuntimeError(f"{evidence_name} source commit is missing")


def _validate_repository_evidence_source(
    repo_root: Path, expected_source_commit: str | None
) -> None:
    """Require repository-owned buyer evidence to match the advertised source tree."""
    if expected_source_commit is None:
        return
    evidence_paths = [*PRODUCT_DOCS, *PRODUCT_MANIFESTS]
    try:
        completed = _impl.subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
                "--",
                *evidence_paths,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=_impl.GIT_METADATA_TIMEOUT_SECONDS,
        )
    except _impl.subprocess.TimeoutExpired as exc:
        raise RuntimeError("repository-owned buyer evidence source check timed out") from exc
    except (OSError, _impl.subprocess.SubprocessError) as exc:
        raise RuntimeError("repository-owned buyer evidence source check failed") from exc
    if completed.stdout.strip():
        raise RuntimeError(
            "repository-owned buyer evidence does not match source commit: "
            f"{expected_source_commit}"
        )


def _validate_repository_source_commit(
    repo_root: Path, expected_source_commit: object
) -> None:
    """Require repository HEAD to remain the source revision sealed at build start."""
    if (
        not isinstance(expected_source_commit, str)
        or _impl._source_commit(repo_root) != expected_source_commit
    ):
        raise RuntimeError("repository source commit changed during buyer packet build")


def _validate_distribution_artifacts(dist_dir: Path) -> None:
    """Reject indirection before distribution files become buyer evidence."""
    for pattern in ("*.whl", "*.tar.gz"):
        for path in sorted(dist_dir.glob(pattern)):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"distribution artifact must be a regular file: {path}"
                )


def _collect_files(
    *,
    repo_root: Path,
    acceptance_path: Path,
    sales_readiness_path: Path,
    dist_dir: Path,
    benchmark_report_path: Path | None = None,
    release_evidence_index_path: Path | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Path]:
    """Validate acceptance provenance, then delegate canonical packet collection."""
    _impl.PRODUCT_DOCS = PRODUCT_DOCS
    _impl.PRODUCT_MANIFESTS = PRODUCT_MANIFESTS
    _validate_acceptance_artifact_digests(acceptance_path, expected_source_commit)
    _validate_sales_readiness_source(sales_readiness_path, expected_source_commit)
    _validate_optional_source_identity(
        benchmark_report_path,
        expected_source_commit,
        evidence_name="benchmark",
    )
    _validate_optional_source_identity(
        release_evidence_index_path,
        expected_source_commit,
        evidence_name="release evidence",
    )
    _validate_repository_evidence_source(repo_root, expected_source_commit)
    _validate_distribution_artifacts(dist_dir)
    return _original_collect_files(
        repo_root=repo_root,
        acceptance_path=acceptance_path,
        sales_readiness_path=sales_readiness_path,
        dist_dir=dist_dir,
        benchmark_report_path=benchmark_report_path,
        release_evidence_index_path=release_evidence_index_path,
        expected_source_commit=expected_source_commit,
    )


def _archive_entry_sha256(archive: zipfile.ZipFile, archive_path: str) -> str:
    """Hash one archived evidence member without materializing it in memory."""
    digest = hashlib.sha256()
    with archive.open(archive_path, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_entries(
    archive_path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Require archived bytes to match the evidence entries sealed before ZIP writing."""
    expected: dict[str, tuple[int, str]] = {}
    for entry in entries:
        name = entry.get("archive_path")
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or name in expected
        ):
            raise RuntimeError("buyer evidence archive has invalid sealed source entries")
        expected[name] = (size_bytes, sha256)

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise RuntimeError(
                    "buyer evidence archive does not match sealed source entries"
                )
            for name, (size_bytes, sha256) in expected.items():
                info = archive.getinfo(name)
                if (
                    info.file_size != size_bytes
                    or _archive_entry_sha256(archive, name) != sha256
                ):
                    raise RuntimeError(
                        "buyer evidence archive does not match sealed source entries"
                    )
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            "buyer evidence archive does not match sealed source entries"
        ) from exc


def _built_archive_entries(
    manifest: dict[str, Any], args
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct the exact payload and delivery entry contracts from the built manifest."""
    source_entries = manifest.get("files")
    if not isinstance(source_entries, list) or not all(
        isinstance(entry, dict) for entry in source_entries
    ):
        raise RuntimeError("buyer evidence manifest source entries are invalid")
    payload_entries = [dict(entry) for entry in source_entries]

    report_path = Path(str(manifest.get("report_file", "")))
    manifest_path = Path(args.out).resolve() / "buyer_evidence_manifest.json"
    report_digest = manifest.get("report_sha256")
    if (
        not report_path.is_file()
        or not manifest_path.is_file()
        or not isinstance(report_digest, str)
    ):
        raise RuntimeError("buyer evidence delivery metadata is incomplete")

    delivery_entries = [
        *payload_entries,
        {
            "archive_path": "buyer_evidence_report.html",
            "size_bytes": report_path.stat().st_size,
            "sha256": report_digest,
        },
        {
            "archive_path": "buyer_evidence_manifest.json",
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _impl._sha256(manifest_path),
        },
    ]
    return payload_entries, delivery_entries


def build_packet(args):
    """Build a packet and verify ZIP members against the already-sealed manifest entries."""
    manifest = _original_build_packet(args)
    payload_entries, delivery_entries = _built_archive_entries(manifest, args)
    try:
        _validate_archive_entries(Path(manifest["payload_zip_file"]), payload_entries)
        _validate_archive_entries(Path(manifest["packet_file"]), delivery_entries)
        _validate_repository_source_commit(
            Path(args.repo_root).resolve(), manifest.get("source_commit")
        )
    except Exception:
        for candidate in (
            manifest.get("payload_zip_file"),
            manifest.get("packet_file"),
            manifest.get("packet_sha256_file"),
            manifest.get("report_file"),
            str(Path(args.out).resolve() / "buyer_evidence_manifest.json"),
        ):
            if candidate:
                Path(str(candidate)).unlink(missing_ok=True)
        raise
    return manifest


_impl._collect_files = _collect_files
build_parser = _impl.build_parser
main = _impl.main
_impl.build_packet = build_packet


if __name__ == "__main__":
    raise SystemExit(main())
