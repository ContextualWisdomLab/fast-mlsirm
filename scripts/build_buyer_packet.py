#!/usr/bin/env python
"""Buyer-packet entry point with acceptance-artifact integrity enforcement."""

from __future__ import annotations

from pathlib import Path

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
    return _original_collect_files(
        repo_root=repo_root,
        acceptance_path=acceptance_path,
        sales_readiness_path=sales_readiness_path,
        dist_dir=dist_dir,
        benchmark_report_path=benchmark_report_path,
        release_evidence_index_path=release_evidence_index_path,
        expected_source_commit=expected_source_commit,
    )


_impl._collect_files = _collect_files
build_packet = _impl.build_packet
build_parser = _impl.build_parser
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
