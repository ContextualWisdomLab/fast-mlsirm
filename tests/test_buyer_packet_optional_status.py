import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE_COMMIT = "a" * 40
OTHER_SOURCE_COMMIT = "b" * 40


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet_optional_status", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_html(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><title>evidence</title>", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    acceptance_path = _write_json(
        tmp_path / "acceptance" / "acceptance_summary.json",
        {"status": "ok", "source_commit": SOURCE_COMMIT, "steps": []},
    )
    sales_path = _write_json(
        tmp_path / "acceptance" / "sales_readiness_manifest.json",
        {"status": "ok"},
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    return repo_root, acceptance_path, sales_path, dist_dir


def test_collect_files_rejects_failed_benchmark_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supplied failed benchmark must not enter an otherwise valid buyer packet."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])
    repo_root, acceptance_path, sales_path, dist_dir = _base_paths(tmp_path)
    html_path = tmp_path / "benchmark" / "benchmark_report.html"
    html_sha = _write_html(html_path)
    benchmark_path = _write_json(
        tmp_path / "benchmark" / "benchmark_report.json",
        {
            "status": "failed",
            "html_report_file": str(html_path),
            "html_report_sha256": html_sha,
        },
    )

    with pytest.raises(RuntimeError, match="benchmark report status is not ok"):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            benchmark_report_path=benchmark_path,
        )


def test_collect_files_rejects_failed_release_evidence_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supplied failed release index must not enter an otherwise valid buyer packet."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])
    repo_root, acceptance_path, sales_path, dist_dir = _base_paths(tmp_path)
    html_path = tmp_path / "release" / "release_evidence_index.html"
    html_sha = _write_html(html_path)
    release_index_path = _write_json(
        tmp_path / "release" / "release_evidence_index.json",
        {
            "status": "failed",
            "html_report_file": str(html_path),
            "html_report_sha256": html_sha,
        },
    )

    with pytest.raises(RuntimeError, match="release evidence index status is not ok"):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            release_evidence_index_path=release_index_path,
        )


def test_collect_files_rejects_cross_revision_benchmark_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly cross-revision benchmark must not be relabeled as current evidence."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])
    repo_root, acceptance_path, sales_path, dist_dir = _base_paths(tmp_path)
    html_path = tmp_path / "benchmark" / "benchmark_report.html"
    html_sha = _write_html(html_path)
    benchmark_path = _write_json(
        tmp_path / "benchmark" / "benchmark_report.json",
        {
            "status": "ok",
            "source_commit": OTHER_SOURCE_COMMIT,
            "html_report_file": str(html_path),
            "html_report_sha256": html_sha,
        },
    )

    with pytest.raises(
        RuntimeError, match="benchmark source commit does not match buyer packet source"
    ):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            benchmark_report_path=benchmark_path,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_collect_files_rejects_cross_revision_release_evidence_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly cross-revision release index must not be relabeled as current evidence."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])
    repo_root, acceptance_path, sales_path, dist_dir = _base_paths(tmp_path)
    html_path = tmp_path / "release" / "release_evidence_index.html"
    html_sha = _write_html(html_path)
    release_index_path = _write_json(
        tmp_path / "release" / "release_evidence_index.json",
        {
            "status": "ok",
            "source_commit": OTHER_SOURCE_COMMIT,
            "html_report_file": str(html_path),
            "html_report_sha256": html_sha,
        },
    )

    with pytest.raises(
        RuntimeError,
        match="release evidence source commit does not match buyer packet source",
    ):
        module._collect_files(
            repo_root=repo_root,
            acceptance_path=acceptance_path,
            sales_readiness_path=sales_path,
            dist_dir=dist_dir,
            release_evidence_index_path=release_index_path,
            expected_source_commit=SOURCE_COMMIT,
        )
