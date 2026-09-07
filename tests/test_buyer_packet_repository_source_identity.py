import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit_repository_evidence(repo: Path, module) -> str:
    for relative in module.PRODUCT_DOCS:
        _write(repo / relative, f"committed evidence: {relative}\n")
    for relative in module.PRODUCT_MANIFESTS:
        _write(repo / relative, json.dumps({"status": "ok"}))

    subprocess.run(
        ["git", "init", "--quiet", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=fast-mlsirm-test",
            "-c",
            "user.email=fast-mlsirm-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "buyer evidence source fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_acceptance(root: Path, source_commit: str) -> Path:
    acceptance_dir = root / "acceptance"
    artifact = _write(acceptance_dir / "artifacts" / "fit_report.html", "<html></html>")
    summary = {
        "status": "ok",
        "source_commit": source_commit,
        "steps": [{"command": "render-report", "files": {"report": str(artifact)}}],
        "artifact_sha256": {
            "artifacts/fit_report.html": hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
    }
    return _write(acceptance_dir / "acceptance_summary.json", json.dumps(summary))


def _build_args(tmp_path: Path, module) -> tuple[argparse.Namespace, Path]:
    repo = tmp_path / "repo"
    source_commit = _commit_repository_evidence(repo, module)
    acceptance = _write_acceptance(tmp_path, source_commit)
    sales = _write(
        tmp_path / "acceptance" / "sales_readiness_manifest.json",
        json.dumps({"status": "ok", "source_commit": source_commit}),
    )
    dist = tmp_path / "dist"
    _write(dist / "fast_mlsirm-0.1.0-py3-none-any.whl", "wheel")
    _write(dist / "fast_mlsirm-0.1.0.tar.gz", "sdist")
    args = argparse.Namespace(
        repo_root=str(repo),
        acceptance=str(acceptance),
        sales_readiness=str(sales),
        dist=str(dist),
        out=str(tmp_path / "packet"),
        contract_value_krw=2_000_000_000,
    )
    return args, repo


def test_source_bound_packet_rejects_dirty_repository_owned_evidence(tmp_path):
    module = _load_packet_builder()
    args, repo = _build_args(tmp_path, module)

    # The packet advertises HEAD as its source identity, so repository-owned
    # procurement evidence must come from that exact tree rather than dirty bytes.
    _write(repo / "README.md", "uncommitted replacement\n")

    with pytest.raises(
        RuntimeError, match="repository-owned buyer evidence does not match source commit"
    ):
        module.build_packet(args)


@pytest.mark.parametrize("archive_write_number", [1, 2])
def test_source_bound_packet_rejects_repository_evidence_mutation_during_archive_write(
    tmp_path, monkeypatch, archive_write_number
):
    module = _load_packet_builder()
    args, repo = _build_args(tmp_path, module)
    original_write_archive = module._impl._write_archive
    write_count = 0

    def mutate_then_write(path, files):
        nonlocal write_count
        write_count += 1
        if write_count == archive_write_number:
            _write(repo / "README.md", "changed after source evidence was sealed\n")
        original_write_archive(path, files)

    monkeypatch.setattr(module._impl, "_write_archive", mutate_then_write)

    with pytest.raises(
        RuntimeError, match="buyer evidence archive does not match sealed source entries"
    ):
        module.build_packet(args)


def test_source_bound_packet_rejects_head_change_during_build(tmp_path, monkeypatch):
    """A packet must not keep an earlier source identity after repository HEAD advances."""
    module = _load_packet_builder()
    args, repo = _build_args(tmp_path, module)
    original_collect_files = module._original_collect_files
    moved = False

    def advance_head_then_collect(**kwargs):
        nonlocal moved
        if not moved:
            moved = True
            _write(repo / "unrelated.txt", "new committed source state\n")
            subprocess.run(
                ["git", "-C", str(repo), "add", "unrelated.txt"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=fast-mlsirm-test",
                    "-c",
                    "user.email=fast-mlsirm-test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "advance source during packet build",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        return original_collect_files(**kwargs)

    monkeypatch.setattr(module, "_original_collect_files", advance_head_then_collect)

    with pytest.raises(
        RuntimeError, match="repository source commit changed during buyer packet build"
    ):
        module.build_packet(args)
