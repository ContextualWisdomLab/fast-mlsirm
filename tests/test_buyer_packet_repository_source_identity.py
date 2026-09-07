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


def test_source_bound_packet_rejects_dirty_repository_owned_evidence(tmp_path):
    module = _load_packet_builder()
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

    # The packet advertises HEAD as its source identity, so repository-owned
    # procurement evidence must come from that exact tree rather than dirty bytes.
    _write(repo / "README.md", "uncommitted replacement\n")

    args = argparse.Namespace(
        repo_root=str(repo),
        acceptance=str(acceptance),
        sales_readiness=str(sales),
        dist=str(dist),
        out=str(tmp_path / "packet"),
        contract_value_krw=2_000_000_000,
    )

    with pytest.raises(
        RuntimeError, match="repository-owned buyer evidence does not match source commit"
    ):
        module.build_packet(args)
