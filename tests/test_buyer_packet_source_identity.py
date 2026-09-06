import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet_source_identity", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _initialize_repo(repo_root: Path) -> str:
    repo_root.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=fast-mlsirm-test",
            "-c",
            "user.email=fast-mlsirm-test@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "source identity fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_build_packet_rejects_acceptance_from_different_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buyer evidence must not relabel acceptance from another source revision."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])

    repo_root = tmp_path / "repo"
    current_source = _initialize_repo(repo_root)
    assert current_source != "0" * 40

    acceptance_root = tmp_path / "acceptance"
    artifact = acceptance_root / "artifacts" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"backend":"rust"}', encoding="utf-8")
    acceptance_path = acceptance_root / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_commit": "0" * 40,
                "steps": [
                    {
                        "command": "fit",
                        "files": {"summary": str(artifact)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sales_path = acceptance_root / "sales_readiness_manifest.json"
    sales_path.write_text('{"status":"ok"}', encoding="utf-8")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "fast_mlsirm-0.0.0-py3-none-any.whl").write_text(
        "wheel", encoding="utf-8"
    )
    (dist_dir / "fast_mlsirm-0.0.0.tar.gz").write_text("sdist", encoding="utf-8")

    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance_path),
        sales_readiness=str(sales_path),
        dist=str(dist_dir),
        out=str(tmp_path / "packet"),
        contract_value_krw=None,
        benchmark_report=None,
        release_evidence_index=None,
    )

    with pytest.raises(RuntimeError, match="acceptance source commit does not match buyer packet source"):
        module.build_packet(args)


def test_build_packet_rejects_sales_readiness_from_different_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buyer evidence must not combine readiness from another source revision."""
    module = _load_packet_builder()
    monkeypatch.setattr(module, "PRODUCT_DOCS", [])
    monkeypatch.setattr(module, "PRODUCT_MANIFESTS", [])

    repo_root = tmp_path / "repo"
    current_source = _initialize_repo(repo_root)
    assert current_source != "0" * 40

    acceptance_root = tmp_path / "acceptance"
    artifact = acceptance_root / "artifacts" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"backend":"rust"}', encoding="utf-8")
    acceptance_path = acceptance_root / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_commit": current_source,
                "steps": [
                    {
                        "command": "fit",
                        "files": {"summary": str(artifact)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sales_path = acceptance_root / "sales_readiness_manifest.json"
    sales_path.write_text(
        json.dumps({"status": "ok", "source_commit": "0" * 40}),
        encoding="utf-8",
    )

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "fast_mlsirm-0.0.0-py3-none-any.whl").write_text(
        "wheel", encoding="utf-8"
    )
    (dist_dir / "fast_mlsirm-0.0.0.tar.gz").write_text("sdist", encoding="utf-8")

    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance_path),
        sales_readiness=str(sales_path),
        dist=str(dist_dir),
        out=str(tmp_path / "packet"),
        contract_value_krw=None,
        benchmark_report=None,
        release_evidence_index=None,
    )

    with pytest.raises(RuntimeError, match="sales readiness source commit does not match buyer packet source"):
        module.build_packet(args)
