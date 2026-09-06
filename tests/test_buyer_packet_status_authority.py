import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet_status_authority", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _packet_args(
    tmp_path: Path,
    module,
    *,
    acceptance_status: str = "ok",
    sales_status: str = "ok",
) -> argparse.Namespace:
    repo_root = tmp_path / "repo"
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
            "status authority fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    monkey_docs = []
    module.PRODUCT_DOCS = monkey_docs
    module.PRODUCT_MANIFESTS = []

    acceptance_root = tmp_path / "acceptance"
    artifact = acceptance_root / "artifacts" / "fit_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"backend":"rust"}', encoding="utf-8")
    acceptance_path = acceptance_root / "acceptance_summary.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": acceptance_status,
                "source_commit": source_commit,
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
    sales_path.write_text(json.dumps({"status": sales_status}), encoding="utf-8")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "fast_mlsirm-0.0.0-py3-none-any.whl").write_text(
        "wheel", encoding="utf-8"
    )
    (dist_dir / "fast_mlsirm-0.0.0.tar.gz").write_text("sdist", encoding="utf-8")

    return argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance_path),
        sales_readiness=str(sales_path),
        dist=str(dist_dir),
        out=str(tmp_path / "packet"),
        contract_value_krw=None,
        benchmark_report=None,
        release_evidence_index=None,
    )


def test_build_packet_rejects_failed_acceptance_status(tmp_path: Path) -> None:
    """A failed acceptance run must not be repackaged as buyer-ready evidence."""
    module = _load_packet_builder()
    args = _packet_args(tmp_path, module, acceptance_status="failed")

    with pytest.raises(RuntimeError, match="acceptance status is not ok"):
        module.build_packet(args)


def test_build_packet_rejects_failed_sales_readiness_status(tmp_path: Path) -> None:
    """A failed sales gate must not be embedded in a packet whose status is ok."""
    module = _load_packet_builder()
    args = _packet_args(tmp_path, module, sales_status="failed")

    with pytest.raises(RuntimeError, match="sales readiness status is not ok"):
        module.build_packet(args)
