import argparse
import importlib.util
import json
import zipfile
from pathlib import Path


def _load_packet_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "ok") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_repo_evidence(root: Path, module) -> None:
    for relative in module.PRODUCT_DOCS:
        _write(root / relative)
    for relative in module.PRODUCT_MANIFESTS:
        payload = {"status": "ok"}
        if relative.endswith("product_completion_manifest.json"):
            payload = {
                "contract_value_krw": 2_000_000_000,
                "checks": [{"id": "buyer_evidence_packet", "status": "go"}],
            }
        _write(root / relative, json.dumps(payload))


def _write_acceptance(tmp_path: Path) -> Path:
    artifacts = tmp_path / "acceptance" / "artifacts"
    summary = {
        "status": "ok",
        "steps": [
            {
                "command": "simulate",
                "files": {
                    "responses": _write(artifacts / "simulate" / "responses.npy"),
                    "factors": _write(artifacts / "simulate" / "item_factor.csv"),
                },
            },
            {
                "command": "render-report",
                "files": {"report": _write(artifacts / "fit_report.html", "<html></html>")},
            },
        ],
    }
    path = tmp_path / "acceptance" / "acceptance_summary.json"
    _write(path, json.dumps(summary))
    return path


def _write_dist(dist: Path) -> None:
    _write(dist / "fast_mlsirm-0.1.0-py3-none-any.whl", "wheel")
    _write(dist / "fast_mlsirm-0.1.0.tar.gz", "sdist")


def test_build_buyer_packet_creates_manifest_and_zip(tmp_path):
    module = _load_packet_builder()
    repo = tmp_path / "repo"
    dist = tmp_path / "dist"
    out = tmp_path / "packet"
    _write_repo_evidence(repo, module)
    _write_dist(dist)
    acceptance = _write_acceptance(tmp_path)
    sales = tmp_path / "acceptance" / "sales_readiness_manifest.json"
    _write(sales, json.dumps({"status": "ok"}))
    args = argparse.Namespace(
        repo_root=str(repo),
        acceptance=str(acceptance),
        sales_readiness=str(sales),
        dist=str(dist),
        out=str(out),
        contract_value_krw=2_000_000_000,
    )

    manifest = module.build_packet(args)

    assert manifest["status"] == "ok"
    assert manifest["contract_value_krw"] == 2_000_000_000
    assert manifest["coverage"]["wheel"] is True
    assert manifest["coverage"]["sdist"] is True
    assert manifest["zip_sha256"]
    with zipfile.ZipFile(manifest["zip_file"]) as packet:
        names = set(packet.namelist())
    assert "buyer_evidence_manifest.json" in names
    assert "acceptance/acceptance_summary.json" in names
    assert "sales/sales_readiness_manifest.json" in names
    assert "docs/20b_product_readiness.md" in names
    assert "examples/enterprise_demo/product_completion_manifest.json" in names
    assert any(name.endswith(".whl") for name in names)
    assert any(name.endswith(".tar.gz") for name in names)


def test_build_buyer_packet_fails_without_source_distribution(tmp_path):
    module = _load_packet_builder()
    repo = tmp_path / "repo"
    dist = tmp_path / "dist"
    out = tmp_path / "packet"
    _write_repo_evidence(repo, module)
    _write(dist / "fast_mlsirm-0.1.0-py3-none-any.whl", "wheel")
    acceptance = _write_acceptance(tmp_path)
    sales = tmp_path / "acceptance" / "sales_readiness_manifest.json"
    _write(sales, json.dumps({"status": "ok"}))
    args = argparse.Namespace(
        repo_root=str(repo),
        acceptance=str(acceptance),
        sales_readiness=str(sales),
        dist=str(dist),
        out=str(out),
        contract_value_krw=2_000_000_000,
    )

    try:
        module.build_packet(args)
    except RuntimeError as exc:
        assert "sdist" in str(exc)
    else:
        raise AssertionError("missing source distribution should fail packet coverage")
